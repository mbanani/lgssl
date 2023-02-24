import time

import torch
import torch.distributed as dist
from hydra.utils import instantiate
from torch.nn.functional import log_softmax, normalize

from lgssl.trainers.base_trainer import BaseTrainer
from lgssl.utils.distributed import get_world_size


class SwAVTrainer(BaseTrainer):
    def __init__(self, cfg, **kwargs):
        super().__init__(cfg)
        self.visual_backbone = instantiate(cfg.model.visual)
        self.visual_project = instantiate(cfg.model.visual_projection)
        self.prototypes = instantiate(cfg.model.prototypes)
        self.freeze_proto = cfg.model.freeze_prototype_iters
        self.no_queue_steps = cfg.model.queue.start_steps
        self.queue_len = self.cfg.model.queue.length
        self.temp = cfg.model.temperature

        self.queue = None

    def create_queue(self):
        world_size = get_world_size()
        self.queue = torch.zeros(
            2, self.cfg.model.queue.length // world_size, self.cfg.model.queue.feat_dim
        ).to(self.device)

        self.queue_initialized = False

    def on_after_backward(self):
        # for timing
        self.AFTER_BACKWARD = time.time()

        # deal with freeze iter stuff
        if self.global_step < self.freeze_proto:
            self.prototypes.weight.grad = None

        # create queue after some number of epochs
        use_queue = self.queue_len > 0 and self.global_step >= self.no_queue_steps
        if self.queue is None and use_queue:
            print("********* initialize queue ********")
            self.create_queue()

    def forward_step(self, batch, batch_idx, split):
        assert split != "test", "test is not implemented"

        # normalize the prototypes -- from SwAV
        with torch.no_grad():
            w = self.prototypes.weight.data.clone()
            w = normalize(w, dim=1, p=2)
            self.prototypes.weight.copy_(w)

        # compute features
        image_0, image_1 = batch["image_0"], batch["image_1"]
        image_0 = batch["augmentation"](image_0)
        image_1 = batch["augmentation"](image_1)

        feat_0 = self.visual_backbone(image_0)
        feat_1 = self.visual_backbone(image_1)

        # log features stats
        self.log_feature_stats(feat_0, "vis", 0, split)
        self.log_feature_stats(feat_1, "vis", 1, split)

        # === following SwAV repo ===
        # project and malize embeddings
        view_0 = self.visual_project(feat_0)
        view_1 = self.visual_project(feat_1)

        # compute prototypes and detach views (not needed since there's no queue)
        view_0 = normalize(view_0, dim=1, p=2)
        view_1 = normalize(view_1, dim=1, p=2)
        p_0 = self.prototypes(view_0)
        p_1 = self.prototypes(view_1)

        # compute cluster assignments
        q_0 = self.compute_cluster_assignments(p_0, view_0, 0)
        q_1 = self.compute_cluster_assignments(p_1, view_1, 1)

        # compute loss
        loss_01 = -1 * (q_0 * log_softmax(p_1 / self.temp, dim=0)).sum(dim=1).mean()
        loss_10 = -1 * (q_1 * log_softmax(p_0 / self.temp, dim=0)).sum(dim=1).mean()
        loss = (loss_01 + loss_10) * 0.5

        self.log(f"loss/{split}", loss.item())

        return loss

    def compute_cluster_assignments(self, proto_feats, embed_feats, index):
        bs = proto_feats.shape[0]
        proto_feats = proto_feats.detach()

        with torch.no_grad():
            if self.queue is not None:
                still_zeros = torch.all(self.queue[index, -1, :] == 0)
                if self.queue_initialized or not still_zeros:
                    self.queue_initialized = True

                    # compute queue prototypes
                    out_q = self.prototypes(self.queue[index])
                    proto_feats = torch.cat((out_q, proto_feats))

                # Fill Queue:
                # - move first 0:N-1 batche slots to 1:N
                # - update first slot
                self.queue[index, bs:] = self.queue[index, :-bs].clone()
                self.queue[index, :bs] = embed_feats.detach()

            q_i = self.distributed_sinkhorn(proto_feats)[-bs:]
            assert q_i.shape[0] == bs

        return q_i

    @torch.no_grad()
    def distributed_sinkhorn(self, out, sinkhorn_iterations=3, epsilon=0.05):
        world_size = get_world_size()

        # Q is K-by-B for consistency with notations from our paper
        Q = torch.exp(out / epsilon).t()
        B = Q.shape[1] * world_size  # number of samples to assign
        K = Q.shape[0]  # how many prototypes

        # make the matrix sums to 1
        sum_Q = torch.sum(Q)
        if world_size > 1:
            dist.all_reduce(sum_Q)
        Q /= sum_Q

        for it in range(sinkhorn_iterations):
            # normalize each row: total weight per prototype must be 1/K
            sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
            if world_size > 1:
                dist.all_reduce(sum_of_rows)
            Q /= sum_of_rows
            Q /= K

            # normalize each column: total weight per sample must be 1/B
            Q /= torch.sum(Q, dim=0, keepdim=True)
            Q /= B

        Q *= B  # the colomns must sum to 1 so that Q is an assignment
        return Q.t()
