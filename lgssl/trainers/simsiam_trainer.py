from hydra.utils import instantiate
from torch.nn.functional import cosine_similarity

from .base_trainer import BaseTrainer


class SimSiamTrainer(BaseTrainer):
    def __init__(self, cfg, **kwargs):
        super().__init__(cfg)

        # define model -- assume model includes encoder
        self.visual_backbone = instantiate(cfg.model.visual)
        self.proj_mlp = instantiate(cfg.model.visual_projection)
        self.pred_mlp = instantiate(cfg.model.visual_prediction)

        # results in worse performance for some reason ...
        # normalize projections -- similar to SimSiam implementation
        print(self)

    def forward_step(self, batch, batch_idx, split):
        assert split != "test", "test is not implemented"

        # compute features
        image_0, image_1 = batch["image_0"], batch["image_1"]
        image_0 = batch["augmentation"](image_0)
        image_1 = batch["augmentation"](image_1)

        feat_0 = self.visual_backbone(image_0)
        feat_1 = self.visual_backbone(image_1)

        proj_0 = self.proj_mlp(feat_0)
        proj_1 = self.proj_mlp(feat_1)

        pred_0 = self.pred_mlp(proj_0)
        pred_1 = self.pred_mlp(proj_1)

        # log features stats
        self.log_feature_stats(feat_0, "vis", 0, split)
        self.log_feature_stats(feat_1, "vis", 1, split)

        # compute loss
        loss_0 = cosine_similarity(proj_0.detach(), pred_1).mean()
        loss_1 = cosine_similarity(proj_1.detach(), pred_0).mean()
        loss = -0.5 * (loss_0 + loss_1)

        self.log(f"simsiam_loss/{split}", loss.item())
        self.log(f"loss/{split}", loss.item())

        return loss
