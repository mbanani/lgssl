from hydra.utils import instantiate
from lightly.models.modules import NNMemoryBankModule

from .base_trainer import BaseTrainer
from .losses import NNCLRLoss


class NNCLRTrainer(BaseTrainer):
    def __init__(self, cfg, **kwargs):
        super().__init__(cfg)

        # define model -- assume model includes encoder
        self.visual_backbone = instantiate(cfg.model.visual)
        self.proj_mlp = instantiate(cfg.model.visual_projection)
        self.pred_mlp = instantiate(cfg.model.visual_prediction)

        self.memory_bank = NNMemoryBankModule(size=cfg.model.queue_len)
        self.nnclr_loss = NNCLRLoss(temperature=cfg.model.temperature)

    def forward_step(self, batch, batch_idx, split):
        assert split != "test", "test is not implemented"

        # compute features
        image_0, image_1 = batch["image_0"], batch["image_1"]
        if "augmentation" in batch:
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

        # get/update nearest neighbors
        proj_0 = self.memory_bank(proj_0, update=False)
        proj_1 = self.memory_bank(proj_1, update=True)

        loss, acc = self.nnclr_loss(proj_0, proj_1, pred_0, pred_1)
        self.log(f"nnclr_loss/{split}", loss.item())
        self.log(f"nnclr_acc/{split}", acc.item())
        self.log(f"loss/{split}", loss.item())

        return loss
