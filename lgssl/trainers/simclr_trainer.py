from hydra.utils import instantiate

from .base_trainer import BaseTrainer
from .losses import SimCLRLoss


class SimCLRTrainer(BaseTrainer):
    def __init__(self, cfg, **kwargs):
        super().__init__(cfg)
        self.visual_backbone = instantiate(cfg.model.visual)
        self.visual_project = instantiate(cfg.model.visual_projection)

        # define losses -- 0.1 is temp default in SimCLR repo
        self.simclr_loss = SimCLRLoss(0.1)

    def forward_step(self, batch, batch_idx, split):
        assert split != "test", "test is not implemented"

        # compute features
        image_0, image_1 = batch["image_0"], batch["image_1"]

        # run additional augmentations on GPU -- faster...
        image_0 = batch["augmentation"](image_0)
        image_1 = batch["augmentation"](image_1)

        feat_0 = self.visual_backbone(image_0)
        feat_1 = self.visual_backbone(image_1)
        view_0 = self.visual_project(feat_0)
        view_1 = self.visual_project(feat_1)

        loss, acc = self.simclr_loss(view_0, view_1)

        # log everything
        self.log_feature_stats(feat_0, "vis", 0, split)
        self.log_feature_stats(feat_1, "vis", 1, split)
        self.log(f"simclr_loss/{split}", loss.item())
        self.log(f"simclr_acc/{split}", acc.item())
        self.log(f"loss/{split}", loss.item())

        del batch
        return loss
