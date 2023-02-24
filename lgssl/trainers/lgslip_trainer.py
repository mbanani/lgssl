import clip
import numpy as np
import torch
from hydra.utils import instantiate

from .base_trainer import BaseTrainer
from .losses import CLIPLoss, SimCLRLoss


class LG_SLIPTrainer(BaseTrainer):
    def __init__(self, cfg, **kwargs):
        super().__init__(cfg)

        self.visual_backbone = instantiate(cfg.model.visual)
        self.language_backbone = instantiate(cfg.model.language)
        self.language_model = getattr(cfg.model, "language_model", "CLIP")

        self.visual_project = instantiate(cfg.model.visual_projection)
        self.language_project = instantiate(cfg.model.language_projection)

        # define losses -- logit scale is learnable
        self.clip_loss = CLIPLoss()
        self.clip_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.simclr_loss = SimCLRLoss(0.1)

        self.ssl_scale = 1.0  # using value from LG_SLIP repo

    def encode_text(self, caption):
        if self.language_model == "frozen_sbert":
            # tokenizer is included within SBERT model
            lang_feat = self.language_backbone.encode(
                caption,
                batch_size=len(caption),
                show_progress_bar=False,
                convert_to_numpy=False,
                device=self.device,
            )
            lang_feat = torch.stack(lang_feat, dim=0).detach().to(self.device)
        elif self.language_model != "none":
            cap_token = clip.tokenize(caption, truncate=True).to(self.device)
            lang_feat = self.language_backbone(cap_token, self.device)
        else:
            raise ValueError(f"Unknown Language Model: {self.language_model}")

        return lang_feat

    def forward_step(self, batch, batch_idx, split):
        assert split != "test", "test is not implemented"

        # compute features
        image_0 = batch["image_0"]
        image_1 = batch["image_1"]

        image_0 = batch["augmentation"](image_0)
        image_1 = batch["augmentation"](image_1)

        # visual features
        vis_feat_0 = self.visual_backbone(image_0)
        vis_feat_1 = self.visual_backbone(image_1)
        vis_view_0 = self.visual_project(vis_feat_0)
        vis_view_1 = self.visual_project(vis_feat_1)

        # language features
        lang_feat_0 = self.encode_text(batch["caption_0"])
        lang_view_0 = self.language_project(lang_feat_0)
        # lang_feat_1 = self.encode_text(batch["caption_1"])
        # lang_view_1 = self.language_project(lang_feat_1)

        # log features stats
        self.log_feature_stats(vis_feat_0, "vis", 0, split)
        self.log_feature_stats(vis_feat_1, "vis", 1, split)
        self.log_feature_stats(lang_feat_0, "lang", 0, split)
        # self.log_feature_stats(lang_feat_1, "lang", 1, split)

        # compute losses -- scale is kept in log-space for stability?!
        loss_c, acc_c = self.clip_loss(vis_view_0, lang_view_0, self.clip_scale.exp())
        # loss_c1, acc_c1 = self.clip_loss(
        #   vis_view_1, lang_view_1, self.clip_scale.exp()
        # )
        loss_s, acc_s = self.simclr_loss(vis_view_0, vis_view_1)

        # log metrics
        self.log(f"clip_loss/{split}", loss_c.item())
        self.log(f"clip_acc/{split}", acc_c.item())
        self.log(f"simclr_loss/{split}", loss_s.item())
        self.log(f"simclr_acc/{split}", acc_s.item())

        loss = self.ssl_scale * loss_s + loss_c
        self.log(f"loss/{split}", loss.item())

        return loss
