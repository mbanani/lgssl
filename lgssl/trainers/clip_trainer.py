import clip
import numpy as np
import torch
from hydra.utils import instantiate

from .base_trainer import BaseTrainer
from .losses import CLIPLoss


class CLIPTrainer(BaseTrainer):
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
        image = batch["image_0"]
        if "augmentation" in batch:
            image = batch["augmentation"](image)

        # visual features
        vis_feat = self.visual_backbone(image)
        vis_view = self.visual_project(vis_feat)

        # language features
        lang_feat = self.encode_text(batch["caption_0"])
        lang_view = self.language_project(lang_feat)

        # log features stats
        self.log_feature_stats(vis_feat, "vis", 0, split)
        self.log_feature_stats(lang_feat, "lang", 0, split)

        # compute losses -- scale is kept in log-space for stability?!
        loss, acc = self.clip_loss(vis_view, lang_view, self.clip_scale.exp())

        # log metrics
        self.log(f"clip_loss/{split}", loss.item())
        self.log(f"clip_acc/{split}", acc.item())
        self.log(f"loss/{split}", loss.item())

        return loss
