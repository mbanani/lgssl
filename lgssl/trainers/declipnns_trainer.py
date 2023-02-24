import clip
import numpy as np
import torch
from hydra.utils import instantiate
from lightly.models.modules import NNMemoryBankModule

from .base_trainer import BaseTrainer
from .losses import CLIPLoss


class DeCLIPNNSTrainer(BaseTrainer):
    def __init__(self, cfg, **kwargs):
        super().__init__(cfg)

        self.visual_backbone = instantiate(cfg.model.visual)
        self.language_backbone = instantiate(cfg.model.language)
        self.language_model = "CLIP"  # hacky!

        self.visual_project = instantiate(cfg.model.visual_projection)
        self.language_project = instantiate(cfg.model.language_projection)

        # define losses -- logit scale is learnable
        self.clip_loss = CLIPLoss()
        self.clip_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        # get NNCLR bank
        self.memory_bank = NNMemoryBankModule(size=16384)

        self.clip_weight = cfg.model.clip_weight
        self.nns_weight = cfg.model.nns_weight

    def encode_text(self, caption):
        if self.language_model == "frozen_sbert":
            # tokenizer is included within SBERT model
            lang_feat = self.language_backbone.encode(
                caption,
                batch_size=len(caption),
                show_progress_bar=False,
                convert_to_numpy=False,
            )

            lang_feat = torch.stack(lang_feat, dim=0).detach()
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

        # get/update nearest neighbors
        lang_nn = self.memory_bank(lang_view, update=True)

        # log features stats
        self.log_feature_stats(vis_feat, "vis", 0, split)
        self.log_feature_stats(lang_feat, "lang", 0, split)

        # compute losses -- scale is kept in log-space for stability?!
        loss_clip, acc_clip = self.clip_loss(vis_view, lang_view, self.clip_scale.exp())
        loss_nns, acc_nns = self.clip_loss(vis_view, lang_nn, self.clip_scale.exp())

        # log metrics
        self.log(f"clip_loss/{split}", loss_clip.item())
        self.log(f"clip_acc/{split}", acc_clip.item())

        self.log(f"declip_nns_loss/{split}", loss_nns.item())
        self.log(f"declip_nns_acc/{split}", acc_nns.item())

        loss = self.clip_weight * loss_clip + self.nns_weight * loss_nns
        self.log(f"loss/{split}", loss.item())

        return loss
