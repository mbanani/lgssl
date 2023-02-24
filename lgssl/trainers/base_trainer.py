import gc
import time

import pytorch_lightning as zeus
import torch
from hydra.utils import instantiate
from torch.nn.functional import normalize


class BaseTrainer(zeus.LightningModule):
    def __init__(self, cfg):
        super().__init__()

        # define hyperparameters
        self.cfg = cfg
        self.save_hyperparameters()
        self.tracker = {}
        self.track_s = 0

    def forward_step(self, batch, batch_idx, split):
        # This method should be implemented by specific trainers
        raise NotImplementedError()

    def log_feature_stats(self, feat, feat_type, view_num, split):
        std = normalize(feat, dim=1).std(dim=1).mean().item()
        std_raw = feat.std(dim=1).mean().item()
        self.log(f"{feat_type}_feature_std_{view_num}/{split}", std)
        self.log(f"{feat_type}_feature_std_{view_num}/raw_{split}", std_raw)

    # ========= Boiler Plate ==========
    def training_step(self, batch, batch_idx):
        return self.forward_step(batch, batch_idx, "train")

    def configure_optimizers(self):
        optim_cfg = self.cfg.optim.optimizer
        optim_cfg.params.model = self
        optimizer = instantiate(optim_cfg)

        lr_scheduler = instantiate(self.cfg.optim.scheduler, optimizer=optimizer)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": lr_scheduler, "interval": "step"},
        }

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        state_dict = checkpoint["state_dict"]
        model_state_dict = self.state_dict()
        is_changed = False
        for k in state_dict:
            if k in model_state_dict:
                if state_dict[k].shape != model_state_dict[k].shape:
                    print(
                        f"Skip loading parameter: {k}, "
                        f"required shape: {model_state_dict[k].shape}, "
                        f"loaded shape: {state_dict[k].shape}"
                    )
                    state_dict[k] = model_state_dict[k]
                    is_changed = True
            else:
                print(f"Dropping parameter {k}")
                is_changed = True

        if is_changed:
            checkpoint.pop("optimizer_states", None)

    def on_train_epoch_start(self):
        self.BATCH_END = time.time()

    def on_before_batch_transfer(self, batch, dataloader_idx):
        self.AFTER_DATA_LOAD = time.time()
        return batch

    def on_after_batch_transfer(self, batch, dataloader_idx):
        self.AFTER_DATA_TRAN = time.time()
        return batch

    def on_before_backward(self, loss):
        self.AFTER_FORWARD = time.time()

    def on_after_backward(self):
        self.AFTER_BACKWARD = time.time()

    def on_before_optimizer_step(self, optimizer, optimizer_idx):
        for i, p_group in enumerate(optimizer.param_groups):
            self.log(
                f"optimizer/params_{i}_lr",
                p_group["lr"],
                on_epoch=False,
                rank_zero_only=True,
            )

    def on_train_batch_end(self, outputs, batch, batch_idx):
        batch_end = time.time()
        data_load = self.AFTER_DATA_LOAD - self.BATCH_END
        data_tran = self.AFTER_DATA_TRAN - self.AFTER_DATA_LOAD
        batch_fwd = self.AFTER_FORWARD - self.AFTER_DATA_TRAN
        batch_bck = self.AFTER_BACKWARD - self.AFTER_FORWARD
        batch_opt = batch_end - self.AFTER_BACKWARD
        batch_total = batch_end - self.BATCH_END

        self.log("time/data_loading", data_load)
        self.log("time/data_transfer", data_tran)
        self.log("time/forward", batch_fwd)
        self.log("time/backward", batch_bck)
        self.log("time/optim", batch_opt)
        self.log("time/total", batch_total)

        self.BATCH_END = time.time()
        self.track_s += 1
        garbage_collection = False

        if garbage_collection:
            num_objs = 0
            new_track = {}

            for obj in gc.get_objects():
                try:
                    check_data = hasattr(obj, "data") and torch.is_tensor(obj.data)
                    if torch.is_tensor(obj) or check_data:
                        num_objs += 1
                        obj_size = tuple(obj.size())
                        if obj_size in new_track:
                            new_track[obj_size] += 1
                        else:
                            new_track[obj_size] = 1
                except:
                    pass

            print(f"Number of gc objects: {num_objs}")
            print("diff:")
            for osize in new_track:
                if osize in self.tracker:
                    diff = new_track[osize] - self.tracker[osize]
                else:
                    diff = new_track[osize]
                if diff > 0:
                    print(f"{osize} -- {diff}")

            for osize in self.tracker:
                if osize not in new_track:
                    print(f"{osize} -- -{self.tracker[osize]}")

            self.tracker = new_track

            gc.collect()
