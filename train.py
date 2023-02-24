import os
import random
from datetime import datetime

import hydra
import numpy as np
import pytorch_lightning as zeus
import torch
from omegaconf import DictConfig, OmegaConf

from lgssl.datasets import build_loader
from lgssl.trainers.clip_trainer import CLIPTrainer
from lgssl.trainers.declipnns_trainer import DeCLIPNNSTrainer
from lgssl.trainers.lgslip_trainer import LG_SLIPTrainer
from lgssl.trainers.nnclr_trainer import NNCLRTrainer
from lgssl.trainers.simclr_trainer import SimCLRTrainer
from lgssl.trainers.simsiam_trainer import SimSiamTrainer
from lgssl.trainers.slip_trainer import SLIPTrainer
from lgssl.trainers.swav_trainer import SwAVTrainer

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


@hydra.main(config_name="train", config_path="./configs")
def main(cfg: DictConfig) -> None:

    # Reproducibility - refer https://pytorch.org/docs/stable/notes/randomness.html
    torch.manual_seed(cfg.system.random_seed)
    random.seed(cfg.system.random_seed)
    np.random.seed(cfg.system.random_seed)

    assert cfg.experiment.name != "", "Experiment name is not defined."
    exp_version = datetime.today().strftime("%m%d-%H%M")
    full_exp_name = f"{cfg.experiment.name}_{exp_version}"

    OmegaConf.set_struct(cfg, False)
    cfg.experiment.full_name = full_exp_name
    cfg.experiment.version = exp_version
    OmegaConf.set_struct(cfg, True)

    # setup checkpoint directory
    exp_dir = os.path.join(cfg.paths.experiments_dir, full_exp_name)
    try:
        os.makedirs(exp_dir)
    except:
        print(f"Path {exp_dir} exist. Overwriting since it probably just failed.")

    print("=====================================")
    print(f"Experiment name: {full_exp_name}")
    print()
    print(OmegaConf.to_yaml(cfg))
    print("=====================================")

    # Set up model
    print("Setting up trainer...")
    if cfg.model.trainer == "CLIP":
        model = CLIPTrainer(cfg)
    elif cfg.model.trainer == "SimCLR":
        model = SimCLRTrainer(cfg)
    elif cfg.model.trainer == "NNCLR":
        model = NNCLRTrainer(cfg)
    elif cfg.model.trainer == "SimSiam":
        model = SimSiamTrainer(cfg)
    elif cfg.model.trainer == "SwAV":
        model = SwAVTrainer(cfg)
    elif cfg.model.trainer == "DeCLIPNNS":
        model = DeCLIPNNSTrainer(cfg)
    elif cfg.model.trainer == "SLIP":
        model = SLIPTrainer(cfg)
    elif cfg.model.trainer == "LG_SLIP":
        model = LG_SLIPTrainer(cfg)
    else:
        raise ValueError(f"Unknown Trainer: {cfg.model.trainer}")

    # Handle multigpu training
    num_gpus = cfg.system.num_gpus if "num_gpus" in cfg.system else 1
    strategy = "ddp" if num_gpus > 1 else None

    print("Setting up datasets...")
    train_loader = build_loader(cfg.dataset, num_gpus=num_gpus)
    train_loader.dataset.__getitem__(0)
    print("Dataset setup done!")

    # Trainer Plugins
    checkpoint_callback = zeus.callbacks.ModelCheckpoint(
        dirpath=exp_dir,
        filename="checkpoint-{step:07d}",
        every_n_train_steps=cfg.optim.checkpoint_step,
        save_top_k=-1,
    )
    checkpoint_epoch_callback = zeus.callbacks.ModelCheckpoint(
        dirpath=exp_dir,
        filename="checkpoint-{epoch:03d}",
        every_n_epochs=1,
        save_top_k=-1,
    )
    logger = zeus.loggers.TensorBoardLogger(
        save_dir=cfg.paths.tensorboard_dir,
        name=cfg.experiment.name,
        version=cfg.experiment.version,
    )

    trainer = zeus.Trainer(
        gpus=num_gpus,
        strategy=strategy,
        benchmark=True,
        logger=logger,
        max_steps=cfg.optim.max_steps + 10,
        track_grad_norm=2,
        sync_batchnorm=cfg.optim.sync_batchnorm,
        precision=cfg.optim.precision,
        callbacks=[checkpoint_callback, checkpoint_epoch_callback],
    )
    print("Trainer setup done...")

    # get checkpoint if cfg.experiment has a resume variable
    prev_checkpoint = getattr(cfg.experiment, "resume", None)
    trainer.fit(model, train_loader, ckpt_path=prev_checkpoint)


if __name__ == "__main__":
    main()
