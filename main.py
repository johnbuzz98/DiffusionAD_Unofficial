import argparse
import logging
import os

import torch
import torch.nn as nn
import wandb
import yaml
from accelerate import Accelerator, DistributedType
from diffusers import DDPMScheduler, UNet2DModel

from data import create_dataloader, create_dataset
from focal_loss import FocalLoss
from log import setup_default_logging
from model import DiscriminativeSubNetwork
from scheduler import CosineAnnealingWarmupRestarts
from train import training
from utils import torch_seed

_logger = logging.getLogger("train")


def run(cfg):
    setup_default_logging()
    torch_seed(cfg["SEED"])

    accelerator = Accelerator(fp16=cfg.get("USE_FP16", False))
    device = accelerator.device
    _logger.info(f"Device: {device}")

    cfg["EXP_NAME"] = f"{cfg['EXP_NAME']}-{cfg['DATASET']['target']}"
    savedir = os.path.join(cfg["RESULT"]["savedir"], cfg["EXP_NAME"])
    os.makedirs(savedir, exist_ok=True)

    if cfg["TRAIN"]["use_wandb"]:
        import wandb

        wandb.init(name=cfg["EXP_NAME"], project="Diffsuion_AD", config=cfg)

    dataset_args = {
        key: cfg["DATASET"][key]
        for key in ["name", "datadir", "target", "resize", "self_aug", "normalize"]
    }

    datasets = [
        create_dataset(train=train_flag, **dataset_args) for train_flag in [True, False]
    ]

    dataloader_args = {
        key: cfg["DATALOADER"][key] for key in ["batch_size", "num_workers"]
    }

    dataloaders = [
        create_dataloader(dataset=dataset, is_training=is_training, **dataloader_args)
        for dataset, is_training in zip(datasets, [True, False])
    ]

    trainloader, testloader = dataloaders

    ddpm_scheduler = (
        DDPMScheduler(
            num_train_timesteps=1000, beta_schedule="linear", prediction_type="epsilon"
        ),
    )

    denoising_subnet = UNet2DModel(
        sample_size=(cfg["DATASET"]["resize"], cfg["DATASET"]["resize"]),
        down_block_types=(
            "DownBlock2D",
            "DownBlock2D",
            "AttnDownBlock2D",
            "AttnDownBlock2D",
            "AttnDownBlock2D",
        ),
        up_block_types=(
            "AttnUpBlock2D",
            "AttnUpBlock2D",
            "AttnUpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        ),
        block_out_channels=(
            128,
            256,
            256,
            512,
            512,
        ),
        attention_head_dim=4,
    ).to(device)

    segment_subnet = DiscriminativeSubNetwork(in_channels=6, out_channels=2).to(device)

    criterions = [
        nn.MSELoss(),
        nn.SmoothL1Loss(),
        FocalLoss(gamma=cfg["TRAIN"]["focal_gamma"], alpha=cfg["TRAIN"]["focal_alpha"]),
    ]

    params_to_optimize = [
        p
        for model in [denoising_subnet, segment_subnet]
        for p in model.parameters()
        if p.requires_grad
    ]

    optimizer = torch.optim.Adam(
        params=params_to_optimize,
        lr=cfg["OPTIMIZER"]["lr"],
        weight_decay=cfg["OPTIMIZER"]["weight_decay"],
    )

    scheduler = (
        CosineAnnealingWarmupRestarts(
            optimizer,
            first_cycle_steps=cfg["TRAIN"]["num_training_steps"],
            max_lr=cfg["OPTIMIZER"]["lr"],
            min_lr=cfg["SCHEDULER"]["min_lr"],
            warmup_steps=int(
                cfg["TRAIN"]["num_training_steps"] * cfg["SCHEDULER"]["warmup_ratio"]
            ),
        )
        if cfg["SCHEDULER"]["use_scheduler"]
        else None
    )

    # prepare to acclelerate
    (
        trainloader,
        testloader,
        denoising_subnet,
        segment_subnet,
        optimizer,
    ) = accelerator.prepare(
        trainloader, testloader, denoising_subnet, segment_subnet, optimizer
    )
    training(
        diffusion_scheduler=ddpm_scheduler,
        model=[denoising_subnet, segment_subnet],
        num_training_steps=cfg["TRAIN"]["num_training_steps"],
        trainloader=trainloader,
        validloader=testloader,
        criterion=criterions,
        loss_weights=[cfg["TRAIN"]["l1_weight"], cfg["TRAIN"]["focal_weight"]],
        optimizer=optimizer,
        scheduler=scheduler,
        log_interval=cfg["LOG"]["log_interval"],
        eval_interval=cfg["LOG"]["eval_interval"],
        savedir=savedir,
        device=device,
        use_wandb=cfg["TRAIN"]["use_wandb"],
        accelerator=accelerator,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diffusion AD")
    parser.add_argument(
        "--yaml_config", type=str, required=True, help="exp config file"
    )
    args = parser.parse_args()

    with open(args.yaml_config, "r") as f:
        cfg = yaml.safe_load(f)

    run(cfg)
