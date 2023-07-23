import argparse
import json
import logging
import os

import torch
import torch.distributed as dist
import torch.nn as nn
import wandb
import yaml
from diffusers import DDPMScheduler, UNet2DModel
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from data import create_dataloader, create_dataset
from focal_loss import FocalLoss
from log import setup_default_logging
from model import DiscriminativeSubNetwork
from scheduler import CosineAnnealingWarmupRestarts
from train import training
from utils import torch_seed

_logger = logging.getLogger("train")


def run(cfg):
    # setting seed and device
    setup_default_logging()
    torch_seed(cfg["SEED"])

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    _logger.info("Device: {}".format(device))

    # savedir
    cfg["EXP_NAME"] = cfg["EXP_NAME"] + f"-{cfg['DATASET']['target']}"
    savedir = os.path.join(cfg["RESULT"]["savedir"], cfg["EXP_NAME"])
    os.makedirs(savedir, exist_ok=True)

    # wandb
    if cfg["TRAIN"]["use_wandb"]:
        wandb.init(name=cfg["EXP_NAME"], project="Diffsuion_AD", config=cfg)

    # build datasets
    trainset = create_dataset(
        cfg["DATASET"]["name"],
        datadir=cfg["DATASET"]["datadir"],
        target=cfg["DATASET"]["target"],
        train=True,
        img_size=cfg["DATASET"]["resize"],
        self_aug=cfg["DATASET"]["self_aug"],
        normalize=cfg["DATASET"]["normalize"],
    )

    testset = create_dataset(
        cfg["DATASET"]["name"],
        datadir=cfg["DATASET"]["datadir"],
        target=cfg["DATASET"]["target"],
        train=False,
        img_size=cfg["DATASET"]["resize"],
        self_aug=cfg["DATASET"]["self_aug"],
        normalize=cfg["DATASET"]["normalize"],
    )

    # build dataloader
    trainloader = create_dataloader(
        dataset=trainset,
        is_training=True,
        batch_size=cfg["DATALOADER"]["batch_size"],
        num_workers=cfg["DATALOADER"]["num_workers"],
    )

    testloader = create_dataloader(
        dataset=testset,
        is_training=False,
        batch_size=cfg["DATALOADER"]["batch_size"],
        num_workers=cfg["DATALOADER"]["num_workers"],
    )

    # build model
    ddpm_scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_schedule="linear",
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
        block_out_channels=(128, 256, 512, 1024, 2048),
        attention_head_dim=4,
    )
    # Segmentation Sub-network
    segment_subnet = DiscriminativeSubNetwork(in_channels=6, out_channels=2).to(
        cfg["device"]
    )
    # Set training
    mse_criterion = nn.MSELoss()
    sml1_criterion = nn.SmoothL1Loss()
    focal_criterion = FocalLoss(
        gamma=cfg["TRAIN"]["focal_gamma"], alpha=cfg["TRAIN"]["focal_alpha"]
    )

    params_to_optimize = []
    params_to_optimize += list(
        filter(lambda p: p.requires_grad, denosing_subnet.parameters())
    )
    params_to_optimize += list(
        filter(lambda p: p.requires_grad, segment_subnet.parameters())
    )

    optimizer = torch.optim.Adam(
        params=params_to_optimize,
        lr=cfg["OPTIMIZER"]["lr"],
        weight_decay=cfg["OPTIMIZER"]["weight_decay"],
    )
    if cfg["SCHEDULER"]["use_scheduler"]:
        scheduler = CosineAnnealingWarmupRestarts(
            optimizer,
            first_cycle_steps=cfg.TRAIN.num_training_steps,
            max_lr=cfg.OPTIMIZER.lr,
            min_lr=cfg.SCHEDULER.min_lr,
            warmup_steps=int(cfg.TRAIN.num_training_steps * cfg.SCHEDULER.warmup_ratio),
        )
    else:
        scheduler = None

    # Fitting model
    training(
        rank=0,
        world_size=torch.cuda.device_count(),
        diffusion_scheduler=ddpm_scheduler,
        model=[denoising_subnet, segment_subnet],
        num_training_steps=cfg["TRAIN"]["num_training_steps"],
        trainloader=trainloader,
        validloader=testloader,
        criterion=[mse_criterion, sml1_criterion, focal_criterion],
        loss_weights=[cfg["TRAIN"]["l1_weight"], cfg["TRAIN"]["focal_weight"]],
        optimizer=optimizer,
        scheduler=scheduler,
        log_interval=cfg["LOG"]["log_interval"],
        eval_interval=cfg["LOG"]["eval_interval"],
        savedir=savedir,
        device=device,
        use_wandb=cfg["TRAIN"]["use_wandb"],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diffusion AD")
    parser.add_argument("--yaml_config", type=str, default=None, help="exp config file")

    args = parser.parse_args()

    # config
    cfg = yaml.load(open(args.yaml_config, "r"), Loader=yaml.FullLoader)

    run(cfg)
