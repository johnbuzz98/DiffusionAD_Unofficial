import argparse
import json
import logging
import os

import torch
import torch.nn as nn
import wandb
import yaml
from scheduler import CosineAnnealingWarmupRestarts

from data import create_dataloader, create_dataset
from focal_loss import FocalLoss
from log import setup_default_logging
from model import DenoiseDiffusion, DiscriminativeSubNetwork, UNet
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
        texture_source_dir=cfg["DATASET"]["texture_source_dir"],
        grid_size=cfg["DATASET"]["structure_grid_size"],
        transparency_range=cfg["DATASET"]["transparency_range"],
        perlin_scale=cfg["DATASET"]["perlin_scale"],
        min_perlin_scale=cfg["DATASET"]["min_perlin_scale"],
        perlin_noise_threshold=cfg["DATASET"]["perlin_noise_threshold"],
        textual_or_structural=cfg["DATASET"]["textual_or_structural"],
        self_aug=cfg["DATASET"]["self_aug"],
    )

    testset = create_dataset(
        cfg["DATASET"]["name"],
        datadir=cfg["DATASET"]["datadir"],
        target=cfg["DATASET"]["target"],
        train=False,
        img_size=cfg["DATASET"]["resize"],
        texture_source_dir=cfg["DATASET"]["texture_source_dir"],
        grid_size=cfg["DATASET"]["structure_grid_size"],
        transparency_range=cfg["DATASET"]["transparency_range"],
        perlin_scale=cfg["DATASET"]["perlin_scale"],
        min_perlin_scale=cfg["DATASET"]["min_perlin_scale"],
        perlin_noise_threshold=cfg["DATASET"]["perlin_noise_threshold"],
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

    # Denoising Sub-network
    # epsmodel
    eps_model = UNet(
        image_channels=cfg["Denoising"]["image_channels"],  # 3
        n_channels=cfg["Denoising"]["n_channels"],  # 128
        ch_mults=cfg["Denoising"]["channel_multipliers"],  # [1, 2, 2, 2, 2]
        is_attn=cfg["Denoising"]["is_attention"],  # [False, Fa;lse, True, True, True]
        n_blocks=cfg["Denoising"]["n_blocks"],  # 2
    ).to(cfg["device"])

    denosing_subnet = DenoiseDiffusion(
        eps_model=eps_model,
        n_steps=cfg["Denoising"]["n_steps"],
        device=cfg["device"],
    ).to(cfg["device"])

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
    # Fitting model
    training(
        model=[denosing_subnet, segment_subnet],
        num_training_steps=cfg["TRAIN"]["num_training_steps"],
        trainloader=trainloader,
        validloader=testloader,
        criterion=[mse_criterion, sml1_criterion, focal_criterion],
        loss_weights=[cfg["TRAIN"]["l1_weight"], cfg["TRAIN"]["focal_weight"]],
        optimizer=optimizer,
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
