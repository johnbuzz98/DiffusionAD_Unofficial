import argparse
import logging
import os

import torch
import torch.nn as nn
from accelerate import Accelerator
from diffusers import UNet2DModel
from omegaconf import OmegaConf

from data import create_dataloader, create_dataset
from focal_loss import FocalLoss
from log import setup_default_logging
from model import DDPMScheduler, DiscriminativeSubNetwork
from scheduler import CosineAnnealingWarmupRestarts
from secrets_ import secrets as sc
from train import training
from utils import torch_seed

_logger = logging.getLogger("train")


def run(conf):
    setup_default_logging()
    torch_seed(conf.SEED)

    accelerator = Accelerator(log_with="wandb" if conf.TRAIN.use_wandb else None)
    device = accelerator.device
    conf.DATALOADER.num_workers = accelerator.num_processes
    _logger.info(f"Device: {device}")

    savedir = os.path.join(conf["RESULT"]["savedir"], conf["EXP_NAME"])
    os.makedirs(savedir, exist_ok=True)

    if conf.TRAIN.use_wandb:
        accelerator.init_trackers(
            project_name="Diffsuion_AD",
            init_kwargs={
                "wandb": {
                    "entity": sc["wandb_entity"],
                    "name": conf.EXP_NAME,
                }
            },
        )

    dataset_args = {
        key: conf["DATASET"][key]
        for key in ["name", "datadir", "target", "img_size", "self_aug", "normalize"]
    }

    datasets = [
        create_dataset(train=train_flag, **dataset_args) for train_flag in [True, False]
    ]

    dataloader_args = {
        key: conf["DATALOADER"][key] for key in ["batch_size", "num_workers"]
    }

    dataloaders = [
        create_dataloader(dataset=dataset, is_training=is_training, **dataloader_args)
        for dataset, is_training in zip(datasets, [True, False])
    ]

    trainloader, testloader = dataloaders[0], dataloaders[1]

    ddpm_scheduler = DDPMScheduler(
        num_train_timesteps=1000, beta_schedule="linear", prediction_type="epsilon"
    )

    denoising_subnet = UNet2DModel(
        sample_size=(conf.DATASET.img_size, conf.DATASET.img_size),
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
        FocalLoss(gamma=conf.TRAIN.focal_gamma, alpha=conf.TRAIN.focal_alpha),
    ]

    params_to_optimize = [
        p
        for model in [denoising_subnet, segment_subnet]
        for p in model.parameters()
        if p.requires_grad
    ]

    optimizer = torch.optim.Adam(
        params=params_to_optimize,
        lr=conf.OPTIMIZER.lr,
        weight_decay=conf.OPTIMIZER.weight_decay,
    )
    conf.TRAIN.num_training_steps = len(trainloader) * conf.TRAIN.num_epochs

    scheduler = (
        CosineAnnealingWarmupRestarts(
            optimizer,
            first_cycle_steps=conf.TRAIN.num_training_steps,
            max_lr=conf.OPTIMIZER.lr,
            min_lr=conf.SCHEDULER.min_lr,
            warmup_steps=int(
                conf.TRAIN.num_training_steps * conf.SCHEDULER.warmup_ratio
            ),
        )
        if conf.SCHEDULER.use_scheduler
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
        num_epochs=conf.TRAIN.num_epochs,
        num_training_steps=conf.TRAIN.num_training_steps,
        trainloader=trainloader,
        validloader=testloader,
        criterion=criterions,
        loss_weights=[
            conf.TRAIN.mse_weight,
            conf.TRAIN.sml1_weight,
            conf.TRAIN.focal_weight,
        ],
        optimizer=optimizer,
        scheduler=scheduler,
        log_interval=conf.LOG.log_interval,
        eval_interval=conf.LOG.eval_interval,
        savedir=savedir,
        device=device,
        use_wandb=conf.TRAIN.use_wandb,
        accelerator=accelerator,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diffusion AD")
    parser.add_argument(
        "--yaml_config", type=str, required=True, help="exp config file"
    )
    args = parser.parse_args()

    conf = OmegaConf.load(args.yaml_config)

    run(conf)
