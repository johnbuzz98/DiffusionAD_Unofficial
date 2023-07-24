import json
import logging
import os
import time
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator, DistributedType
from anomalib.utils.metrics import AUPRO, AUROC

import wandb

_logger = logging.getLogger("train")


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def training(
    diffusion_scheduler,
    model,
    trainloader,
    validloader,
    criterion,
    optimizer,
    scheduler,
    num_training_steps: int = 1000,
    loss_weights: List[float] = [0.5, 0.5, 0.5],
    log_interval: int = 1,
    eval_interval: int = 1,
    savedir: str = None,
    device: str = "cpu",
    use_wandb: bool = False,
    accelerator: Accelerator = None,
) -> dict:
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    losses_m = AverageMeter()
    mse_losses_m = AverageMeter()
    sml1_losses_m = AverageMeter()
    focal_losses_m = AverageMeter()

    # metrics
    auroc_image_metric = AUROC(num_classes=1, pos_label=1)
    auroc_pixel_metric = AUROC(num_classes=1, pos_label=1)
    aupro_pixel_metric = AUPRO()

    # criterion
    mse_criterion, sml1_criterion, focal_criterion = criterion
    mse_weight, sml1_weight, focal_weight = loss_weights

    # set train mode
    denoising_subnet, segmentation_subnet = model
    denoising_subnet.train()
    segmentation_subnet.train()

    # set optimizer
    optimizer.zero_grad()

    # training
    best_score = 0
    step = 0
    train_mode = True
    while train_mode:
        end = time.time()
        for inputs, masks, targets in trainloader:
            # batch
            inputs, masks, targets = (
                inputs.to(device),
                masks.to(device),
                targets.to(device),
            )

            data_time_m.update(time.time() - end)

            # Denoising Loss (MSE)
            noise = torch.randn(inputs.shape, dtype=(torch.float32)).to(device)
            timesteps = torch.randint(100, 200, (inputs.shape[0],)).long().to(device)
            noisy_images = diffusion_scheduler.add_noise(inputs, noise, timesteps)
            noise_pred = denoising_subnet(noisy_images, timesteps).sample
            mse_loss = mse_criterion(noise_pred, noise)

            # One Step Denoising
            denoised_result = diffusion_scheduler.onestep_denoise(
                inputs, noise, timesteps, noise_pred
            ).to(device)

            joined_in = torch.cat((inputs, denoised_result), dim=1)
            out_mask = segmentation_subnet(joined_in)
            out_mask_sm = torch.softmax(out_mask, dim=1)

            sml1_loss = sml1_criterion(out_mask_sm[:, 1, :], masks)
            focal_loss = focal_criterion(out_mask_sm, masks)
            loss = (mse_weight * mse_loss) + (1 - mse_weight) * (
                (sml1_weight * sml1_loss) + (focal_weight * focal_loss)
            )

            # loss.backward()
            accelerator.backward(loss)
            # update weight
            optimizer.step()
            optimizer.zero_grad()

            # log loss
            mse_losses_m.update(mse_loss.item())
            sml1_losses_m.update(sml1_loss.item())
            focal_losses_m.update(focal_loss.item())
            losses_m.update(loss.item())

            batch_time_m.update(time.time() - end)

            # wandb
            if use_wandb:
                accelerator.log(
                    {
                        "lr": optimizer.param_groups[0]["lr"],
                        "train_mse_loss": mse_losses_m.val,
                        "train_sml1_loss": sml1_losses_m.val,
                        "train_focal_loss": focal_losses_m.val,
                        "train_loss": losses_m.val,
                    },
                    step=step,
                )

            if (step + 1) % log_interval == 0 or step == 0:
                _logger.info(
                    "TRAIN [{:>4d}/{}] "
                    "Loss: {loss.val:>6.4f} ({loss.avg:>6.4f}) "
                    "MSE Loss: {mse_loss.val:>6.4f} ({mse_loss.avg:>6.4f}) "
                    "Smooth L1 Loss: {sml1_loss.val:>6.4f} ({sml1_loss.avg:>6.4f}) "
                    "Focal Loss: {focal_loss.val:>6.4f} ({focal_loss.avg:>6.4f}) "
                    "LR: {lr:.3e} "
                    "Time: {batch_time.val:.3f}s, {rate:>7.2f}/s ({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s) "
                    "Data: {data_time.val:.3f} ({data_time.avg:.3f})".format(
                        step + 1,
                        num_training_steps,
                        loss=losses_m,
                        mse_loss=mse_losses_m,
                        sml1_loss=sml1_losses_m,
                        focal_loss=focal_losses_m,
                        lr=optimizer.param_groups[0]["lr"],
                        batch_time=batch_time_m,
                        rate=inputs.size(0) / batch_time_m.val,
                        rate_avg=inputs.size(0) / batch_time_m.avg,
                        data_time=data_time_m,
                    )
                )

            if ((step + 1) % eval_interval == 0 and step != 0) or (
                step + 1
            ) == num_training_steps:
                eval_metrics = evaluate(
                    model=model,
                    dataloader=validloader,
                    metrics=[
                        auroc_image_metric,
                        auroc_pixel_metric,
                        aupro_pixel_metric,
                    ],
                    device=device,
                )
                denoising_subnet.train()
                segmentation_subnet.train()

                eval_log = dict([(f"eval_{k}", v) for k, v in eval_metrics.items()])

                # wandb
                if use_wandb:
                    accelerator.log(eval_log, step=step)

                # checkpoint
                if best_score < np.mean(list(eval_metrics.values())):
                    # save best score
                    state = {"best_step": step}
                    state.update(eval_log)
                    json.dump(
                        state,
                        open(os.path.join(savedir, "best_score.json"), "w"),
                        indent="\t",
                    )

                    # save best model
                    torch.save(
                        model.state_dict(), os.path.join(savedir, f"best_model.pt")
                    )

                    _logger.info(
                        "Best Score {0:.3%} to {1:.3%}".format(
                            best_score, np.mean(list(eval_metrics.values()))
                        )
                    )

                    best_score = np.mean(list(eval_metrics.values()))
            # scheduler
            if scheduler:
                scheduler.step()

            end = time.time()

            step += 1

            if step == num_training_steps:
                train_mode = False
                break

    # print best score and step
    _logger.info(
        "Best Metric: {0:.3%} (step {1:})".format(best_score, state["best_step"])
    )

    # save latest model
    torch.save(model.state_dict(), os.path.join(savedir, f"latest_model.pt"))

    # save latest score
    state = {"latest_step": step}
    state.update(eval_log)
    json.dump(state, open(os.path.join(savedir, "latest_score.json"), "w"), indent="\t")


def evaluate(model, dataloader, metrics: list, device: str = "cpu"):
    denoising_subnet, segmentation_subnet = model
    # metrics
    auroc_image_metric, auroc_pixel_metric, aupro_pixel_metric = metrics

    # reset
    auroc_image_metric.reset()
    auroc_pixel_metric.reset()
    aupro_pixel_metric.reset()

    denoising_subnet.eval()
    segmentation_subnet.eval()
    with torch.no_grad():
        for idx, (inputs, masks, targets) in enumerate(dataloader):
            inputs, masks, targets = (
                inputs.to(device),
                masks.to(device),
                targets.to(device),
            )

            time_stamp = torch.randint(
                0,
                denoising_subnet.n_steps,
                (inputs.size(0),),
                device=device,
                dtype=torch.long,
            )
            noised_result = denoising_subnet.q_sample(inputs, time_stamp)
            denoised_input = denoising_subnet.p_sample(noised_result, time_stamp)

            # predict mask
            joined_in = torch.cat((inputs, denoised_input), dim=1)
            out_mask = segmentation_subnet(joined_in)
            outputs = F.softmax(out_mask, dim=1)
            anomaly_score = torch.topk(
                torch.flatten(outputs[:, 1, :], start_dim=1), 50
            )[0].mean(dim=1)

            # update metrics
            auroc_image_metric.update(preds=anomaly_score.cpu(), target=targets.cpu())
            auroc_pixel_metric.update(preds=outputs[:, 1, :].cpu(), target=masks.cpu())
            aupro_pixel_metric.update(preds=outputs[:, 1, :].cpu(), target=masks.cpu())

    # metrics
    metrics = {
        "AUROC-image": auroc_image_metric.compute().item(),
        "AUROC-pixel": auroc_pixel_metric.compute().item(),
        "AUPRO-pixel": aupro_pixel_metric.compute().item(),
    }

    _logger.info(
        "TEST: AUROC-image: %.3f%% | AUROC-pixel: %.3f%% | AUPRO-pixel: %.3f%%"
        % (metrics["AUROC-image"], metrics["AUROC-pixel"], metrics["AUPRO-pixel"])
    )

    return metrics
