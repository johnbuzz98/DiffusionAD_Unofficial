import random
from typing import List

from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler

from .dataset import MVTecAD, VisA


def instantiate_dataset(
    dataset_class,
    datadir: str,
    target: str,
    train: bool,
    img_size: int,
    self_aug: str = "self-augmentation",
    normalize: bool = False,
):
    return dataset_class(
        datadir=datadir,
        target=target,
        train=train,
        img_size=img_size,
        self_aug=self_aug,
        normalize=normalize,
    )


class StratifiedBatchSampler(Sampler):
    def __init__(self, anomaly_indices, non_anomaly_indices, batch_size):
        self.anomaly_indices = anomaly_indices
        self.non_anomaly_indices = non_anomaly_indices
        self.batch_size = batch_size

    def __iter__(self):
        anomaly_batches = len(self.anomaly_indices) // (self.batch_size // 2)
        for _ in range(anomaly_batches):
            anomaly_sample = random.sample(self.anomaly_indices, self.batch_size // 2)
            non_anomaly_sample = random.sample(
                self.non_anomaly_indices, self.batch_size // 2
            )
            batch = list(anomaly_sample + non_anomaly_sample)
            random.shuffle(batch)
            for idx in batch:
                yield idx

    def __len__(self):
        return len(self.anomaly_indices) // (self.batch_size // 2)


def create_dataset(name: str, *args, **kwargs):
    dataset_classes = {"MVTecAD": MVTecAD, "VisA": VisA}
    dataset_class = dataset_classes.get(name)
    if not dataset_class:
        raise ValueError(f"Invalid dataset name: {name}")

    return instantiate_dataset(dataset_class, *args, **kwargs)


def create_dataloader(
    dataset, is_training: bool, batch_size: int = 6, num_workers: int = 1
):
    if is_training:
        anomaly_indices = dataset.anomaly_indices
        non_anomaly_indices = dataset.non_anomaly_indices
        sampler = StratifiedBatchSampler(
            anomaly_indices, non_anomaly_indices, batch_size
        )
        return DataLoader(
            dataset,
            sampler=sampler,
            batch_size=batch_size,
            num_workers=num_workers,
        )
    else:
        return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
