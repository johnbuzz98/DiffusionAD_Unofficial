from typing import List

from torch.utils.data import DataLoader

from .dataset import MVTecAD, VisA


def instantiate_dataset(
    dataset_class,
    datadir: str,
    target: str,
    train: bool,
    img_size: int,
    texture_source_dir: str = "./dataset/dtd/images",
    grid_size: int = 8,
    perlin_scale: int = 6,
    min_perlin_scale: int = 0,
    perlin_noise_threshold: float = 0.5,
    textual_or_structural: str = "structural",
    transparency_range: List[float] = [0.15, 1.0],
    self_aug: str = "self-augmentation",
):
    return dataset_class(
        datadir=datadir,
        target=target,
        train=train,
        img_size=img_size,
        texture_source_dir=texture_source_dir,
        grid_size=grid_size,
        perlin_scale=perlin_scale,
        min_perlin_scale=min_perlin_scale,
        perlin_noise_threshold=perlin_noise_threshold,
        textual_or_structural=textual_or_structural,
        transparency_range=transparency_range,
        self_aug=self_aug,
    )


def create_dataset(datasetname: str, *args, **kwargs):
    dataset_classes = {"MVTecAD": MVTecAD, "VisA": VisA}
    dataset_class = dataset_classes.get(datasetname)
    if not dataset_class:
        raise ValueError(f"Invalid dataset name: {datasetname}")

    return instantiate_dataset(dataset_class, *args, **kwargs)


def create_dataloader(
    dataset, is_training: bool, batch_size: int = 16, num_workers: int = 1
):
    return DataLoader(
        dataset, shuffle=is_training, batch_size=batch_size, num_workers=num_workers
    )
