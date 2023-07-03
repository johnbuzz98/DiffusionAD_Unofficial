# some code adapted from https://github.com/TooTouch/MemSeg/blob/main/data/dataset.py
import os
from glob import glob
from typing import List, Tuple

import cv2
import imgaug.augmenters as iaa
import numpy as np
import pandas as pd
import torch
from einops import rearrange
from torch.utils.data import Dataset
from torchvision import transforms

from data.perlin import rand_perlin_2d_np


class MVTecAD(Dataset):
    """
    MVTecAD dataset
    """

    def __init__(
        self,
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
        mean: Tuple[float] = (0.485, 0.456, 0.406),
        std: Tuple[float] = (0.229, 0.224, 0.225),
        self_aug: str = "self-augmentation",  # or "dtd-augmentation"
    ):
        self.datadir = os.path.join(datadir, target)
        self.train = train

        data_path = "train" if train else "test"
        full_path = os.path.join(self.datadir, data_path)

        self.category = os.listdir(full_path)
        self.file_list = glob(os.path.join(full_path, "*/*"))

        self.img_size = (img_size, img_size)

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

        if train:
            self.transparency_range = transparency_range
            self.perlin_noise_threshold = perlin_noise_threshold
            self.grid_size = grid_size
            self.textual_or_structural = textual_or_structural
            self.self_aug = self_aug
            self.perlin_scale_range = [
                2**i for i in range(min_perlin_scale, perlin_scale)
            ]
            if self_aug == "dtd-augmentation":
                self.texture_source_dir = texture_source_dir
                self.textual_datalist = glob(
                    os.path.join(self.texture_source_dir, "*/*.jpg")
                )

    def __getitem__(self, idx):
        file_path = self.file_list[idx]

        # image
        img = cv2.imread(file_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, dsize=self.img_size)
        # target
        target = 0 if "good" in self.file_list[idx] else 1

        if self.train:
            img, mask = self.anomaly_synthesis(img)
        else:
            try:
                mask = cv2.imread(
                    file_path.replace("test", "ground_truth").replace(
                        ".png", "_mask.png"
                    ),
                    cv2.IMREAD_GRAYSCALE,
                )
                mask = cv2.resize(mask, dsize=self.img_size).astype(bool).astype(int)
            except:
                mask = np.zeros(self.img_size).astype(bool).astype(int)

        img = self.transform(img)
        mask = torch.Tensor(mask).to(torch.int64)

        return img, mask, target

    def generate_target_foreground_mask(self, img: np.ndarray) -> np.ndarray:
        # convert RGB into GRAY scale
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # generate binary mask of gray scale image
        _, target_background_mask = cv2.threshold(  # srcm thresh, maxval, type, dst
            img_gray, 100, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
        )
        target_background_mask = target_background_mask.astype(np.int_)

        # invert mask for foreground mask
        target_foreground_mask = -(target_background_mask - 1)
        target_foreground_mask = target_foreground_mask + 254

        return target_foreground_mask

    def generate_perlin_noise_mask(self) -> np.ndarray:
        # define perlin noise scale
        perlin_scalex = perlin_scaley = np.random.choice(self.perlin_scale_range)

        # generate perlin noise
        perlin_noise = rand_perlin_2d_np(
            (self.img_size[0], self.img_size[1]), (perlin_scalex, perlin_scaley)
        )

        # apply affine transform
        rot = iaa.Affine(rotate=(-90, 90))
        perlin_noise = rot(image=perlin_noise)

        # make a mask by applying threshold
        mask_noise = np.where(
            perlin_noise > self.perlin_noise_threshold,
            np.ones_like(perlin_noise),
            np.zeros_like(perlin_noise),
        )

        return mask_noise

    def crop_and_resize_random_quarter(self, image: np.ndarray) -> np.ndarray:
        # Get the dimensions of the input image
        height, width = image.shape[:2]

        # Calculate the quarter size
        quarter_height = height // 2
        quarter_width = width // 2

        # Generate random starting coordinates for cropping
        start_y = np.random.randint(0, height - quarter_height)
        start_x = np.random.randint(0, width - quarter_width)

        # Crop and resize the random quarter of the image
        cropped_quarter = image[
            start_y : start_y + quarter_height, start_x : start_x + quarter_width
        ]
        resized_quarter = cv2.resize(cropped_quarter, (width, height))

        return resized_quarter

    def randAugmenter(self) -> iaa.Sequential:
        augmenters = [
            iaa.GammaContrast((0.5, 2.0), per_channel=True),
            iaa.pillike.EnhanceSharpness(),
            iaa.Solarize(0.5, threshold=(32, 128)),
            iaa.Posterize(),
            iaa.Invert(),
            iaa.pillike.Autocontrast(),
            iaa.pillike.Equalize(),
        ]
        aug_ind = np.random.choice(np.arange(len(augmenters)), 2, replace=False)
        aug = iaa.Sequential(
            [
                augmenters[aug_ind[0]],
                augmenters[aug_ind[1]],
            ]
        )
        return aug

    def random_arrange(self, img: np.ndarray) -> np.ndarray:
        assert (
            self.img_size[0] % self.grid_size == 0
        ), "structure should be devided by grid size accurately"
        grid_w = self.img_size[1] // self.grid_size
        grid_h = self.img_size[0] // self.grid_size

        source_img = rearrange(
            tensor=img,
            pattern="(h gh) (w gw) c -> (h w) gw gh c",
            gw=grid_w,
            gh=grid_h,
        )
        disordered_idx = np.arange(source_img.shape[0])
        np.random.shuffle(disordered_idx)

        source_img = rearrange(
            tensor=source_img[disordered_idx],
            pattern="(h w) gw gh c -> (h gh) (w gw) c",
            h=self.grid_size,
            w=self.grid_size,
        ).astype(np.uint8)

        return source_img

    def anomaly_synthesis(self, source_img):
        """
        step 1. generate mask
            - target foreground mask (For textural datasets, the foreground is replaced by a random part of the whole image)
            - perlin noise mask

        step 2. Visual Inconsistencies
            - self-augmentation
            - DTD dataset augmentation

        step 3. blending image and anomaly source
        """

        # step 1. generate mask
        ## target foreground mask (For textural datasets, the foreground is replaced by a random part of the whole image)
        if self.textual_or_structural == "textual":
            source_img = self.crop_and_resize_random_quarter(source_img)
        target_foreground_mask = self.generate_target_foreground_mask(source_img)
        ## perlin noise mask
        perlin_noise_mask = self.generate_perlin_noise_mask()
        ## mask
        mask = perlin_noise_mask * target_foreground_mask
        mask_inversed = 1 - mask

        # step 2. Visual Inconsistencies

        ## self-augmentation or DTD dataset augmentation
        if self.self_aug == "dtd-augmentation":
            textual_datapath = np.random.choice(self.textual_datalist, 1)[0]
            augmented_image = cv2.imread(textual_datapath)
            augmented_image = cv2.cvtColor(augmented_image, cv2.COLOR_BGR2RGB)
            augmented_image = cv2.resize(augmented_image, dsize=self.img_size)
        else:
            augmented_image = source_img.copy()

        augmented_image = self.randAugmenter()(images=source_img)
        augmented_image = self.random_arrange(augmented_image)

        # step 3. blending image and anomaly source
        factor = np.random.uniform(*self.transparency_range, size=1)[0]

        reshaped_mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
        reshaped_reversed_mask = np.repeat(mask_inversed[:, :, np.newaxis], 3, axis=2)

        anomaly_synthesis_img = (
            factor * (reshaped_mask * source_img)
            + (1 - factor) * (reshaped_mask * augmented_image)
            + reshaped_reversed_mask * source_img
        )

        anomaly_synthesis_img = anomaly_synthesis_img.astype(np.uint8)

        return anomaly_synthesis_img, mask

    def __len__(self):
        return len(self.file_list)


class VisA(Dataset):

    """
    VisA dataset

    """

    def __init__(
        self,
        datadir: str,
        target: str,
        train: bool,
        img_size: int,
        texture_source_dir: str = "./dataset/dtd/images",
        grid_size: int = 8,
        perlin_scale: int = 6,
        min_perlin_scale: int = 0,
        perlin_noise_threshold: float = 0.5,
        textual_or_structural: str = "textual",
        transparency_range: List[float] = [0.15, 1.0],
        self_aug: str = "self-augmentation",  # or "dtd-augmentation"
    ):
        self.datadir = datadir
        self.train = train

        file_df = pd.read_csv(os.path.join(datadir, "split_csv", "1cls.csv"))
        file_df = file_df[file_df["object"] == target]

        # full_path = os.path.join(self.datadir, "Data/Images", self.data_path, "*")
        # self.file_list = glob(full_path)
        self.img_size = (img_size, img_size)

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                # transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )

        if train:
            self.file_df = file_df[file_df["split"] == "train"].reset_index(drop=True)
            self.transparency_range = transparency_range
            self.perlin_noise_threshold = perlin_noise_threshold
            self.grid_size = grid_size
            self.textual_or_structural = textual_or_structural
            self.self_aug = self_aug
            self.perlin_scale_range = [
                2**i for i in range(min_perlin_scale, perlin_scale)
            ]
            if self_aug == "dtd-augmentation":
                self.texture_source_dir = texture_source_dir
                self.textual_datalist = glob(
                    os.path.join(self.texture_source_dir + "*/*.jpg")
                )
        else:
            self.file_df = file_df[file_df["split"] == "test"].reset_index(drop=True)

    def __getitem__(self, idx):
        file_path = os.path.join(self.datadir, self.file_df["image"][idx])

        # image
        img = cv2.imread(file_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, dsize=self.img_size)

        # mask
        if self.train:
            img, mask = self.anomaly_synthesis(img)
            target = 0
        else:
            if self.file_df["label"][idx] == "normal":
                target = 0
                mask = np.zeros((self.img_size)).astype(bool).astype(int)
            else:
                mask = cv2.imread(os.path.join(self.datadir, self.file_df["mask"][idx]))
                mask = cv2.resize(mask, dsize=self.img_size).astype(bool).astype(int)
                mask = mask[:, :, 0]
                target = 1
        img = self.transform(img)
        mask = torch.Tensor(mask).to(torch.int64)

        return img, mask, target

    def generate_target_foreground_mask(self, img: np.ndarray) -> np.ndarray:
        # convert RGB into GRAY scale
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # generate binary mask of gray scale image
        _, target_background_mask = cv2.threshold(
            img_gray, 100, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
        )
        target_foreground_mask = -(target_background_mask - 1)
        target_foreground_mask = target_foreground_mask + 254

        # invert mask for foreground mask
        target_foreground_mask = 1 - target_background_mask

        return target_foreground_mask

    def generate_perlin_noise_mask(self) -> np.ndarray:
        # define perlin noise scale
        perlin_scalex = perlin_scaley = np.random.choice(self.perlin_scale_range)

        # generate perlin noise
        perlin_noise = rand_perlin_2d_np(
            (self.img_size[0], self.img_size[1]), (perlin_scalex, perlin_scaley)
        )

        # apply affine transform
        rot = iaa.Affine(rotate=(-90, 90))
        perlin_noise = rot(image=perlin_noise)

        # make a mask by applying threshold
        mask_noise = np.where(
            perlin_noise > self.perlin_noise_threshold,
            np.ones_like(perlin_noise),
            np.zeros_like(perlin_noise),
        )

        return mask_noise

    def crop_and_resize_random_quarter(self, image: np.ndarray) -> np.ndarray:
        # Get the dimensions of the input image
        height, width = image.shape[:2]

        # Calculate the quarter size
        quarter_height = height // 2
        quarter_width = width // 2

        # Generate random starting coordinates for cropping
        start_y = np.random.randint(0, height - quarter_height)
        start_x = np.random.randint(0, width - quarter_width)

        # Crop and resize the random quarter of the image
        cropped_quarter = image[
            start_y : start_y + quarter_height, start_x : start_x + quarter_width
        ]
        resized_quarter = cv2.resize(cropped_quarter, (width, height))

        return resized_quarter

    def randAugmenter(self) -> iaa.Sequential:
        augmenters = [
            iaa.GammaContrast((0.5, 2.0), per_channel=True),
            iaa.pillike.EnhanceSharpness(),
            iaa.Solarize(0.5, threshold=(32, 128)),
            iaa.Posterize(),
            iaa.Invert(),
            iaa.pillike.Autocontrast(),
            iaa.pillike.Equalize(),
        ]
        aug_ind = np.random.choice(np.arange(len(augmenters)), 2, replace=False)
        aug = iaa.Sequential(
            [
                augmenters[aug_ind[0]],
                augmenters[aug_ind[1]],
            ]
        )
        return aug

    def random_arrange(self, img: np.ndarray) -> np.ndarray:
        assert (
            self.img_size[0] % self.grid_size == 0
        ), "structure should be devided by grid size accurately"
        grid_w = self.img_size[1] // self.grid_size
        grid_h = self.img_size[0] // self.grid_size

        source_img = rearrange(
            tensor=img,
            pattern="(h gh) (w gw) c -> (h w) gw gh c",
            gw=grid_w,
            gh=grid_h,
        )
        disordered_idx = np.arange(source_img.shape[0])
        np.random.shuffle(disordered_idx)

        source_img = rearrange(
            tensor=source_img[disordered_idx],
            pattern="(h w) gw gh c -> (h gh) (w gw) c",
            h=self.grid_size,
            w=self.grid_size,
        ).astype(np.uint8)

        return source_img

    def anomaly_synthesis(self, source_img):
        """
        step 1. generate mask
            - target foreground mask (For textural datasets, the foreground is replaced by a random part of the whole image)
            - perlin noise mask

        step 2. Visual Inconsistencies
            - self-augmentation
            - DTD dataset augmentation

        step 3. blending image and anomaly source
        """

        # step 1. generate mask
        ## target foreground mask (For textural datasets, the foreground is replaced by a random part of the whole image)
        if self.textual_or_structural == "textual":
            source_img = self.crop_and_resize_random_quarter(source_img)
        target_foreground_mask = self.generate_target_foreground_mask(source_img)
        ## perlin noise mask
        perlin_noise_mask = self.generate_perlin_noise_mask()

        ## mask
        mask = perlin_noise_mask * target_foreground_mask
        mask_inversed = 1 - mask

        # step 2. Visual Inconsistencies

        ## self-augmentation or DTD dataset augmentation
        if self.self_aug == "dtd-augmentation":
            textual_datapath = np.random.choice(self.textual_datalist, 1)[0]
            augmented_image = cv2.imread(textual_datapath)
            augmented_image = cv2.cvtColor(augmented_image, cv2.COLOR_BGR2RGB)
            augmented_image = cv2.resize(augmented_image, dsize=self.img_size)
        else:
            augmented_image = source_img.copy()

        augmented_image = self.randAugmenter()(images=source_img)
        augmented_image = self.random_arrange(augmented_image)

        # step 3. blending image and anomaly source
        factor = np.random.uniform(*self.transparency_range, size=1)[0]
        reshaped_mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
        reshaped_reversed_mask = np.repeat(mask_inversed[:, :, np.newaxis], 3, axis=2)

        anomaly_synthesis_img = (
            factor * (reshaped_mask * source_img)
            + (1 - factor) * (reshaped_mask * augmented_image)
            + reshaped_reversed_mask * source_img
        )

        anomaly_synthesis_img = anomaly_synthesis_img.astype(np.uint8)

        return anomaly_synthesis_img, mask

    def __len__(self):
        return len(self.file_df)


class dagm(Dataset):

    """
    VisA dataset

    """

    def __init__(
        self,
        datadir: str,
        target: str,
        train: bool,
        img_size: int,
        texture_source_dir: str = "./dataset/dtd/images",
        grid_size: int = 8,
        perlin_scale: int = 6,
        min_perlin_scale: int = 0,
        perlin_noise_threshold: float = 0.5,
        textual_or_structural: str = "textual",
        transparency_range: List[float] = [0.15, 1.0],
        self_aug: str = "self-augmentation",
    ):
        self.datadir = os.path.join(datadir, target)
        self.train = train
        self.data_path = "Normal" if train else "Anomaly"
        full_path = os.path.join(self.datadir, "Data/Images", self.data_path, "*")
        self.file_list = glob(full_path)
        self.img_size = (img_size, img_size)

        if train:
            self.transparency_range = transparency_range
            self.perlin_noise_threshold = perlin_noise_threshold
            self.grid_size = grid_size
            self.rot = iaa.Sequential([iaa.Affine(rotate=(-90, 90))])
            self.textual_or_structural = textual_or_structural
            self.self_aug = self_aug
            self.texture_source_dir = texture_source_dir
            self.perlin_scale_range = [
                2**i for i in range(min_perlin_scale, perlin_scale)
            ]

    def __getitem__(self, idx):
        file_path = self.file_list[idx]

        # image
        img = cv2.imread(file_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, dsize=self.img_size)

        # mask
        if self.train:
            img, mask = self.anomaly_synthesis(img)
        else:
            mask = cv2.imread(
                glob(os.path.join(self.datadir, "Data/Masks/Anomaly", "*"))[idx],
                cv2.IMREAD_GRAYSCALE,
            )
            mask = cv2.resize(mask, dsize=self.img_size).astype(bool).astype(int)

        # img = self.transform(img)
        mask = torch.Tensor(mask).to(torch.int64)

        return img, mask

    def generate_target_foreground_mask(self, img: np.ndarray) -> np.ndarray:
        # convert RGB into GRAY scale
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # generate binary mask of gray scale image
        _, target_background_mask = cv2.threshold(
            img_gray, 100, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
        )
        target_background_mask = target_background_mask.astype(np.int_)

        # invert mask for foreground mask
        target_foreground_mask = 1 - target_background_mask

        return target_foreground_mask

    def generate_perlin_noise_mask(self) -> np.ndarray:
        # define perlin noise scale
        perlin_scalex = perlin_scaley = np.random.choice(self.perlin_scale_range)

        # generate perlin noise
        perlin_noise = rand_perlin_2d_np(
            (self.img_size[0], self.img_size[1]), (perlin_scalex, perlin_scaley)
        )

        # apply affine transform
        rot = iaa.Affine(rotate=(-90, 90))
        perlin_noise = rot(image=perlin_noise)

        # make a mask by applying threshold
        mask_noise = np.where(
            perlin_noise > self.perlin_noise_threshold,
            np.ones_like(perlin_noise),
            np.zeros_like(perlin_noise),
        )

        return mask_noise

    def crop_and_resize_random_quarter(self, image: np.ndarray) -> np.ndarray:
        # Get the dimensions of the input image
        height, width = image.shape[:2]

        # Calculate the quarter size
        quarter_height = height // 2
        quarter_width = width // 2

        # Generate random starting coordinates for cropping
        start_y = np.random.randint(0, height - quarter_height)
        start_x = np.random.randint(0, width - quarter_width)

        # Crop and resize the random quarter of the image
        cropped_quarter = image[
            start_y : start_y + quarter_height, start_x : start_x + quarter_width
        ]
        resized_quarter = cv2.resize(cropped_quarter, (width, height))

        return resized_quarter

    def randAugmenter(self) -> iaa.Sequential:
        augmenters = [
            iaa.GammaContrast((0.5, 2.0), per_channel=True),
            iaa.pillike.EnhanceSharpness(),
            iaa.Solarize(0.5, threshold=(32, 128)),
            iaa.Posterize(),
            iaa.Invert(),
            iaa.pillike.Autocontrast(),
            iaa.pillike.Equalize(),
        ]
        aug_ind = np.random.choice(np.arange(len(augmenters)), 2, replace=False)
        aug = iaa.Sequential(
            [
                augmenters[aug_ind[0]],
                augmenters[aug_ind[1]],
            ]
        )
        return aug

    def random_arrange(self, img: np.ndarray) -> np.ndarray:
        assert (
            self.img_size[0] % self.grid_size == 0
        ), "structure should be devided by grid size accurately"
        grid_w = self.img_size[1] // self.grid_size
        grid_h = self.img_size[0] // self.grid_size

        source_img = rearrange(
            tensor=img,
            pattern="(h gh) (w gw) c -> (h w) gw gh c",
            gw=grid_w,
            gh=grid_h,
        )
        disordered_idx = np.arange(source_img.shape[0])
        np.random.shuffle(disordered_idx)

        source_img = rearrange(
            tensor=source_img[disordered_idx],
            pattern="(h w) gw gh c -> (h gh) (w gw) c",
            h=self.grid_size,
            w=self.grid_size,
        ).astype(np.uint8)

        return source_img

    def anomaly_synthesis(self, source_img):
        """
        step 1. generate mask
            - target foreground mask (For textural datasets, the foreground is replaced by a random part of the whole image)
            - perlin noise mask

        step 2. Visual Inconsistencies
            - self-augmentation
            - DTD dataset augmentation

        step 3. blending image and anomaly source
        """

        # step 1. generate mask
        ## target foreground mask (For textural datasets, the foreground is replaced by a random part of the whole image)
        if self.textual_or_structural == "textual":
            source_img = self.crop_and_resize_random_quarter(source_img)
        target_foreground_mask = self.generate_target_foreground_mask(source_img)
        ## perlin noise mask
        perlin_noise_mask = self.generate_perlin_noise_mask()

        ## mask
        mask = perlin_noise_mask * target_foreground_mask
        mask_inversed = 1 - mask

        # step 2. Visual Inconsistencies

        ## self-augmentation or DTD dataset augmentation
        if self.self_aug == "dtd-augmentation":
            textual_datapath = np.random.choice(self.textual_datalist, 1)[0]
            augmented_image = cv2.imread(textual_datapath)
            augmented_image = cv2.cvtColor(augmented_image, cv2.COLOR_BGR2RGB)
            augmented_image = cv2.resize(augmented_image, dsize=self.img_size)
        else:
            augmented_image = source_img.copy()

        augmented_image = self.randAugmenter()(images=source_img)
        augmented_image = self.random_arrange(augmented_image)

        # step 3. blending image and anomaly source
        factor = np.random.uniform(*self.transparency_range, size=1)[0]
        reshaped_mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
        reshaped_reversed_mask = np.repeat(mask_inversed[:, :, np.newaxis], 3, axis=2)

        anomaly_synthesis_img = (
            factor * (reshaped_mask * source_img)
            + (1 - factor) * (reshaped_mask * augmented_image)
            + reshaped_reversed_mask * source_img
        )

        anomaly_synthesis_img = anomaly_synthesis_img.astype(np.uint8)

        return anomaly_synthesis_img, mask

    def __len__(self):
        return len(self.file_list)
