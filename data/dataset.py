# some code adapted from https://github.com/TooTouch/MemSeg/blob/main/data/dataset.py
import json
import os
from glob import glob
from pathlib import Path
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
    Dataset class for the MVTec Anomaly Detection (MVTecAD) dataset.

    This class is responsible for loading and transforming the MVTecAD dataset.
    It supports various configurations including different target categories,
    image sizes, augmentation strategies, and more.

    Attributes:
        datadir (Path): Path to the dataset directory.
        target (str): Target category for anomaly detection.
        train (bool): If set to True, the dataset loads training data. If False, it loads testing data.
        img_size (int): Size of the image after resizing.
        texture_source_dir (Path): Directory path containing texture images for augmentation.
        grid_size (int): Grid size used in image processing.
        perlin_scale (int): Scale factor for generating Perlin noise.
        min_perlin_scale (int): Minimum scale factor for generating Perlin noise.
        perlin_noise_threshold (float): Threshold for creating a Perlin noise mask.
        transparency_range (List[float]): Range of transparency values for blending images.
        self_aug (str): Type of augmentation strategy to employ.
        normalize (bool): Flag indicating whether the images should be normalized.
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
        transparency_range: List[float] = [0.15, 1.0],
        self_aug: str = "self-augmentation",
        normalize: bool = True,
    ):
        """
        Initialize the MVTecAD dataset with the provided configurations.

        Args:
            ... (as described in the class Attributes section)
        """

        self.datadir = Path(datadir) / target
        self.train = train
        data_path = "train" if train else "test"
        self.file_list = list((self.datadir / data_path).rglob("*/*"))

        self.img_size = (img_size, img_size)
        with open("./data/mvtec_ad.json", "r") as f:
            self.json_data = json.load(f)[target]

        if normalize:
            self.transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=self.json_data["mean"], std=self.json_data["std"]
                    ),
                ]
            )
        else:
            self.transform = transforms.ToTensor()

        if train:
            self.anomaly_indices = list(range(len(self.file_list) // 2))
            self.non_anomaly_indices = list(
                range(len(self.file_list) // 2, len(self.file_list))
            )
            self.transparency_range = transparency_range
            self.perlin_noise_threshold = perlin_noise_threshold
            self.grid_size = grid_size
            self.self_aug = self_aug
            self.perlin_scale_range = [
                2**i for i in range(min_perlin_scale, perlin_scale)
            ]

            if self_aug == "dtd-augmentation":
                self.texture_source_dir = Path(texture_source_dir)
                self.textual_datalist = list(self.texture_source_dir.rglob("*/*.jpg"))

    def __getitem__(self, idx):
        """
        Get a sample from the dataset.

        Args:
            idx (int): Index of the sample.

        Returns:
            tuple: A tuple containing:
                - img (torch.Tensor): The processed image.
                - mask (torch.Tensor): The anomaly mask.
                - target (int): The target label (0 for "good", 1 otherwise).
        """
        file_path = self.file_list[idx]

        # Load and preprocess image
        img = cv2.imread(str(file_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, dsize=self.img_size)

        # Determine target
        target = 0 if "good" in str(file_path) else 1

        if self.train:
            if idx in self.anomaly_indices:
                img, mask = self.anomaly_synthesis(img)
                target = 1
            else:
                mask = np.zeros(self.img_size).astype(bool).astype(int)
        else:
            mask_path = (
                str(file_path)
                .replace("test", "ground_truth")
                .replace(".png", "_mask.png")
            )
            try:
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                mask = cv2.resize(mask, dsize=self.img_size).astype(bool).astype(int)
            except:
                mask = np.zeros(self.img_size).astype(bool).astype(int)

        img, mask = img / 255.0, mask / 255.0
        img = self.transform(img).to(torch.float32)
        mask = torch.Tensor(mask).to(torch.int64)

        return img, mask, target

    def generate_target_foreground_mask(self, img: np.ndarray) -> np.ndarray:
        """
        Generate a target foreground mask for a given image.

        The function first converts the image to grayscale and then applies a
        binary thresholding. The result is further adjusted based on the
        "bg_reverse" property from the json_data attribute.

        Args:
            img (np.ndarray): Input image array with shape (height, width, channels).

        Returns:
            np.ndarray: The target foreground mask.
        """

        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        _, target_background_mask = cv2.threshold(
            img_gray,
            self.json_data["bg_threshold"],
            255,
            cv2.THRESH_BINARY | cv2.THRESH_OTSU,
        )

        target_foreground_mask = np.where(target_background_mask == 0, 0, 255)

        if not self.json_data["bg_reverse"]:
            target_foreground_mask = 255 - target_foreground_mask

        return target_foreground_mask

    def generate_partial_foremask(
        self, image: np.ndarray, split_size: float = 3
    ) -> np.ndarray:
        """
        Generate a partial foreground mask for the given image based on the specified split size.

        Args:
        - image (np.ndarray): Input image array with shape (height, width, channels).
        - split_size (int, optional): Specifies how the image should be split.
                                    Determines the size of the foreground region.
                                    Should be one of {2, 3, 4}. Defaults to 4.

        Returns:
        - np.ndarray: A mask with a portion set to 255 (foreground) and the rest set to 0.

        Raises:
        - ValueError: If an unsupported split_size is provided.
        """

        height, width, _ = image.shape
        mask = np.zeros((height, width), dtype=np.uint8)

        roi_size = (height // split_size, width // split_size)

        # Randomly select the starting point for the ROI
        start_y = np.random.randint(0, height - roi_size[0] + 1)
        start_x = np.random.randint(0, width - roi_size[1] + 1)

        # Set the region of interest to 255
        mask[start_y : start_y + roi_size[0], start_x : start_x + roi_size[1]] = 255

        return mask

    def generate_perlin_noise_mask(self) -> np.ndarray:
        """
        Generate a mask based on Perlin noise.

        The function defines a Perlin noise scale, generates Perlin noise
        based on the defined scale, applies an affine rotation, and then
        creates a mask by thresholding the noise based on a predefined
        noise threshold.

        Returns:
            np.ndarray: The mask generated from Perlin noise.
        """

        # Define Perlin noise scale
        perlin_scalex = perlin_scaley = np.random.choice(self.perlin_scale_range)

        # Generate Perlin noise
        perlin_noise = rand_perlin_2d_np(
            (self.img_size[0], self.img_size[1]), (perlin_scalex, perlin_scaley)
        )

        # Apply affine transform
        rot = iaa.Affine(rotate=(-90, 90))
        perlin_noise = rot(image=perlin_noise)

        # Create a mask by applying threshold
        mask_noise = np.where(
            perlin_noise > self.perlin_noise_threshold,
            np.ones_like(perlin_noise),
            np.zeros_like(perlin_noise),
        )

        return mask_noise

    def randAugmenter(self) -> iaa.Sequential:
        """
        Randomly select two augmentations and return them as a sequential operation.

        The function first defines a list of possible augmentations. It then
        randomly selects two unique augmentations from the list and
        returns them as a sequential operation.

        Returns:
            iaa.Sequential: A sequence of two randomly chosen augmentations.
        """

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
        aug = iaa.Sequential([augmenters[aug_ind[0]], augmenters[aug_ind[1]]])

        return aug

    def random_arrange(self, img: np.ndarray) -> np.ndarray:
        """
        Randomly rearrange the image based on grid size.

        The function splits the image into small grids and then shuffles the
        order of these grids to produce a randomly rearranged image.

        Args:
            img (np.ndarray): Input image array to be rearranged.

        Returns:
            np.ndarray: The randomly rearranged image.

        Raises:
            AssertionError: If the image size is not divisible by the grid size.
        """

        assert (
            self.img_size[0] % self.grid_size == 0
        ), "Image should be divisible by grid size accurately."

        grid_w = self.img_size[1] // self.grid_size
        grid_h = self.img_size[0] // self.grid_size

        source_img = rearrange(
            tensor=img, pattern="(h gh) (w gw) c -> (h w) gw gh c", gw=grid_w, gh=grid_h
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

    def anomaly_synthesis(
        self, source_img: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Synthesize an anomaly on the source image.

        This function creates an anomaly on the source image by generating a mask,
        applying visual inconsistencies, and blending the original image with
        the augmented image based on the mask.

        Args:
            source_img (np.ndarray): The source image on which the anomaly will be synthesized.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing the synthesized anomalous image and the applied mask.
        """

        # Generate mask
        if not self.json_data["use_mask"]:
            target_foreground_mask = self.generate_partial_foremask(source_img)
        else:
            target_foreground_mask = self.generate_target_foreground_mask(source_img)

        perlin_noise_mask = self.generate_perlin_noise_mask()
        mask = perlin_noise_mask * target_foreground_mask
        mask_inversed = 1 - mask

        # Apply Visual Inconsistencies
        if self.self_aug == "dtd-augmentation":
            textual_datapath = np.random.choice(self.textual_datalist, 1)[0]
            augmented_image = cv2.imread(str(textual_datapath))
            augmented_image = cv2.cvtColor(augmented_image, cv2.COLOR_BGR2RGB)
            augmented_image = cv2.resize(augmented_image, dsize=self.img_size)
        else:
            augmented_image = source_img.copy()
        # 256 256 3 -> 3 256 256
        augmented_image = self.randAugmenter()(
            images=rearrange(source_img, "h w c -> c h w")
        )
        # 3 256 256 -> 256 256 3
        augmented_image = rearrange(augmented_image, "c h w -> h w c")
        augmented_image = self.random_arrange(augmented_image)

        # Blend image and anomaly source
        factor = np.random.uniform(*self.transparency_range, size=1)[0]
        reshaped_mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
        reshaped_reversed_mask = np.repeat(mask_inversed[:, :, np.newaxis], 3, axis=2)

        anomaly_synthesis_img = (
            factor * reshaped_mask * source_img
            + (1 - factor) * reshaped_mask * augmented_image
            + reshaped_reversed_mask * source_img
        ).astype(np.uint8)

        return anomaly_synthesis_img, mask

    def __len__(self):
        return len(self.file_list)


class VisA(Dataset):
    """
    Dataset class for the VisA dataset.

    This class is responsible for loading and transforming the VisA dataset.
    It can perform various augmentations, handle different classes, and
    provides support for training and testing splits.

    Attributes:
        datadir (str): Path to the dataset directory.
        target (str): Target object for the dataset.
        train (bool): Flag to indicate if the dataset is for training or testing.
        img_size (int): Size of the image (height and width are assumed to be the same).
        texture_source_dir (str): Directory containing texture images for augmentation.
        grid_size (int): Grid size for random arrangement of image.
        perlin_scale (int): Scale factor for Perlin noise.
        min_perlin_scale (int): Minimum scale factor for Perlin noise.
        perlin_noise_threshold (float): Threshold value for Perlin noise mask generation.
        transparency_range (List[float]): Range for transparency during anomaly synthesis.
        self_aug (str): Type of self-augmentation to apply.
        cls (str): Classification type - can be "1cls", "2cls_fewshot", or "2cls_highshot".
        normalize (bool): Flag to indicate if normalization should be applied.
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
        transparency_range: List[float] = [0.15, 1.0],
        self_aug: str = "self-augmentation",
        cls: str = "1cls",
        normalize: bool = True,
    ):
        """
        Initialize the VisA dataset with given parameters.

        Args:
            ... (as mentioned in the class Attributes section)
        """

        self.datadir = datadir
        self.train = train

        # Load dataset split details
        file_df = pd.read_csv(os.path.join(datadir, "split_csv", f"{cls}.csv"))
        self.file_df = file_df[file_df["object"] == target]

        self.img_size = (img_size, img_size)

        # Load normalization values from JSON
        with open(f"./data/VisA_{cls}.json", "r") as f:
            self.json_data = json.load(f)[target]

        if normalize:
            self.transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=self.json_data["mean"], std=self.json_data["std"]
                    ),
                ]
            )
        else:
            self.transform = transforms.ToTensor()
        # Dataset split specific assignments
        if train:
            self.file_df = self.file_df[self.file_df["split"] == "train"].reset_index(
                drop=True
            )
            self.anomaly_indices = list(range(len(self.file_list) // 2))
            self.non_anomaly_indices = list(
                range(len(self.file_list) // 2, len(self.file_list))
            )
            self.transparency_range = transparency_range
            self.perlin_noise_threshold = perlin_noise_threshold
            self.grid_size = grid_size
            self.self_aug = self_aug
            self.perlin_scale_range = [
                2**i for i in range(min_perlin_scale, perlin_scale)
            ]

            if self_aug == "dtd-augmentation":
                self.texture_source_dir = texture_source_dir
                self.textual_datalist = glob(
                    os.path.join(self.texture_source_dir, "*/*.jpg")
                )
        else:
            self.file_df = self.file_df[self.file_df["split"] == "test"].reset_index(
                drop=True
            )

    def __getitem__(self, idx):
        file_path = os.path.join(self.datadir, self.file_df["image"][idx])

        # image
        img = cv2.imread(file_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, dsize=self.img_size)

        # mask
        if self.train and idx in self.anomaly_indices:
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

        img, mask = img / 255.0, mask / 255.0
        img = self.transform(img).to(torch.float32)
        mask = torch.Tensor(mask).to(torch.int64)

        return img, mask, target

    def generate_target_foreground_mask(self, img: np.ndarray) -> np.ndarray:
        """
        Generate a target foreground mask for a given image.

        The function first converts the image to grayscale and then applies a
        binary thresholding. The result is further adjusted based on the
        "bg_reverse" property from the json_data attribute.

        Args:
            img (np.ndarray): Input image array with shape (height, width, channels).

        Returns:
            np.ndarray: The target foreground mask.
        """

        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        _, target_background_mask = cv2.threshold(
            img_gray,
            self.json_data["bg_threshold"],
            255,
            cv2.THRESH_BINARY | cv2.THRESH_OTSU,
        )

        target_foreground_mask = np.where(target_background_mask == 0, 0, 255)

        if not self.json_data["bg_reverse"]:
            target_foreground_mask = 255 - target_foreground_mask

        return target_foreground_mask

    def generate_partial_foremask(
        self, image: np.ndarray, split_size: float = 3
    ) -> np.ndarray:
        """
        Generate a partial foreground mask for the given image based on the specified split size.

        Args:
        - image (np.ndarray): Input image array with shape (height, width, channels).
        - split_size (int, optional): Specifies how the image should be split.
                                    Determines the size of the foreground region.
                                    Should be one of {2, 3, 4}. Defaults to 4.

        Returns:
        - np.ndarray: A mask with a portion set to 255 (foreground) and the rest set to 0.

        Raises:
        - ValueError: If an unsupported split_size is provided.
        """

        height, width, _ = image.shape
        mask = np.zeros((height, width), dtype=np.uint8)

        roi_size = (height // split_size, width // split_size)

        # Randomly select the starting point for the ROI
        start_y = np.random.randint(0, height - roi_size[0] + 1)
        start_x = np.random.randint(0, width - roi_size[1] + 1)

        # Set the region of interest to 255
        mask[start_y : start_y + roi_size[0], start_x : start_x + roi_size[1]] = 255

        return mask

    def generate_perlin_noise_mask(self) -> np.ndarray:
        """
        Generate a mask based on Perlin noise.

        The function defines a Perlin noise scale, generates Perlin noise
        based on the defined scale, applies an affine rotation, and then
        creates a mask by thresholding the noise based on a predefined
        noise threshold.

        Returns:
            np.ndarray: The mask generated from Perlin noise.
        """

        # Define Perlin noise scale
        perlin_scalex = perlin_scaley = np.random.choice(self.perlin_scale_range)

        # Generate Perlin noise
        perlin_noise = rand_perlin_2d_np(
            (self.img_size[0], self.img_size[1]), (perlin_scalex, perlin_scaley)
        )

        # Apply affine transform
        rot = iaa.Affine(rotate=(-90, 90))
        perlin_noise = rot(image=perlin_noise)

        # Create a mask by applying threshold
        mask_noise = np.where(
            perlin_noise > self.perlin_noise_threshold,
            np.ones_like(perlin_noise),
            np.zeros_like(perlin_noise),
        )

        return mask_noise

    def randAugmenter(self) -> iaa.Sequential:
        """
        Randomly select two augmentations and return them as a sequential operation.

        The function first defines a list of possible augmentations. It then
        randomly selects two unique augmentations from the list and
        returns them as a sequential operation.

        Returns:
            iaa.Sequential: A sequence of two randomly chosen augmentations.
        """

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
        aug = iaa.Sequential([augmenters[aug_ind[0]], augmenters[aug_ind[1]]])

        return aug

    def random_arrange(self, img: np.ndarray) -> np.ndarray:
        """
        Randomly rearrange the image based on grid size.

        The function splits the image into small grids and then shuffles the
        order of these grids to produce a randomly rearranged image.

        Args:
            img (np.ndarray): Input image array to be rearranged.

        Returns:
            np.ndarray: The randomly rearranged image.

        Raises:
            AssertionError: If the image size is not divisible by the grid size.
        """

        assert (
            self.img_size[0] % self.grid_size == 0
        ), "Image should be divisible by grid size accurately."

        grid_w = self.img_size[1] // self.grid_size
        grid_h = self.img_size[0] // self.grid_size

        source_img = rearrange(
            tensor=img, pattern="(h gh) (w gw) c -> (h w) gw gh c", gw=grid_w, gh=grid_h
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

    def anomaly_synthesis(
        self, source_img: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Synthesize an anomaly on the source image.

        This function creates an anomaly on the source image by generating a mask,
        applying visual inconsistencies, and blending the original image with
        the augmented image based on the mask.

        Args:
            source_img (np.ndarray): The source image on which the anomaly will be synthesized.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing the synthesized anomalous image and the applied mask.
        """

        # Generate mask
        if not self.json_data["use_mask"]:
            target_foreground_mask = self.generate_partial_foremask(source_img)
        else:
            target_foreground_mask = self.generate_target_foreground_mask(source_img)

        perlin_noise_mask = self.generate_perlin_noise_mask()
        mask = perlin_noise_mask * target_foreground_mask
        mask_inversed = 1 - mask

        # Apply Visual Inconsistencies
        if self.self_aug == "dtd-augmentation":
            textual_datapath = np.random.choice(self.textual_datalist, 1)[0]
            augmented_image = cv2.imread(str(textual_datapath))
            augmented_image = cv2.cvtColor(augmented_image, cv2.COLOR_BGR2RGB)
            augmented_image = cv2.resize(augmented_image, dsize=self.img_size)
        else:
            augmented_image = source_img.copy()
        # 3 256 256 -> 256 256 3
        augmented_image = rearrange(augmented_image, "c h w -> h w c")
        augmented_image = self.random_arrange(augmented_image)

        # Blend image and anomaly source
        factor = np.random.uniform(*self.transparency_range, size=1)[0]
        reshaped_mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
        reshaped_reversed_mask = np.repeat(mask_inversed[:, :, np.newaxis], 3, axis=2)

        anomaly_synthesis_img = (
            factor * reshaped_mask * source_img
            + (1 - factor) * reshaped_mask * augmented_image
            + reshaped_reversed_mask * source_img
        ).astype(np.uint8)

        return anomaly_synthesis_img, mask

    def __len__(self):
        return len(self.file_df)
