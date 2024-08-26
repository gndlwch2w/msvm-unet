from __future__ import annotations
import os
import numpy as np
from scipy.ndimage import zoom
from torch.utils.data import Dataset
import imgaug.augmenters as iaa
from dataset_synapse import resize_mask, augment_seg
from typing import Callable, Any

np.bool = bool

class ACDCDataset(Dataset):
    def __init__(
        self,
        base_dir: str,
        split: str = "train",
        list_dir: str = "./lists/lists_ACDC",
        img_size: int = 224,
        norm_x_transform: Callable[..., Any] = None,
        norm_y_transform: Callable[..., Any] = None,
        deep_supervision_scales: list[[list[float, float]]] | None = None,
    ) -> None:
        self.norm_x_transform = norm_x_transform
        self.norm_y_transform = norm_y_transform
        self.split = split
        self.sample_list = open(os.path.join(list_dir, self.split + ".txt")).readlines()
        self.data_dir = base_dir
        self.img_size = img_size
        self.deep_supervision_scales = deep_supervision_scales

        self.img_aug = iaa.SomeOf((0, 4), [
            iaa.Flipud(0.5, name="Flipud"),
            iaa.Fliplr(0.5, name="Fliplr"),
            iaa.AdditiveGaussianNoise(scale=0.005 * 255),
            iaa.GaussianBlur(sigma=1.0),
            iaa.LinearContrast((0.5, 1.5), per_channel=0.5),
            iaa.Affine(scale={"x": (0.5, 2), "y": (0.5, 2)}),
            iaa.Affine(rotate=(-40, 40)),
            iaa.Affine(shear=(-16, 16)),
            iaa.PiecewiseAffine(scale=(0.008, 0.03)),
            iaa.Affine(translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)})
        ], random_order=True)

    def __len__(self) -> int:
        return len(self.sample_list)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """
        output tensor shape:
            {
                "case_name": str,
                "image": [1, height, width] | [depth, height, width],
                "label": [1, height, width] | [depth, height, width]
            }
        """
        filename = self.sample_list[idx].strip("\n")
        filepath = os.path.join(self.data_dir, self.split, filename)
        data = np.load(filepath)
        image, label = data["img"].astype(np.float32), data["label"].astype(np.float32)

        if self.split == "train":
            image, label = augment_seg(self.img_aug, image, label)
        if self.split in ["train", "valid"]:
            h, w = image.shape
            if h != self.img_size or w != self.img_size:
                image = zoom(image, (self.img_size / h, self.img_size / w), order=3)
                label = zoom(label, (self.img_size / h, self.img_size / w), order=0)

        sample = {"image": image, "label": label}
        if self.norm_x_transform is not None:
            sample["image"] = self.norm_x_transform(sample["image"].copy())
        if self.norm_y_transform is not None:
            sample["label"] = self.norm_y_transform(sample["label"].copy())

        if self.deep_supervision_scales is not None:
            sample["label"] = [resize_mask(sample["label"], scale) for scale in self.deep_supervision_scales]

        sample["case_name"] = self.sample_list[idx].strip("\n")
        return sample
