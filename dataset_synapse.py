from __future__ import annotations
import os
import h5py
import numpy as np
from scipy.ndimage import zoom
import torch
from torch.utils.data import Dataset
import imgaug as ia
import imgaug.augmenters as iaa
from typing import Callable, Sequence, Any

np.bool = bool

def resize_mask(mask: torch.Tensor, scale: Sequence[float, float]) -> torch.Tensor:
    mask = mask.numpy()[0, :, :]
    return torch.from_numpy(zoom(mask, scale, order=0)[None])

def mask_to_onehot(mask: np.ndarray) -> np.ndarray:
    """Converts a segmentation mask (H, W, C) to (H, W, K) where the last dim is a one
    hot encoding vector, C is usually 1 or 3, and K is the number of class.
    """
    semantic_map = []
    mask = np.expand_dims(mask, -1)
    for colour in range(9):
        equality = np.equal(mask, colour)
        class_map = np.all(equality, axis=-1)
        semantic_map.append(class_map)
    semantic_map = np.stack(semantic_map, axis=-1).astype(np.int32)
    return semantic_map

def augment_seg(img_aug: iaa.Augmenter, img: np.ndarray, seg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    seg = mask_to_onehot(seg)
    aug_det = img_aug.to_deterministic()
    image_aug = aug_det.augment_image(img)

    seg_map = ia.SegmentationMapsOnImage(seg, shape=img.shape)
    seg_map_aug = aug_det.augment_segmentation_maps(seg_map)
    seg_map_aug = seg_map_aug.get_arr()
    seg_map_aug = np.argmax(seg_map_aug, axis=-1).astype(np.float32)
    return image_aug, seg_map_aug

class SynapseDataset(Dataset):
    def __init__(
        self,
        base_dir: str,
        split: str = "train",
        list_dir: str = "./lists/lists_Synapse",
        img_size: int = 224,
        norm_x_transform: Callable[..., torch.Tensor] | None = None,
        norm_y_transform: Callable[..., torch.Tensor] | None = None,
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
        if self.split == "train":
            slice_name = self.sample_list[idx].strip("\n")
            data_path = os.path.join(self.data_dir, slice_name + ".npz")
            data = np.load(data_path)
            image, label = data["image"], data["label"]
            image, label = augment_seg(self.img_aug, image, label)
            x, y = image.shape
            if x != self.img_size or y != self.img_size:
                image = zoom(image, (self.img_size / x, self.img_size / y), order=3)
                label = zoom(label, (self.img_size / x, self.img_size / y), order=0)
        else:
            vol_name = self.sample_list[idx].strip('\n')
            filepath = self.data_dir + "/{}.npy.h5".format(vol_name)
            data = h5py.File(filepath)
            image, label = data["image"][:], data["label"][:]

        sample = {"image": image, 'label': label}
        if self.norm_x_transform is not None:
            sample["image"] = self.norm_x_transform(sample["image"].copy())
        if self.norm_y_transform is not None:
            sample["label"] = self.norm_y_transform(sample["label"].copy())

        if self.deep_supervision_scales is not None:
            sample["label"] = [resize_mask(sample["label"], scale) for scale in self.deep_supervision_scales]

        sample["case_name"] = self.sample_list[idx].strip("\n")
        return sample
