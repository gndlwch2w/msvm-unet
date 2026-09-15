import os.path as osp

from torch.utils.data import DataLoader

from .dataset import BaseDataset
from .transform import TRANSFORMS

DATALOADERS = {"default": DataLoader}

SYNAPSE_CLSNAME2COLOR = {
    "Aorta": (1, [255, 0, 0]),
    "GB": (2, [0, 255, 0]),
    "KL": (3, [30, 144, 255]),
    "KR": (4, [0, 255, 255]),
    "Liver": (5, [255, 0, 255]),
    "PC": (6, [255, 255, 0]),
    "SP": (7, [128, 0, 255]),
    "SM": (8, [255, 128, 0]),
}

ACDC_CLSNAME2COLOR = {
    "RV": (1, [30, 144, 255]),
    "Myo": (2, [0, 255, 0]),
    "LV": (3, [255, 0, 0]),
}

CLS2COLOR_MAPPING = {
    4: ACDC_CLSNAME2COLOR,
    9: SYNAPSE_CLSNAME2COLOR,
}

DATASETS = {
    "acdc": {
        "num_classes": 4,
        "root_suffix": osp.join("mis", "acdc"),
        "list_dir": osp.join("lists", "lists_ACDC"),
    },
    "synapse": {
        "num_classes": 9,
        "root_suffix": osp.join("mis", "synapse"),
        "list_dir": osp.join("lists", "lists_Synapse"),
    },
}
