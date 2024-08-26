from __future__ import annotations
import os
from os.path import join
from collections import defaultdict
import torch
import numpy as np
import monai
from monai import data
from monai.metrics import CumulativeAverage
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from loguru import logger
from typing import Callable, Optional
from lr_scheduler import LR_SCHEDULERS
from loss import LOSSES, DeepSupervisionWrapper
from eval import eval_single_volume
from model import build_model
from dataset_acdc import ACDCDataset
from torchvision.transforms import transforms

torch.set_float32_matmul_precision("medium")
device: str = "cuda" if torch.cuda.is_available() else "cpu"

OPTIMIZERS = {
    "Adam": torch.optim.Adam,
    "SGD": torch.optim.SGD,
    "RMSprop": torch.optim.RMSprop,
    "AdamW": torch.optim.AdamW
}

class ACDC(L.LightningModule):
    def __init__(self, name: str) -> None:
        super(ACDC, self).__init__()
        self.name = name
        self.num_classes = 4
        self.max_epochs = 300
        self.freeze_encoder_epochs = 10
        self.deep_supervision = False

        self._model = build_model(
            num_classes=self.num_classes,
            deep_supervision=self.deep_supervision,
        ).to(device)

        self.deep_supervision_scales: Optional[list[list[float, float]]] = None
        if self.deep_supervision:
            self.deep_supervision_scales = self._model.deep_supervision_scales

        self.build_loss()
        self.tl_metric = CumulativeAverage()
        self.tl_metric = CumulativeAverage()
        self.vs_metric = defaultdict(lambda: defaultdict(list))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._model(x)

    def prepare_data(self) -> None:
        self.norm_x_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

        self.train_dataset = ACDCDataset(
            base_dir="dataset/acdc",
            split="train",
            norm_x_transform=self.norm_x_transform,
            norm_y_transform=transforms.ToTensor(),
            deep_supervision_scales=self.deep_supervision_scales if self.deep_supervision else None
        )

        self.val_dataset = ACDCDataset(
            base_dir="dataset/acdc",
            split="test",
            # norm_x_transform=self.norm_x_transform,
            # norm_y_transform=transforms.ToTensor(),
        )

    def train_dataloader(self) -> data.DataLoader:
        tdl_0 = {
            "batch_size": 32,
            "num_workers": 6,
            "shuffle": True,
            "pin_memory": True,
            "persistent_workers": True,
            "worker_init_fn": None
        }

        tdl = tdl_0
        logger.info(f"Training dataloader: {tdl}")
        return data.DataLoader(self.train_dataset, **tdl)

    def val_dataloader(self) -> data.DataLoader:
        vdl_0 = {
            "batch_size": 1,
            "shuffle": False,
            "pin_memory": True,
            "num_workers": 1,
            "persistent_workers": True
        }

        vdl = vdl_0
        logger.info(f"Validation dataloader: {vdl}")
        return data.DataLoader(self.val_dataset, **vdl)

    def build_loss(self):
        loss_0 = ("DiceCELoss", {
            "ce_weight": 0.4,
            "dc_weight": 0.6,
            "ce_class_weights": None,
            "dc_class_weights": None
        })

        self.loss = LOSSES[loss_0[0]](**loss_0[1])
        if self.deep_supervision:
            weights = [1.] * len(self.deep_supervision_scales)
            self.loss = DeepSupervisionWrapper(self.loss, weights)

    @property
    def criterion(self) -> Callable[..., torch.Tensor]:
        return self.loss

    def configure_optimizers(self) -> dict:
        optimizer_0 = ("AdamW", {
            "lr": 5e-4,
            "weight_decay": 1e-4,
            "eps": 1e-8,
            "amsgrad": False,
            "betas": (0.9, 0.999)
        })
        optimizer = OPTIMIZERS[optimizer_0[0]](self._model.parameters(), **optimizer_0[1])

        scheduler_1 = ("CosineAnnealingLR", {
            "T_max": self.max_epochs,
            "eta_min": 1e-6
        })
        scheduler = LR_SCHEDULERS[scheduler_1[0]](optimizer, **scheduler_1[1])

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler
        }

    def log_and_logger(self, name: str, value: ..., **kwargs: ...) -> None:
        self.log(name, value, **kwargs)
        logger.info(f"{name}: {value}")

    def on_train_epoch_start(self) -> None:
        if self.current_epoch < self.freeze_encoder_epochs:
            self._model.freeze_encoder()
        else:
            self._model.unfreeze_encoder()
        super().on_train_epoch_start()

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        image, label = batch["image"].to(device), batch["label"]
        label = [e.to(device) for e in label] if self.deep_supervision else label.to(device)

        pred = self.forward(image)
        loss = self.criterion(pred, label)

        self.log("loss", loss.item(), prog_bar=True)
        self.tl_metric.append(loss.item())
        self.log("lr", self.optimizers().param_groups[0]["lr"], prog_bar=True)

        return loss

    def on_train_epoch_end(self) -> None:
        tl = self.tl_metric.aggregate().item()
        self.log_and_logger("mean_train_loss", tl)
        self.tl_metric.reset()

    def validation_step(self, batch: dict[str, torch.Tensor]) -> None:
        volume, label, case_name = batch["image"], batch["label"], batch["case_name"][0]
        metric = eval_single_volume(
            model=self._model,
            volume=volume,
            label=label,
            case_name=case_name,
            num_classes=self.num_classes,
            output=join(self.name, str(self.current_epoch)),
            patch_size=(224, 224),
            device=device,
            deep_supervision=self.deep_supervision,
            norm_x_transform=getattr(self, "norm_x_transform", None),
        )

        for metric_name, class_metric in metric.items():
            for class_name, value in class_metric.items():
                self.vs_metric[metric_name][class_name].append(np.mean(value))

    def on_validation_epoch_end(self) -> None:
        for metric_name, class_metric in self.vs_metric.items():
            avg_metric = []
            for class_name, value in class_metric.items():
                t = np.mean(value)
                self.log(f"val_{metric_name}_{class_name}", t)
                avg_metric.append(t)
            self.log_and_logger(f"val_mean_{metric_name}", np.mean(avg_metric))
        self.vs_metric = defaultdict(lambda: defaultdict(list))

def train(name: str) -> None:
    os.makedirs(name, exist_ok=True)
    logger.add(join(name, "training.log"))

    model = ACDC(name)

    checkpoint_callback = ModelCheckpoint(
        dirpath=join(name, "checkpoints"),
        monitor="val_mean_dice",
        mode="max",
        filename="{epoch:02d}-{val_mean_dice:.4f}",
        save_last=True
    )

    early_stop_callback = EarlyStopping(
        monitor="mean_train_loss",
        mode="min",
        min_delta=0.00,
        patience=15
    )

    trainer = L.Trainer(
        precision=32,
        accelerator=device,
        devices="auto",
        max_epochs=model.max_epochs,
        check_val_every_n_epoch=20,
        gradient_clip_val=None,
        default_root_dir=name,
        callbacks=[checkpoint_callback, early_stop_callback],
        enable_checkpointing=True
    )

    trainer.fit(model, ckpt_path=None)

if __name__ == "__main__":
    L.seed_everything(42)
    monai.utils.set_determinism(42)
    train("/log/msvm-unet-acdc")
