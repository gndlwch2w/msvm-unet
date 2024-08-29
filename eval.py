from __future__ import annotations
import torch
import numpy as np
from collections import defaultdict
from utils import calc_dice_gpu, CLASS_COLOR_MAPS
from scipy.ndimage import zoom
from typing import Any

class SegMeter:
    def __init__(self, num_classes: int) -> None:
        self.num_classes = num_classes
        self.reset()

    def reset(self) -> None:
        self.metric = {
            "dice": (defaultdict(list), calc_dice_gpu),
        }

    def __call__(self, pred: torch.Tensor, label: torch.Tensor) -> None:
        """
        input tensor shape:
            input: [b, 1, h, w]; target: [b, 1, h, w]
        """
        for batch_idx in range(pred.shape[0]):
            y_hat, y = pred[batch_idx], label[batch_idx]
            for class_name, (i, _) in CLASS_COLOR_MAPS[self.num_classes].items():
                for _, (v, f) in self.metric.items():
                    v[class_name].append(f(
                        torch.asarray(y_hat == i, dtype=torch.int),
                        torch.asarray(y == i, dtype=torch.int)
                    ))

    def get_metric(self) -> dict[str, dict[str, list]]:
        """
        output tensor shape:
            {
                "metric name": {
                    "class name": [val1, val2, ...], ...
                }, ...
            }
        """
        result = {}
        for metric_name, (v, _) in self.metric.items():
            result[metric_name] = v
        return result

def eval_single_volume(
    model: torch.nn.Module,
    volume: torch.Tensor,
    label: torch.Tensor,
    num_classes: int,
    patch_size: tuple[int, int] = (224, 224),
    device: str | torch.device = None,
    **kwargs: Any,
) -> dict:
    volume = volume.squeeze(0).cpu().detach().numpy()
    label = label.squeeze(0).cpu().detach().numpy()

    model.eval()
    prediction = np.zeros_like(label)
    for depth in range(volume.shape[0]):
        vol_slice = volume[depth, :, :]
        h, w = vol_slice.shape[0], vol_slice.shape[1]

        if h != patch_size[0] or w != patch_size[1]:
            vol_slice = zoom(vol_slice, (patch_size[0] / h, patch_size[1] / w), order=3)

        if kwargs.get("norm_x_transform", None) is not None:
            input = kwargs.get("norm_x_transform")(vol_slice)
        else:
            input = torch.from_numpy(vol_slice).unsqueeze(0)
        input = input.unsqueeze(0).float().to(device)

        with torch.no_grad():
            out = model(input)
            out = torch.argmax(torch.softmax(out, dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
            if h != patch_size[0] or w != patch_size[1]:
                pred = zoom(out, (h / patch_size[0], w / patch_size[1]), order=0)
            else:
                pred = out

            prediction[depth] = pred

    meter = SegMeter(num_classes=num_classes)
    meter(torch.from_numpy(prediction[None]).to(device), torch.from_numpy(label[None]).to(device))
    metric = meter.get_metric()
    return metric
