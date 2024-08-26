from __future__ import annotations
import torch
import numpy as np
from calflops import calculate_flops

np.bool = bool

SYNAPSE_CLASS_COLOR_MAP = {
    "aorta": (1, [30, 144, 255]),
    "gallbladder": (2, [0, 255, 0]),
    "left_kidney": (3, [255, 0, 0]),
    "right_kidney": (4, [0, 255, 255]),
    "liver": (5, [255, 0, 255]),
    "pancreas": (6, [255, 255, 0]),
    "spleen": (7, [128, 0, 255]),
    "stomach": (8, [255, 128, 0])
}

ACDC_CLASS_COLOR_MAP = {
    "RV": (1, [30, 144, 255]),
    "Myo": (2, [0, 255, 0]),
    "LV": (3, [255, 0, 0]),
}

CLASS_COLOR_MAPS = {
    4: ACDC_CLASS_COLOR_MAP,
    9: SYNAPSE_CLASS_COLOR_MAP
}

def dc(result: torch.Tensor, reference: torch.Tensor) -> float:
    result = torch.atleast_1d(result.type(torch.bool))
    reference = torch.atleast_1d(reference.type(torch.bool))

    intersection = torch.count_nonzero(result & reference)

    size_i1 = torch.count_nonzero(result)
    size_i2 = torch.count_nonzero(reference)

    try:
        dc = (2. * intersection / float(size_i1 + size_i2)).item()
    except ZeroDivisionError:
        dc = 0.0

    return dc

def calc_dice_gpu(pred: torch.Tensor, gt: torch.Tensor) -> float:
    """
    input tensor shape:
        pred: [[d,] h, w]; gt: [[d,] h, w]
    """
    if pred.sum() > 0 and gt.sum() > 0:
        return dc(pred, gt)
    elif pred.sum() > 0 and gt.sum() == 0:
        return 1
    return 0

def print_flops_params(
    model: torch.nn.Module,
    input_shape: tuple[int, ...] = (1, 3, 224, 224),
    output_as_string: bool = True,
    output_precision: int = 4,
    verbose: bool = True,
) -> None:
    flops, macs, params = calculate_flops(
        model=model,
        input_shape=input_shape,
        output_as_string=output_as_string,
        output_precision=output_precision,
        print_results=verbose,
        print_detailed=verbose
    )
    print(f"FLOPs: {flops}, MACs: {macs}, Params: {params}")
