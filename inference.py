from __future__ import annotations
import os.path
from tqdm import tqdm
from collections import OrderedDict
import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
from scipy.ndimage import zoom
from medpy import metric
from loguru import logger
from utils import CLASS_COLOR_MAPS
from typing import Any

def calc_metric_per_case(pred: np.ndarray, gt: np.ndarray) -> tuple[float, float, float, float]:
    """
    input ndarray shape:
        pred: [depth, height, width]; gt: [depth, height, width]
    output float: (dice, hd95, jaccard, asd)
    """
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        jaccard = metric.binary.jc(pred, gt)
        asd = np.mean([metric.binary.asd(pred, gt), metric.binary.asd(gt, pred)])
        return dice, hd95, jaccard, asd
    elif pred.sum() > 0 and gt.sum() == 0:
        return 1, 0, 1, 0
    else:
        return 0, 0, 0, 0

def test_single_volume(
    model: nn.Module,
    volume: torch.Tensor,
    label: torch.Tensor,
    case_name: str,
    num_classes: int = 9,
    patch_size: list[int] = (224, 224),
    deep_supervision: bool = False,
    device: str = "cuda:0",
    output_folder: str = "testing",
    **kwargs: Any
) -> list[tuple[float, float, float, float]]:
    """
    input tensor shape:
        image: [1, depth, height, width]; label: [1, depth, height, width]

    output list: [(dice, hd95, jaccard, asd), ...]
    """
    volume = volume.squeeze(0).cpu().detach().numpy()
    label = label.squeeze(0).cpu().detach().numpy()

    logger.info("Predicting...")
    prediction = np.zeros_like(label)
    for depth in tqdm(range(volume.shape[0])):
        image_slice = volume[depth, :, :]
        h, w = image_slice.shape
        if h != patch_size[0] or w != patch_size[1]:
            image_slice = zoom(image_slice, (patch_size[0] / h, patch_size[1] / w), order=3)

        if kwargs.get("norm_x_transform", None) is not None:
            input = kwargs.get("norm_x_transform")(image_slice)
        else:
            input = torch.from_numpy(image_slice).unsqueeze(0)
        input = input.unsqueeze(0).float().to(device)

        model.eval()
        with torch.no_grad():
            outputs = model(input)
            if deep_supervision: outputs = sum(outputs)

            out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()

            if h != patch_size[0] or w != patch_size[1]:
                pred = zoom(out, (h / patch_size[0], w / patch_size[1]), order=0)
            else:
                pred = out

            prediction[depth] = pred

    logger.info("Evaluating...")
    metrics = []
    for class_id in tqdm(range(1, num_classes)):
        # noinspection PyTypeChecker
        metrics.append(calc_metric_per_case(prediction == class_id, label == class_id))
    return metrics

def inference(
    model: nn.Module,
    dataloader: DataLoader,
    num_classes: int = 9,
    patch_size: list[int] = (224, 224),
    deep_supervision: bool = False,
    output_folder: str = "testing",
    device: str = "cuda:0",
    **kwargs: Any,
) -> None:
    logger.info(f"Testing iterations: {len(dataloader)}")
    os.makedirs(output_folder, exist_ok=True)

    metric_list = 0.0
    for sample in tqdm(dataloader):
        # noinspection PyTypeChecker
        image, label, case_name = sample["image"], sample["label"], sample['case_name'][0]

        metric_per_case = test_single_volume(
            model=model,
            volume=image,
            label=label,
            num_classes=num_classes,
            case_name=case_name,
            patch_size=patch_size,
            deep_supervision=deep_supervision,
            device=device,
            output_folder=output_folder,
            **kwargs
        )

        # per case
        metric_list += np.array(metric_per_case)
        # noinspection PyTypeChecker
        mean_metric = np.mean(metric_per_case, axis=0)
        logger.info(f"case_name: {case_name} "
                    f"mean_dice: {mean_metric[0]}, "
                    f"mean_hd95: {mean_metric[1]}, "
                    f"mean_jacquard: {mean_metric[2]}, "
                    f"mean_asd: {mean_metric[3]}")

    # per class
    metric_list = metric_list / len(dataloader)
    for class_name, (i, _) in CLASS_COLOR_MAPS[num_classes].items():
        logger.info(f"class_name: {class_name} "
                    f"mean_dice: {metric_list[i - 1][0]}, "
                    f"mean_hd95: {metric_list[i - 1][1]}, "
                    f"mean_jacquard: {metric_list[i - 1][2]}, "
                    f"mean_asd: {metric_list[i - 1][3]}")

    # per metric
    mean_dice = np.mean(metric_list, axis=0)[0]
    mean_hd95 = np.mean(metric_list, axis=0)[1]
    mean_jacquard = np.mean(metric_list, axis=0)[2]
    mean_asd = np.mean(metric_list, axis=0)[3]
    logger.info(f"Testing performance: "
                f"mean_dice: {mean_dice}, "
                f"mean_hd95: {mean_hd95}, "
                f"mean_jacquard: {mean_jacquard}, "
                f"mean_asd: {mean_asd}")

def get_model(ckpt: str, **kwargs: Any) -> nn.Module:
    from model import build_model

    state_dict = OrderedDict()
    for k, v in torch.load(ckpt, map_location="cpu")["state_dict"].items():
        state_dict[k.replace("_model.", "")] = v

    model = build_model(**kwargs)
    model.load_state_dict(state_dict)
    logger.info(f"Loaded model checkpoint: {ckpt}")
    return model

def test_acdc(ckpt: str) -> None:
    from dataset_acdc import ACDCDataset
    from torchvision.transforms import transforms

    norm_x_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    output_folder = "testing_acdc"
    logger.add(os.path.join(output_folder, "testing.log"))

    device = "cuda:0"
    model = get_model(ckpt=ckpt, in_channels=3, num_classes=4).to(device)
    dataset = ACDCDataset(base_dir="dataset/acdc", split="test")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    inference(
        model=model,
        dataloader=dataloader,
        num_classes=4,
        output_folder=output_folder,
        device=device,
        norm_x_transform=norm_x_transform,
    )

def test_synapse(ckpt: str) -> None:
    from dataset_synapse import SynapseDataset
    from torchvision.transforms import transforms

    norm_x_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    output_folder = "testing_synapse"
    logger.add(os.path.join(output_folder, "testing.log"))

    device = "cuda:0"
    model = get_model(ckpt=ckpt, in_channels=3, num_classes=9).to(device)
    dataset = SynapseDataset(base_dir="dataset/synapse/test_vol", split="test_vol")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    inference(
        model=model,
        dataloader=dataloader,
        num_classes=9,
        output_folder=output_folder,
        device=device,
        norm_x_transform=norm_x_transform,
    )

if __name__ == '__main__':
    # test_synapse(ckpt="log/msvm-unet-synapse/checkpoints/epoch=259-val_mean_dice=0.8500.ckpt")
    test_acdc(ckpt="log/msvm-unet-acdc/checkpoints/epoch.219-val_mean_dice.0.9258.ckpt")
