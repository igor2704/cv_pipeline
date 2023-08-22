import torch


def iou(mask1: torch.Tensor, mask2: torch.Tensor) -> float:
    intersection = (mask1 * mask2).sum()
    if intersection == 0:
        return 0.0
    union = torch.logical_or(mask1, mask2).to(torch.int).sum()
    return float(intersection / union)
