import numpy as np
import cv2
import typing as tp
import torch

from genetic_algorithm.GeneticAlgorithm import GeneticAlgorithm


class IncorrectShapeImage(Exception):
    pass


def mask_tgi(img: np.ndarray, b: float, r: float, threshold: float) -> np.ndarray:
    """
    :param img: image.
    :param b: blue coefficient.
    :param r: red coefficient.
    :param threshold: threshold.
    :return: mask for this tgi index.
    """
    B: np.ndarray = img[:, :, 0]
    G: np.ndarray = img[:, :, 1]
    R: np.ndarray = img[:, :, 2]

    TGI: np.ndarray = b * B + G + r * R

    mask: np.ndarray = np.ones((img.shape[0], img.shape[1]))
    mask *= TGI

    _, mask = cv2.threshold(mask, threshold, 255, cv2.THRESH_BINARY)
    mask = mask.astype(np.uint8)
    mask = np.where(mask > 0, 1, 0).astype(np.uint8)

    return mask


def iou(mask1: torch.Tensor, mask2: torch.Tensor) -> float:
    intersection = (mask1 * mask2).sum()
    if intersection == 0:
        return 0.0
    union = torch.logical_or(mask1, mask2).to(torch.int).sum()
    return float(intersection / union)


def get_tgi(img_seq: tp.Sequence[np.ndarray], mask_seq: tp.Sequence[np.ndarray], cuda_name: bool = '',
            mut_force: float = 0.5, prob_mut: float = 0.75, internal_mut_bound: float = 1e-4,
            max_iter: int = 75, delta_converged: float = 5e-3, population_count: int = 30,
            initial_population: np.ndarray | None = None,
            initial_population_bound: float | np.ndarray = np.array([1, 1, 25]),
            iter_increase: int = 30, increase_force: float = 0.2, decrease_increase_force: float = 0.5,
            silent: bool = False) -> GeneticAlgorithm:
    """
    :param cuda_name: if not '', then the calculation of jaccard score is done using selected gpu.
    :param img_seq: sequence of images.
    :param mask_seq: sequence of masks.
    the remaining parameters are the parameters of the GeneticAlgorithm.
    Target function - Jaccard score.
    :return: best tgi index.
    """
    max_size = max(img.shape[1] for img in img_seq), max(img.shape[0] for img in img_seq)
    img = np.vstack([cv2.resize(img, max_size) for img in img_seq])
    mask = np.vstack([cv2.resize(mask, max_size) for mask in mask_seq])
    dim: int = 3
    gen_alg: GeneticAlgorithm = GeneticAlgorithm(lambda b, r, thr:
                                                 iou(torch.from_numpy(mask.clip(0, 1)).cuda(cuda_name),
                                                     torch.from_numpy(mask_tgi(img, b, r, thr)).cuda(cuda_name))
                                                 if cuda_name
                                                 else iou(torch.from_numpy(mask.clip(0, 1)),
                                                          torch.from_numpy(mask_tgi(img, b, r, thr))),
                                                 mut_force, prob_mut, internal_mut_bound, max_iter, delta_converged,
                                                 population_count, dim, initial_population, initial_population_bound,
                                                 iter_increase, increase_force, decrease_increase_force, silent)
    gen_alg.run()
    return gen_alg
