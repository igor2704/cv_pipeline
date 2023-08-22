import numpy as np
import cv2
import typing as tp
import torch

from genetic_algorithm.GeneticAlgorithm import GeneticAlgorithm


class IncorrectShapeImage(Exception):
    pass


def iou(mask1: torch.Tensor, mask2: torch.Tensor) -> float:
    intersection = (mask1 * mask2).sum()
    if intersection == 0:
        return 0.0
    union = torch.logical_or(mask1, mask2).to(torch.int).sum()
    return float(intersection / union)


def get_threshold_mask(img: np.ndarray, threshold: int) -> np.ndarray:
    return cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), threshold, 255, cv2.THRESH_BINARY)[1]


def get_threshold(img_seq: tp.Sequence[np.ndarray], mask_seq: tp.Sequence[np.ndarray], cuda_name: bool = '',
            mut_force: float = 0.5, prob_mut: float = 0.75, internal_mut_bound: float = 1e-4,
            max_iter: int = 75, delta_converged: float = 1e-5, population_count: int = 75,
            initial_population: np.ndarray | None = None,
            initial_population_bound: float | np.ndarray = 255,
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
    img = np.vstack([cv2.resize(img, max_size) for img in img_seq]).astype('uint8')
    mask = np.vstack([cv2.resize(mask, max_size) for mask in mask_seq]).astype('uint8')
    dim: int = 1
    gen_alg: GeneticAlgorithm = GeneticAlgorithm(lambda thr:
                                                 iou(torch.from_numpy(mask.clip(0, 1)).cuda(cuda_name),
                                                     torch.from_numpy(get_threshold_mask(img,
                                                                                       thr).clip(0, 1)).cuda(cuda_name))
                                                 if cuda_name
                                                 else iou(torch.from_numpy(mask.clip(0, 1)),
                                                          torch.from_numpy(get_threshold_mask(img,
                                                                                              thr).clip(0, 1))),
                                                 mut_force, prob_mut, internal_mut_bound, max_iter, delta_converged,
                                                 population_count, dim, initial_population, initial_population_bound,
                                                 iter_increase, increase_force, decrease_increase_force, silent)
    gen_alg.run()
    return gen_alg
