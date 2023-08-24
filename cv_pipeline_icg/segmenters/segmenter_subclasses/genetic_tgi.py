import typing as tp

import numpy as np
import torch
import cv2

import json

from cv_pipeline_icg.segmenters.segmenter import Segmenter
from cv_pipeline_icg.metrics.segmentation_metrics import iou

# https://github.com/igor2704/GeneticAlgorithm
from GeneticAlgorithm.genetic_algorithm import GeneticAlgorithm
from GeneticAlgorithm.genetic_algorithm.examples import get_tgi, mask_tgi

import matplotlib.pyplot as plt


class NotTrainedModel(Exception):
    pass


class GeneticTgiSegmenter(Segmenter):
    """
    Based by https://github.com/igor2704/GeneticAlgorithm.
    For using this class you need clone this project in your directory.
    """

    def __init__(self, cuda_name: str = '',
                 mut_force: float = 0.5, prob_mut: float = 0.75, internal_mut_bound: float = 1e-4,
                 max_iter: int = 75, delta_converged: float = 5e-3, population_count: int = 30,
                 initial_population: np.ndarray | None = None,
                 initial_population_bound: float | np.ndarray = np.array([1, 1, 25]),
                 iter_increase: int = 30, increase_force: float = 0.2, decrease_increase_force: float = 0.5,
                 silent: bool = False):
        """
        Target function - Jaccard score.

        :param cuda_name: if not '', then the calculation of jaccard score is done using selected gpu.
        :param mut_force: mutation force.
        :param prob_mut: mutation probability.
        :param internal_mut_bound: probability of internal mutation.
                Each attribute of the descendant always varies in the range from
                (-internal_mut_bound% , internal_mut_bound%) inherited.
        :param max_iter: maximum number of iterations (generations)
        :param delta_converged: if the difference between the target function
                values of the best and worst sample in population is less than delta_converged,
                then the search for the best population ends.
        :param population_count: number of samples in a population.
        :param initial_population: initial population.
        :param initial_population_bound: if initial_population is not specified, then the
                initial population is vectors whose components are random numbers from the range
                (-initial_population_bound, initial_population_bound).
        :param iter_increase: number of iteration (population) starting from which the
                mut_force increases.
        :param increase_force: how much does mut_force increase.
        :param decrease_increase_force: how much does increase_force decrease.
        :param silent: whether intermediate information will be displayed.
        """
        self.cuda_name = cuda_name
        self.mut_force = mut_force
        self.prob_mut = prob_mut
        self.internal_mut_bound = internal_mut_bound
        self.max_iter = max_iter
        self.delta_converged = delta_converged
        self.population_count = population_count
        self.initial_population = initial_population
        self.initial_population_bound = initial_population_bound
        self.iter_increase = iter_increase
        self.increase_force = increase_force
        self.decrease_increase_force = decrease_increase_force
        self.silent = silent
        self.model: GeneticAlgorithm | None = None

    def fit(self, images: tp.Sequence[np.ndarray], masks: tp.Sequence[np.ndarray]) -> None:
        max_size = max(img.shape[1] for img in images), max(img.shape[0] for img in images)
        img = np.vstack([cv2.resize(img, max_size) for img in images])
        mask = np.vstack([cv2.resize(mask, max_size) for mask in images])
        dim: int = 3
        self.model = GeneticAlgorithm(lambda b, r, thr:
                                      iou(torch.from_numpy(mask.clip(0, 1)).cuda(self.cuda_name),
                                          torch.from_numpy(mask_tgi(img, b, r, thr)).cuda(self.cuda_name))
                                      if self.cuda_name
                                      else iou(torch.from_numpy(mask.clip(0, 1)),
                                               torch.from_numpy(mask_tgi(img, b, r, thr))),
                                      self.mut_force, self.prob_mut, self.internal_mut_bound, self.max_iter,
                                      self.delta_converged, self.population_count, dim,
                                      self.initial_population, self.initial_population_bound,
                                      self.iter_increase, self.increase_force, self.decrease_increase_force,
                                      self.silent)
        self.model.run()

    def predict(self, images: tp.Sequence[np.ndarray]) -> tp.Sequence[np.ndarray]:
        if self.model is None:
            raise NotTrainedModel('The model has not been trained yet')
        masks: list[np.ndarray] = list()
        for img in images:
            masks.append(img, *self.model.best_sample)
        return masks

    def save_model(self, model_path: str) -> None:
        if self.model is None:
            raise NotTrainedModel('The model has not been trained yet')
        with open(model_path) as f:
            json.dump(self.model.best_sample, f)

    def load_model(self, model_path: str) -> None:
        with open(model_path) as f:
            self.model.best_sample = json.load(f)

    def get_graph(self, figsize: tuple[float, float] = (7, 7)):
        if self.model is None:
            raise NotTrainedModel('The model has not been trained yet')
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(np.array(self.model.get_history())[:, 2], label='best jaccard score value in population')
        plt.xlabel('population (iteration)')
        plt.ylabel('best jaccard score value')
        ax.legend()
        ax.grid()
        plt.show()
        return ax
