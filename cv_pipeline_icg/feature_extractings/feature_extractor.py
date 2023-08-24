from abc import ABC, abstractmethod

import numpy as np

import typing as tp


class FeatureExtractor(ABC):

    @abstractmethod
    def extract(self, image: np.ndarray) -> np.ndarray:
        pass
