from abc import ABC, abstractmethod

import numpy as np

import typing as tp


class Segmenter(ABC):

    @abstractmethod
    def fit(self, images: tp.Sequence[np.ndarray], masks: tp.Sequence[np.ndarray]) -> None:
        """
        Fit model.
        :param images: dataset for train.
        :param masks: masks for train.
        :return: None
        """
        pass

    @abstractmethod
    def predict(self, images: tp.Sequence[np.ndarray]) -> tp.Sequence[np.ndarray]:
        """
        Predict mask for images.
        :param images: sequence of images.
        :return: sequence of masks (values 0 or 1).
        """
        pass

    @abstractmethod
    def save_model(self, model_path: str) -> None:
        """
        Save model.
        :param model_path: model path.
        :return: None
        """
        pass

    @abstractmethod
    def load_model(self, model_path: str) -> None:
        """
        Load model.
        :param model_path: model path.
        :return: None
        """
        pass

    def get_segmentation_image(self, images: tp.Sequence[np.ndarray],
                               negative: bool = False,
                               mask_processing: tp.Callable[[tp.Sequence[np.ndarray]],
                                                                tp.Sequence[np.ndarray]] | None = None):
        """
        Returns segmented images.
        :param images: images for segmentation.
        :param negative: if False, then multiply by mask, otherwise by 1 - mask. Defaults False.
        :param mask_processing: mask preprocessing function.
        :return: segmented images
        """
        masks = self.predict(images)
        if mask_processing is not None:
            masks = mask_processing(masks)
        segment_images: list[np.ndarray] = list()
        for mask, image in zip(masks, images):
            mask = np.where(mask == 0, 0, 1) if not negative else np.where(mask != 0, 0, 1)
            segment_images.append(mask * image)
        return segment_images
