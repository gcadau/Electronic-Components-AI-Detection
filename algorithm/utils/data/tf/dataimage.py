import os
import tensorflow as tf
from PIL import Image, UnidentifiedImageError
from algorithm.utils.data.exceptions import *


class DataImage:
    def __init__(self, data_path="./data", split="train", transform=None, normalize=False, mean=None, std=None,
                 resize=False, height=None, width=None, name=None, format=None):
        self.__format = format()
        # To be implemented

    def __len__(self):
        # To be implemented
        pass

    def __getitem__(self, index):
        # To be implemented
        pass

    def __repr__(self):
        # To be implemented
        pass

    def print_item(self, index):  # TO BE IMPLEMENTED.
        # To be implemented
        pass

    def __set_normalization_parameters(self, param):
        normalization_values = \
            {
                "RGB":
                    {
                        "mean": [0.485, 0.456, 0.406],
                        "std": [0.229, 0.224, 0.225]
                    },
                "Grayscale":
                    {
                        "mean": [0.5],
                        "std": [0.5]
                    },
                "RGBA":
                    {
                        "mean": [0.485, 0.456, 0.406, 0.0],
                        "std": [0.229, 0.224, 0.225, 1.0]
                    }
            }
        try:
            return normalization_values[self.__format][param]
        except KeyError:
            return [0.5, 0.5, 0.5]

    def __set_resize_parameters(self, param):
        if param == "height":
            return 128
        if param == "width":
            return 128
