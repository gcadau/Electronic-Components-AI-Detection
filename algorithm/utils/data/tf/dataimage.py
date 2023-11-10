import os
import tensorflow as tf
from tensorflow.python.data import AUTOTUNE
from tensorflow.python.framework.errors_impl import InvalidArgumentError
from PIL import Image
from algorithm.utils.data.exceptions import *


class DataImage:
    def __init__(self, data_path="./data", split=1, transform=None, normalize=False, mean=None, std=None,
                 resize=False, height=None, width=None, name=None, format=None, buffer_size=500, batch_size=32):

        self.__SEPARATOR = "_"
        self.__LABEL_ID = 0

        self.__format = format
        self.dataset_name = name
        self.data_path = data_path

        default_split, only_train_split = 0.2, 1

        if os.path.exists(self.data_path):
            self.data = tf.data.Dataset.list_files(f"{self.data_path}\\*", shuffle=False)
        else:
            raise NotFoundDirectoryException(self.data_path)

        self.split = split
        if self.split is None:
            self.split = default_split
        if isinstance(self.split, str):
            if self.split.lower() == 'auto':
                self.split = default_split
        else:
            if not isinstance(self.split, int) and not isinstance(self.split, float):
                raise NotCorrectSplitException(self.split, "no number")
            if self.split <= 0 or self.split > 1:
                raise NotCorrectSplitException(self.split, "wrong number")
        self.data = self.data.shuffle(len(self.data), reshuffle_each_iteration=False)
        train_data = self.data.skip(int(len(self.data) * self.split))
        if self.split != 1:
            val_data = self.data.take(int(len(self.data) * self.split))
        else:
            val_data = None
        self.data_splitted = {"train": train_data, "validation": val_data}

        self.labels = self.__get_class_names(self.data)

        self.resize = resize
        self.dims = height, width

        self.normalize = normalize  # to be implemented.

        self.transforms = transform  # to be implemented various transforms.
        if self.transforms is None:
            self.transforms = []
        if self.resize:
            height, width = self.dims
            if self.dims[0] is None or self.dims[1] is None:
                raise NotCorrectResizeException()
            if self.dims[0] == 'auto':
                height = self.__set_resize_parameters("height")
            if self.dims[1] == 'auto':
                width = self.__set_resize_parameters("width")
            self.dims = (height, width)
            self.transforms.append(f"Resize(size=({height}, {width})")
        if self.normalize:
            self.transforms.append(f"Normalize(mean={mean}, std={std})")

        self.buffer_size = buffer_size
        self.batch_size = batch_size

        self.__format = Image.open(next(iter(self.data.take(1))).numpy()).mode

    def __len__(self):
        return self.data.cardinality()

    def get_set(self, split="train"):
        ds = self.data_splitted[split.lower()]
        ds = ds.map(self.__process_path, num_parallel_calls=AUTOTUNE)
        ds = self.configure_for_performance(ds)
        return ds

    def __repr__(self):
        repr = ""
        dataset = self.dataset_name if self.dataset_name is not None else self.data_path
        number = f"Number of points: {self.__len__()}"
        loc = f"Root location: {self.data_path}"
        split = f"Split: {self.split}"
        trans = f"Transform used:"
        for t in self.transforms:
            trans += "\t" + t + "\n"
        if not self.transforms:
            trans += "\t" + "-"
        for s in (dataset, number, loc, split, trans):
            if s != dataset:
                s = '\t' + s
            repr += s + "\n"
        return repr

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

    def __get_class_names(self, dataset):
        return list(set([os.path.basename(el.numpy().decode('utf-8')).split(self.__SEPARATOR)[self.__LABEL_ID] for el in
                         dataset]))

    def __get_label(self, file_path):
        file_name = tf.strings.split(file_path, os.path.sep)[-1]
        name = tf.strings.split(file_name, self.__SEPARATOR)[self.__LABEL_ID]
        return name

    def __decode_img(self, img, dims, file_path):
        img_height, img_width = dims
        try:
            num_channels = self.__set_channels_parameters(self.__format)
            img = tf.io.decode_jpeg(img, channels=num_channels)
        except InvalidArgumentError:
            raise NotCorrectImageFormatException(file_path)
        if img_height is not None:
            t = tf.image.resize(img, [img_height, img_width])
        else:
            t = tf.image
        return t

    @tf.autograph.experimental.do_not_convert
    def __process_path(self, file_path):
        label = self.__get_label(file_path)
        img = tf.io.read_file(file_path)
        height, width = None, None
        if self.resize:
            height, width = self.dims
        img = self.__decode_img(img, (height, width), file_path)
        return img, label

    def __set_channels_parameters(self, param):
        n_channels = {"RGB": 3, "Grayscale": 1, "RGBA": 4}
        try:
            return n_channels[self.__format]
        except KeyError:
            return 3

    def configure_for_performance(self, ds):
        ds = ds.cache()
        ds = ds.shuffle(self.buffer_size)
        ds = ds.batch(self.batch_size)
        ds = ds.prefetch(buffer_size=AUTOTUNE)
        return ds
