import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.python.data import AUTOTUNE
from tensorflow.python.framework.errors_impl import InvalidArgumentError
from PIL import Image
from algorithm.utils.data.exceptions import *


class DataImage:
    def __init__(self, data_path="./data", split=0, transform=None, normalize=False, mean=None, std=None,
                 resize=False, height=None, width=None, one_hot_encoding=False, name=None, format=None, buffer_size=500, batch_size=32):

        self.__SEPARATOR = "_"
        self.__LABEL_ID = 0

        self.__format = format
        self.dataset_name = name
        self.data_path = data_path

        default_split, only_train_split = 0.2, 0

        if os.path.exists(self.data_path):
            self.data = tf.data.Dataset.list_files(f"{self.data_path}/*", shuffle=False)
            self.__identifier = [os.path.join(self.data_path, n) for n in os.listdir(self.data_path)]
        else:
            raise NotFoundDirectoryException(self.data_path)

        self.split = split
        if self.split is None:
            self.split = default_split
        if isinstance(self.split, str):
            if self.split.lower() == 'auto':
                self.split = default_split
            if self.split.lower() == 'train only':
                self.split = only_train_split
        else:
            if not isinstance(self.split, int) and not isinstance(self.split, float):
                raise NotCorrectSplitException(self.split, "no number")
            if self.split < 0 or self.split >= 1:
                raise NotCorrectSplitException(self.split, "wrong number")
        self.data = self.data.shuffle(len(self.data), reshuffle_each_iteration=False)
        train_data = self.data.skip(int(len(self.data) * self.split))
        if self.split != 0:
            val_data = self.data.take(int(len(self.data) * self.split))
        else:
            val_data = None
        self.data_splitted = {"train": train_data, "validation": val_data}

        self.labels = sorted(self.__get_class_names(self.data))

        self.resize = resize
        self.dims = height, width

        self.normalize = normalize
        self.norms = mean, std

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
            mean, std = self.norms
            if self.norms[0] is None:
                raise NotCorrectNormalizationException()
            if self.norms[0] == 'auto':
                mean = self.__set_normalization_parameters("mean")
            if self.norms[1] == 'auto':
                std = self.__set_normalization_parameters("std")
            self.norms = (mean, std)
            self.transforms.append(f"Normalize(mean={mean}, std={std})")

        self.one_hot_encoding = one_hot_encoding # to be implemented

        self.buffer_size = buffer_size

        self.batch_size = batch_size
        if isinstance(self.batch_size, str):
            if self.batch_size.lower() == 'no batches':
                self.batch_size = 0
            else:
                raise NotCorrectBatchException(self.batch_size)

        self.__format = Image.open(next(iter(self.data.take(1))).numpy()).mode



    def __len__(self):
        return self.data.cardinality()

    def get_set(self, split="train"):
        ds = self.data_splitted[split.lower()]
        if ds is not None:
            ds = ds.map(self.__process_path, num_parallel_calls=AUTOTUNE)
            ds = ds.map(self.__add_identifier, num_parallel_calls=tf.data.AUTOTUNE)
            ds = self.__configure_for_performance(ds)
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

    def print_item(self, id):
        int_id = tf.strings.to_number(id, out_type=tf.int32)
        id_value = int(int_id.numpy())
        image = Image.open(self.__identifier[id_value])
        return image


    def __one_hot_encode(self, label_batch, label_mapping):
        one_hot_label_batch = []
        for label in label_batch:
            label_str = label.numpy() if isinstance(label, tf.Tensor) else label
            label_int = label_mapping[label_str.decode("utf-8")]
            one_hot_label = tf.one_hot(label_int, depth=len(self.labels))
            one_hot_label_batch.append(one_hot_label)
        return tf.stack(one_hot_label_batch)

    @tf.autograph.experimental.do_not_convert
    def apply_one_hot_encoding(self, ds):
        label_mapping = {label: idx for idx, label in enumerate(self.labels)}

        def one_hot_encode_wrapper(image_batch, label_batch):
            label_encoded_batch =  tf.py_function(
                func=lambda lbls: self.__one_hot_encode(lbls, label_mapping),
                inp=[label_batch],
                Tout=(tf.float32)  # Adjust the output data types as needed
            )
            return image_batch, label_encoded_batch

        ds = ds.map(one_hot_encode_wrapper, num_parallel_calls=tf.data.AUTOTUNE)
        return ds


    def __set_normalization_parameters(self, param):
        normalization_values = \
            {
                "RGB":
                    {
                        "mean": [float(1/255), float(1/255), float(1/255)],
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
            return [float(1/255), float(1/255), float(1/255)]

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

    def __decode_img(self, img, dims, norms, file_path):
        img_height, img_width = dims
        mean, std = norms
        try:
            num_channels = self.__set_channels_parameters(self.__format)
            img = tf.io.decode_jpeg(img, channels=num_channels)
        except InvalidArgumentError:
            raise NotCorrectImageFormatException(file_path)
        if img_height is not None:
            t = tf.image.resize(img, [img_height, img_width])
        else:
            t = tf.image
        if mean is not None:
            if std is not None:
                # To be implemented
                pass
            t = t*mean
        return t

    @tf.autograph.experimental.do_not_convert
    def __process_path(self, file_path):
        label = self.__get_label(file_path)
        img = tf.io.read_file(file_path)
        height, width = None, None
        if self.resize:
            height, width = self.dims
        mean, std = None, None
        if self.normalize:
            mean, std = self.norms
        img = self.__decode_img(img, (height, width), (mean, std), file_path)
        return img, label, file_path

    @tf.autograph.experimental.do_not_convert
    def __add_identifier(self, img, label, filepath):
        conditions = tf.equal(self.__identifier, filepath)
        id = tf.where(conditions)
        id = tf.strings.as_string(tf.squeeze(id))
        return {'data': img, 'print_object': id}, label

    def __set_channels_parameters(self, param):
        n_channels = {"RGB": 3, "Grayscale": 1, "RGBA": 4}
        try:
            return n_channels[self.__format]
        except KeyError:
            return 3

    def __configure_for_performance(self, ds):
        ds = ds.cache()
        size = self.buffer_size
        if isinstance(self.buffer_size, str):
            if self.buffer_size == 'auto':
                size = ds[1].shape[0]
        ds = ds.shuffle(size)
        if self.batch_size!=0:
            ds = ds.batch(self.batch_size)
        ds = ds.prefetch(buffer_size=AUTOTUNE)
        return ds
