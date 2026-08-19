import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import glob
from algorithm.utils.data.exceptions import *


class FullImages:
    def __init__(self, data_path="./data/regions", resize=False, height=None, width=None, classes_path=None):

        self.data_path = data_path

        self.resize = resize
        self.dims = height, width

        self.classes_path = classes_path

        if self.resize:
            height, width = self.dims
            if self.dims[0] is None or self.dims[1] is None:
                raise NotCorrectResizeException()
            if self.dims[0] == 'auto':
                height = self.__set_resize_parameters("height")
            if self.dims[1] == 'auto':
                width = self.__set_resize_parameters("width")
            self.dims = (height, width)

        if os.path.exists(self.data_path):
            self.images = []
            image_paths = glob.glob(os.path.join(self.data_path, '*.jpg'))
            for img_path in image_paths:
                if self.resize:
                    img = image.load_img(img_path, target_size=self.dims)
                else:
                    img = image.load_img(img_path)
                img_array = image.img_to_array(img)
                self.images.append(img_array)
            self.labels = sorted(self.__get_class_names(self.data_path))
        else:
            raise NotFoundDirectoryException(self.data_path)



    def __len__(self):
        return len(self.images)

    def get_set(self,):
        return self.images

    def print_item(self, obj):
        return np.uint8(obj)


    def __set_resize_parameters(self, param):
        if param == "height":
            return 128
        if param == "width":
            return 128

    def __get_class_names(self, path):
        try:
            names = glob.glob(os.path.join(self.data_path, '*.txt'))[0]
        except IndexError:
            names = glob.glob(os.path.join(self.classes_path, '*.txt'))[0]
        return [line.strip() for line in open(names, 'r')]

class FullImage:
    def __init__(self, data_path="./data/region", resize=False, height=None, width=None, classes_path=None):

        self.data_path = data_path

        self.resize = resize
        self.dims = height, width

        self.classes_path = classes_path

        if self.resize:
            height, width = self.dims
            if self.dims[0] is None or self.dims[1] is None:
                raise NotCorrectResizeException()
            if self.dims[0] == 'auto':
                height = self.__set_resize_parameters("height")
            if self.dims[1] == 'auto':
                width = self.__set_resize_parameters("width")
            self.dims = (height, width)

        if os.path.exists(self.data_path):
            img_path = self.data_path
            if os.path.splitext(img_path)[1].lower() == '.jpg':
                if self.resize:
                    img = image.load_img(img_path, target_size=self.dims)
                else:
                    img = image.load_img(img_path)
                img_array = image.img_to_array(img)
                self.image = img_array
                self.labels = sorted(self.__get_class_names(self.data_path))
            else:
                raise NotCorrectImageFormatException(img_path)
        else:
            raise NotFoundFileException(self.data_path)



    def __len__(self):
        return 1

    def get_element(self,):
        return self.image

    def print_item(self, obj):
        return np.uint8(obj)


    def __set_resize_parameters(self, param):
        if param == "height":
            return 128
        if param == "width":
            return 128

    def __get_class_names(self, path):
        try:
            names = glob.glob(os.path.join(self.data_path, '*.txt'))[0]
        except IndexError:
            names = glob.glob(os.path.join(self.classes_path, '*.txt'))[0]
        return [line.strip() for line in open(names, 'r')]


def sliding_window(image, window_dims, stride):
    window_height, window_width = window_dims
    windows = []
    positions = []
    h, w = image.shape[:2]

    full_slides_h = (h - window_height) // stride + 1
    full_slides_w = (w - window_width) // stride + 1

    for y in range(0, full_slides_h * stride, stride):
        for x in range(0, full_slides_w * stride, stride):
            window = image[y:y + window_height, x:x + window_width]
            windows.append(window)
            positions.append((x, y))

    if h % stride != 0:
        y = h - window_height
        for x in range(0, full_slides_w * stride, stride):
            window = image[y:y + window_height, x:x + window_width]
            windows.append(window)
            positions.append((x, y))

    if w % stride != 0:
        x = w - window_width
        for y in range(0, full_slides_h * stride, stride):
            window = image[y:y + window_height, x:x + window_width]
            windows.append(window)
            positions.append((x, y))

    return windows, positions


def preprocess_windows(windows, positions, batch_size=100, resize=False, normalize=False, img_height=None, img_width=None, mean=None, printable_object=None):
    windows_batched = []
    positions_batched = []
    batch_windows__data = []
    batch_windows__po = []
    batch_positions = []
    for i in range(len(windows)):
        img = image.img_to_array(windows[i])
        img = tf.convert_to_tensor(img)
        if resize:
            img = tf.image.resize(img, [img_height, img_width])
        if normalize:
            img = img*mean
        batch_windows__data.append(img)
        batch_windows__po.append(printable_object)
        batch_positions.append(positions[i])
        if len(batch_windows__data) == batch_size:
            batch_tens = {
                'data': tf.stack(batch_windows__data), 
                'print_object': tf.stack(batch_windows__po)
                }
            windows_batched.append(batch_tens)
            positions_batched.append(batch_positions)
            batch_windows__data = []
            batch_windows__po = []
            batch_positions = []
    if len(batch_windows__data) != 0:
        batch_tens = {
                'data': tf.stack(batch_windows__data), 
                'print_object': tf.stack(batch_windows__po)
                }
        windows_batched.append(batch_tens)
        positions_batched.append(batch_positions)
    batches = list(zip(windows_batched, positions_batched))
    return batches


def preprocess_image(image_, resize=False, normalize=False, img_height=None, img_width=None, mean=None, printable_object=None):
    img = image.img_to_array(image_)
    img = tf.convert_to_tensor(img)
    if resize:
        img = tf.image.resize(img, [img_height, img_width])
    if normalize:
            img = img*mean
    image_data = tf.expand_dims(img, axis=0)
    po = tf.expand_dims(printable_object, axis=0) 
    batch_tens = {
        'data': image_data, 
        'print_object': po
        }
    return batch_tens