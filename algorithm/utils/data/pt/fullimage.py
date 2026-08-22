import glob
import os

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from torchvision import transforms

from algorithm.utils.data.exceptions import *

_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


class FullImage:
    def __init__(self, data_path="./data/regions", resize=False, height=None, width=None,
                 format=None):

        self.data_path = data_path
        self.__format = format

        self.resize = resize
        self.dims = height, width

        if self.resize:
            height, width = self.dims
            if self.dims[0] is None or self.dims[1] is None:
                raise NotCorrectResizeException()
            if self.dims[0] == 'auto':
                height = self.__set_resize_parameters("height")
            if self.dims[1] == 'auto':
                width = self.__set_resize_parameters("width")
            self.dims = (height, width)

        if not os.path.exists(self.data_path):
            raise NotFoundDirectoryException(self.data_path)

        self.images = []
        self.image_paths = self.__list_images()
        for img_path in self.image_paths:
            try:
                img = Image.open(img_path)
            except UnidentifiedImageError:
                raise NotCorrectImageFormatException(img_path)
            if self.__format is None:
                self.__format = img.mode
            img = img.convert(self.__format)
            if self.resize:
                img = img.resize((self.dims[1], self.dims[0]))
            self.images.append(transforms.functional.pil_to_tensor(img).to(torch.float32))
        self.labels = sorted(self.__get_class_names(self.data_path))


    def __len__(self):
        return len(self.images)

    def get_set(self):
        return self.images

    def print_item(self, obj):
        if torch.is_tensor(obj):
            obj = obj.detach().cpu().numpy()
        obj = np.asarray(obj)
        if obj.ndim == 3 and obj.shape[0] in (1, 3, 4) and obj.shape[-1] not in (1, 3, 4):
            obj = np.transpose(obj, (1, 2, 0))
        return np.uint8(obj.clip(0, 255))


    def __list_images(self):
        found = []
        for entry in sorted(os.listdir(self.data_path)):
            if os.path.splitext(entry)[1].lower() in _IMAGE_EXTENSIONS:
                found.append(os.path.join(self.data_path, entry))
        return found

    def __set_resize_parameters(self, param):
        if param == "height":
            return 128
        if param == "width":
            return 128

    def __get_class_names(self, path):
        names = glob.glob(os.path.join(self.data_path, '*.txt'))
        if not names:
            raise NotFoundDirectoryException(
                f"{self.data_path} (no .txt file listing the class names)")
        with open(names[0], 'r', encoding='utf8') as f:
            return [line.strip() for line in f if line.strip()]

    def __repr__(self):
        return (f"FullImage\n"
                f"\tNumber of images: {len(self)}\n"
                f"\tRoot location: {self.data_path}\n"
                f"\tClasses: {len(self.labels)}\n"
                f"\tResize: {self.dims if self.resize else '-'}\n")



def _offsets(total, window, stride):
    if window > total:
        raise ValueError(f"window ({window}) is larger than the image ({total})")
    offsets = list(range(0, total - window + 1, stride))
    if offsets[-1] + window < total:
        offsets.append(total - window)
    return offsets


def sliding_window(image, window_dims, stride):
    window_height, window_width = window_dims
    h, w = image.shape[-2:]

    windows, positions = [], []
    for y in _offsets(h, window_height, stride):
        for x in _offsets(w, window_width, stride):
            windows.append(image[..., y:y + window_height, x:x + window_width])
            positions.append((x, y))
    return windows, positions


def preprocess_windows(windows, positions, batch_size=100, resize=False, normalize=False,
                       img_height=None, img_width=None, mean=None, printable_object=None):
    batches = []
    batch_data, batch_po, batch_positions = [], [], []

    if normalize and mean is not None:
        mean_t = torch.as_tensor(mean, dtype=torch.float32).reshape(-1, 1, 1)
    else:
        mean_t = None

    def flush():
        batches.append(({'data': torch.stack(batch_data),
                         'print_object': list(batch_po)},
                        list(batch_positions)))
        batch_data.clear()
        batch_po.clear()
        batch_positions.clear()

    for window, position in zip(windows, positions):
        img = window if torch.is_tensor(window) else torch.as_tensor(np.asarray(window))
        img = img.to(torch.float32)
        if resize:
            img = transforms.functional.resize(img, [img_height, img_width], antialias=True)
        if mean_t is not None:
            img = img * mean_t
        batch_data.append(img)
        batch_po.append(printable_object)
        batch_positions.append(position)
        if len(batch_data) == batch_size:
            flush()

    if batch_data:
        flush()
    return batches
