import os

import torch
from PIL import Image, UnidentifiedImageError
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms

from algorithm.utils.data.exceptions import *


class DataImage(Dataset):
    def __init__(self, data_path="./data", split=0, transform=None, normalize=False, mean=None,
                 std=None, resize=False, height=None, width=None, one_hot_encoding=False,
                 name=None, format=None, buffer_size=500, batch_size=32, seed=None):

        self.__SEPARATOR = "_"
        self.__LABEL_ID = 0

        self.__format = format
        self.dataset_name = name
        self.data_path = data_path

        default_split, only_train_split = 0.2, 0

        if not os.path.exists(self.data_path):
            raise NotFoundDirectoryException(self.data_path)

        self.data = []
        for f in sorted(os.listdir(self.data_path)):
            path = os.path.join(self.data_path, f)
            try:
                with Image.open(path) as image:
                    if self.__format is None:
                        self.__format = image.mode
                self.data.append(path)
            except UnidentifiedImageError:
                raise NotCorrectImageFormatException(path)
        self.__identifier = list(self.data)

        self.split = split
        if self.split is None:
            self.split = default_split
        if isinstance(self.split, str):
            if self.split.lower() == 'auto':
                self.split = default_split
            elif self.split.lower() == 'train only':
                self.split = only_train_split
            else:
                raise NotCorrectSplitException(self.split, "no number")
        else:
            if not isinstance(self.split, (int, float)) or isinstance(self.split, bool):
                raise NotCorrectSplitException(self.split, "no number")
            if self.split < 0 or self.split >= 1:
                raise NotCorrectSplitException(self.split, "wrong number")

        self.seed = seed
        generator = torch.Generator()
        if seed is not None:
            generator.manual_seed(int(seed))
        order = torch.randperm(len(self.data), generator=generator).tolist()
        n_val = int(len(self.data) * self.split)
        self.data_splitted = {
            "train": order[n_val:],
            "validation": order[:n_val] if self.split != 0 else None,
        }

        self.labels = sorted(self.__get_class_names())
        self.label_mapping = {label: i for i, label in enumerate(self.labels)}

        self.resize = resize
        self.dims = height, width

        self.normalize = normalize
        self.norms = mean, std

        self.transforms = transform
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

        self.one_hot_encoding = one_hot_encoding

        self.buffer_size = buffer_size

        self.batch_size = batch_size
        if isinstance(self.batch_size, str):
            if self.batch_size.lower() == 'no batches':
                self.batch_size = 0
            else:
                raise NotCorrectBatchException(self.batch_size)


    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        file_path = self.data[index]
        img = self.__decode_img(file_path)
        label = self.__get_label(file_path)
        if self.one_hot_encoding:
            label = torch.nn.functional.one_hot(
                torch.tensor(self.label_mapping[label]), num_classes=len(self.labels)
            ).to(torch.float32)
        else:
            label = torch.tensor(self.label_mapping[label], dtype=torch.long)
        return {'data': img, 'print_object': str(index)}, label

    def get_set(self, split="train"):
        """Dual of the TF ``get_set``: returns a ready-to-iterate DataLoader."""
        indices = self.data_splitted[split.lower()]
        if indices is None:
            return None
        subset = Subset(self, indices)
        batch_size = self.batch_size if self.batch_size != 0 else 1
        generator = torch.Generator()
        if self.seed is not None:
            generator.manual_seed(int(self.seed))
        return DataLoader(subset, batch_size=batch_size, shuffle=True, generator=generator)


    def __get_class_names(self):
        return list({os.path.basename(p).split(self.__SEPARATOR)[self.__LABEL_ID]
                     for p in self.data})

    def __get_label(self, file_path):
        return os.path.basename(file_path).split(self.__SEPARATOR)[self.__LABEL_ID]

    def __decode_img(self, file_path):
        try:
            img = Image.open(file_path)
        except UnidentifiedImageError:
            raise NotCorrectImageFormatException(file_path)
        img = img.convert(self.__format) if self.__format else img
        t = transforms.functional.pil_to_tensor(img).to(torch.float32)
        if self.resize:
            img_height, img_width = self.dims
            if img_height is not None:
                t = transforms.functional.resize(t, [img_height, img_width], antialias=True)
        if self.normalize:
            mean, std = self.norms
            if mean is not None:
                t = t * torch.tensor(mean, dtype=torch.float32).reshape(-1, 1, 1)
        return t

    def __set_normalization_parameters(self, param):
        normalization_values = \
            {
                "RGB":
                    {
                        "mean": [float(1 / 255), float(1 / 255), float(1 / 255)],
                        "std": [0.229, 0.224, 0.225]
                    },
                "L":
                    {
                        "mean": [float(1 / 255)],
                        "std": [0.5]
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
            return [float(1 / 255), float(1 / 255), float(1 / 255)]

    def __set_resize_parameters(self, param):
        if param == "height":
            return 128
        if param == "width":
            return 128


    def print_item(self, id):
        return Image.open(self.__identifier[int(id)])

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


class Lambda_(transforms.Lambda):
    def __init__(self, func, name):
        super(Lambda_, self).__init__(func)
        self.name = name

    def __repr__(self) -> str:
        return self.name
