import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image, UnidentifiedImageError
from algorithm.utils.data.exceptions import *


class DataImage(Dataset):
    def __init__(self, data_path="./data", split="train", transform=None, normalize=False, mean=None, std=None,
                 resize=False, height=None, width=None, name=None, format=None):

        self.__format = format
        self.dataset_name = name
        self.data_path = data_path

        self.data = []
        if os.path.exists(self.data_path):
            file_list = [os.path.join(self.data_path, f) for f in os.listdir(self.data_path)]
            for f in file_list:
                try:
                    image = Image.open(f)
                    self.data.append(f)
                    if self.__format is None:
                        self.__format = image.mode
                except UnidentifiedImageError:
                    raise NotCorrectImageFormatException(f)
        else:
            raise NotFoundDirectoryException(self.data_path)

        self.split = split.lower()
        if self.split not in ["train", "test"]:
            raise NotCorrectSplitException(self.split, "category")

        self.labels = None

        convertion = Lambda_(lambda x: x.to(torch.float32), 'ConvertToFloat32')
        if transform is not None:
            self.transform = transforms.Compose([
                *transform.transforms,
                transforms.PILToTensor(),
                convertion
            ])
        else:
            self.transform = transforms.Compose([
                transforms.PILToTensor(),
                convertion
            ])

        if resize:
            if height is None or width is None:
                raise NotCorrectResizeException()
            if height == 'auto':
                height = self.__set_resize_parameters("height")
            if width == 'auto':
                width = self.__set_resize_parameters("width")
            self.transform = transforms.Compose([
                *self.transform.transforms,
                transforms.Resize((height, width))
            ])

        if normalize:
            if mean is None or std is None:
                raise NotCorrectNormalizationException()
            if mean == 'auto':
                mean = self.__set_normalization_parameters("mean")
            if std == 'auto':
                std = self.__set_normalization_parameters("std")
            self.transform = transforms.Compose([
                *self.transform.transforms,
                transforms.Normalize(mean=mean, std=std)
            ])


    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # GESTIRE LABELS E SPLIT.
        file = self.data[index]
        img = Image.open(file)
        try:
            lbl = self.labels[index]
        except TypeError:
            lbl = "-No label available-"

        if self.transform is not None:
            img = self.transform(img)

        return img, lbl

    def __repr__(self):
        repr = ""
        dataset = self.dataset_name if self.dataset_name is not None else self.data_path
        number = f"Number of points: {self.__len__()}"
        loc = f"Root location: {self.data_path}"
        split = f"Split: {self.split}"
        trans = f"Transform used:"
        for t in str(self.transform).split("\n"):
            trans += "\t" + t + "\n"
        for s in (dataset, number, loc, split, trans):
            if s != dataset:
                s = '\t' + s
            repr += s + "\n"
        return repr

    def print_item(self, index):  # TO BE IMPLEMENTED.
        pass
        # functions to show an image

        # def imshow(img):
        # img = img / 2 + 0.5  # unnormalize
        # npimg = img.numpy()
        # plt.imshow(np.transpose(npimg, (1, 2, 0)))
        # plt.show()

        # show images
        # imshow(torchvision.utils.make_grid(img_tensor))

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


class Lambda_(transforms.Lambda):
    def __init__(self, func, name):
        super(Lambda_, self).__init__(func)
        self.name = name

    def __repr__(self) -> str:
        return self.name
