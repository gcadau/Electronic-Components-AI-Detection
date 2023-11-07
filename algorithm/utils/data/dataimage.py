import os
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image, UnidentifiedImageError


class NotFoundDirectoryException(Exception):
    def __init__(self, path):
        self.path = path
        self.message = f"Invalid path. {self.path} does not exist."
        super().__init__(self.message)


class NotCorrectSplitException(Exception):
    def __init__(self, split):
        self.split = split
        self.message = f"Invalid category. {self.split} is not a valid option. You can only choose between train and test."
        super().__init__(self.message)


class NotCorrectNormalizationException(Exception):
    def __init__(self):
        self.message = f"Mean and/or standard deviation has to be specified."
        super().__init__(self.message)


class NotCorrectImageFormatException(Exception):
    def __init__(self, file):
        self.message = f"{file} is not a valid image file."
        super().__init__(self.message)


class DataImage(Dataset):
    def __init__(self, data_path="./data", split="train", transform=None, normalize=False, mean=None, std=None,
                 name=None, format=None):

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
            raise NotCorrectSplitException(self.split)

        self.labels = None

        if transform is not None:
            self.transform = transforms.Compose([
                *transform.transforms,
                transforms.ToTensor()
            ])
        else:
            self.transform = transforms.Compose([
                transforms.PILToTensor()
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
            lbl = None

        if self.transform is not None:
            img = self.transform(img)

        return img, lbl

    def __repr__(self):
        repr = ""
        dataset = self.dataset_name if self.dataset_name is not None else self.data_path
        number = f"Number of points: {self.__len__()}"
        loc = f"Root location: {self.data_path}"
        split = f"Split: {self.split}"
        trans = f"Transform used: {str(self.transform)}"
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