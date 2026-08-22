import numpy as np
import torch

from algorithm.domain_randomization.pt import functional as Fdr
from algorithm.domain_randomization.pt.functional import DRLayer, NoneTransformation


class RandomInvert(DRLayer):

    def __init__(self, factor=0.5, seed=None, **kwargs):
        super().__init__(seed=seed, factor=factor)

    def __random_invert_img(self, x, p=0.5):
        if torch.rand((), generator=self.generator).item() < p:
            return Fdr.invert(x)
        return x

    def forward(self, x, training=None):
        if self._training(training):
            return self.__random_invert_img(x, self.factor)
        return x

    def get_config(self):
        return {"factor": self.factor}


class Brightness(DRLayer):
    def __init__(self, par, **kwargs):
        super().__init__()
        if par < -1:
            par = -1
        if par > 1:
            par = 1
        self.delta = par

    def forward(self, x, training=None):
        if self._training(training):
            return Fdr.adjust_brightness(x, self.delta)
        return x

    def get_config(self):
        return {"par": self.delta}


class Contrast(DRLayer):
    def __init__(self, par, **kwargs):
        super().__init__()
        self.contrast_factor = par

    def forward(self, x, training=None):
        if self._training(training):
            return Fdr.adjust_contrast(x, self.contrast_factor)
        return x

    def get_config(self):
        return {"par": self.contrast_factor}


# possible to define class Crop(DRLayer), not so useful.


class HorizontallyFlip(DRLayer):
    def __init__(self, par, **kwargs):
        super().__init__()
        self.prob = par

    def forward(self, x, training=None):
        if self._training(training) and self.prob < 0.5:
            return Fdr.flip_left_right(x)
        return x

    def get_config(self):
        return {"par": self.prob}


class VerticallyFlip(DRLayer):
    def __init__(self, par, **kwargs):
        super().__init__()
        self.prob = par

    def forward(self, x, training=None):
        if self._training(training) and self.prob < 0.5:
            return Fdr.flip_up_down(x)
        return x

    def get_config(self):
        return {"par": self.prob}


class Hue(DRLayer):
    def __init__(self, par, **kwargs):
        super().__init__()
        if par < -1:
            par = -1
        if par > 1:
            par = 1
        self.delta = par

    def forward(self, x, training=None):
        if self._training(training):
            return Fdr.adjust_hue(x, self.delta)
        return x

    def get_config(self):
        return {"par": self.delta}


class JpegQuality(DRLayer):
    def __init__(self, par, **kwargs):
        super().__init__()
        if par < 0:
            par = 0
        if par > 100:
            par = 100
        self.jpeg_quality = par

    def forward(self, x, training=None):
        if self._training(training):
            return Fdr.adjust_jpeg_quality(x, self.jpeg_quality)
        return x

    def get_config(self):
        return {"par": self.jpeg_quality}


class Saturation(DRLayer):
    def __init__(self, par, **kwargs):
        super().__init__()
        if par < 0:
            par = 0
        self.saturation_factor = par

    def forward(self, x, training=None):
        if self._training(training):
            return Fdr.adjust_saturation(x, self.saturation_factor)
        return x

    def get_config(self):
        return {"par": self.saturation_factor}


class RandomParameters(DRLayer):

    TRANSFORMATIONS = (Brightness, Contrast, HorizontallyFlip, VerticallyFlip,
                       Hue, JpegQuality, Saturation)

    def __init__(self, mean_vector=None, variancecovariance_matrix=None, seed=None, factors=None, **kwargs):
        super().__init__(seed=seed)

        if factors is None:
            factors = [0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9]
        if mean_vector is None:
            mean_vector = [0, 1.25, 0.5, 0.5, 0, 60, 1.25]
        if variancecovariance_matrix is None:
            variancecovariance_matrix = [
                [0.15, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0.1, 0, 0, 0, 0],
                [0, 0, 0, 0.1, 0, 0, 0],
                [0, 0, 0, 0, 0.15, 0, 0],
                [0, 0, 0, 0, 0, 25, 0],
                [0, 0, 0, 0, 0, 0, 1.125]
            ]

        self.factor = factors
        self.mean = np.array(mean_vector)
        self.variance = np.array(variancecovariance_matrix)
        self.randoms = []

    def forward(self, x, training=None):
        if not self._training(training):
            return x

        batched = x.ndim == 4
        n = self._n(x)
        random_parameters = self.rng.multivariate_normal(self.mean, self.variance, size=n)

        self.randoms = []
        for k, transformation in enumerate(self.TRANSFORMATIONS):
            per_sample = []
            for i in range(n):
                if torch.rand((), generator=self.generator).item() <= self.factor[k]:
                    per_sample.append(transformation(par=random_parameters[i][k]))
                else:
                    per_sample.append(NoneTransformation())
            self.randoms.append(per_sample)

        for k in range(len(self.factor)):
            ran = self.randoms[k]
            if batched:
                x = torch.stack([ran[i](x[i], training=True) for i in range(n)])
            else:
                x = ran[0](x, training=True)

        return x

    def get_config(self):
        return {
            "mean_vector": self.mean.tolist(),
            "variancecovariance_matrix": self.variance.tolist(),
            "seed": self.seed,
            "factors": self.factor,
        }
