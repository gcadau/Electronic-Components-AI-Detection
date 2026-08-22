import torch

from algorithm.domain_randomization.pt import functional as Fdr
from algorithm.domain_randomization.pt.functional import DRLayer, NoneTransformation
from algorithm.domain_randomization.pt.r_multivariatenormal.layers import (
    Brightness, Contrast, HorizontallyFlip, VerticallyFlip, Hue, JpegQuality, Saturation
)


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


class RandomParameters(DRLayer):

    TRANSFORMATIONS = (Brightness, Contrast, HorizontallyFlip, VerticallyFlip,
                       Hue, JpegQuality, Saturation)

    def __init__(self, seed=None, factors=None, **kwargs):
        super().__init__(seed=seed)

        if factors is None:
            factors = [0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9]

        self.factor = factors
        self.factors = factors
        self.randoms = []

    def forward(self, x, mean=None, variance=None, values=None, training=None, rand=True):
        if not self._training(training):
            return x

        batched = x.ndim == 4
        n = self._n(x)

        if rand:
            parameters = self.rng.multivariate_normal(mean, variance, size=n)
        else:
            if values is None:
                raise ValueError("`values` must be provided when `rand=False`.")
            parameters = values

        self.randoms = []
        for k, transformation in enumerate(self.TRANSFORMATIONS):
            per_sample = []
            for i in range(n):
                if torch.rand((), generator=self.generator).item() <= self.factor[k]:
                    per_sample.append(transformation(par=parameters[i][k]))
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
        return {"seed": self.seed, "factors": self.factors}
