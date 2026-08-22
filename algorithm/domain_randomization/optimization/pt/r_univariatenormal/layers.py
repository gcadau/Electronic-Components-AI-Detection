import torch

from algorithm.domain_randomization.pt import functional as Fdr
from algorithm.domain_randomization.pt.functional import DRLayer


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


class RandomBrightness(DRLayer):
    def __init__(self, seed=None, factor=0.9, **kwargs):
        super().__init__(seed=seed, factor=factor)

    def forward(self, x, training=None, mean=0.0, variance=0.15, value=None, rand=True):
        if not self._training(training):
            return x
        n = self._n(x)
        delta = self._normal(n, mean, variance) if rand else self._values(value, n)
        delta = self._gate(delta.clamp(-1, 1), self._coin(n), identity=0.0)
        return Fdr.adjust_brightness(x, delta)


class RandomContrast(DRLayer):
    def __init__(self, seed=None, factor=0.9, **kwargs):
        super().__init__(seed=seed, factor=factor)

    def forward(self, x, training=None, mean=1.25, variance=1, value=None, rand=True):
        if not self._training(training):
            return x
        n = self._n(x)
        contrast_factor = self._normal(n, mean, variance) if rand else self._values(value, n)
        contrast_factor = self._gate(contrast_factor, self._coin(n), identity=1.0)
        return Fdr.adjust_contrast(x, contrast_factor)


# possible to define class RandomCrop(DRLayer), not so useful.


class RandomHorizontallyFlip(DRLayer):
    def __init__(self, seed=None, factor=0.9, **kwargs):
        super().__init__(seed=seed, factor=factor)

    def forward(self, x, training=None, mean=0.5, variance=0.1, value=None, rand=True):
        if not self._training(training):
            return x
        n = self._n(x)
        prob = self._normal(n, mean, variance) if rand else self._values(value, n)
        mask = self._coin(n) & (prob < 0.5)
        return self._where(mask, Fdr.flip_left_right(x), x)


class RandomVerticallyFlip(DRLayer):
    def __init__(self, seed=None, factor=0.9, **kwargs):
        super().__init__(seed=seed, factor=factor)

    def forward(self, x, training=None, mean=0.5, variance=0.1, value=None, rand=True):
        if not self._training(training):
            return x
        n = self._n(x)
        prob = self._normal(n, mean, variance) if rand else self._values(value, n)
        mask = self._coin(n) & (prob < 0.5)
        return self._where(mask, Fdr.flip_up_down(x), x)


class RandomHue(DRLayer):
    def __init__(self, seed=None, factor=0.9, **kwargs):
        super().__init__(seed=seed, factor=factor)

    def forward(self, x, training=None, mean=0.0, variance=0.15, value=None, rand=True):
        if not self._training(training):
            return x
        n = self._n(x)
        delta = self._normal(n, mean, variance) if rand else self._values(value, n)
        delta = self._gate(delta.clamp(-1, 1), self._coin(n), identity=0.0)
        return Fdr.adjust_hue(x, delta)


class RandomJpegQuality(DRLayer):
    def __init__(self, seed=None, factor=0.9, **kwargs):
        super().__init__(seed=seed, factor=factor)

    def forward(self, x, training=None, mean=60, variance=25, value=None, rand=True):
        if not self._training(training):
            return x
        n = self._n(x)
        jpeg_quality = self._normal(n, mean, variance) if rand else self._values(value, n)
        jpeg_quality = jpeg_quality.clamp(0, 100)
        return self._where(self._coin(n), Fdr.adjust_jpeg_quality(x, jpeg_quality), x)


class RandomSaturation(DRLayer):
    def __init__(self, seed=None, factor=0.9, **kwargs):
        super().__init__(seed=seed, factor=factor)

    def forward(self, x, training=None, mean=1.25, variance=1.125, value=None, rand=True):
        if not self._training(training):
            return x
        n = self._n(x)
        saturation_factor = self._normal(n, mean, variance) if rand else self._values(value, n)
        saturation_factor = self._gate(saturation_factor.clamp(min=0), self._coin(n), identity=1.0)
        return Fdr.adjust_saturation(x, saturation_factor)
