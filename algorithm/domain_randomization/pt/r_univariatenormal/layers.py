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
    def __init__(self, mean=0.0, variance=0.15, seed=None, factor=0.9, **kwargs):
        # usual behaviour (not mandatory): mean={0}, variance={sigma_delta} (-> delta sampled from
        # N(0, sigma_delta), with delta value to be applyed to brightness)
        super().__init__(seed=seed, factor=factor)
        self.mean = mean
        self.variance = variance

    def forward(self, x, training=None):
        if not self._training(training):
            return x
        n = self._n(x)
        delta = self._normal(n, self.mean, self.variance).clamp(-1, 1)
        delta = self._gate(delta, self._coin(n), identity=0.0)
        return Fdr.adjust_brightness(x, delta)

    def get_config(self):
        return {"mean": self.mean, "variance": self.variance, "seed": self.seed, "factor": self.factor}


class RandomContrast(DRLayer):
    def __init__(self, mean=1.25, variance=1, seed=None, factor=0.9, **kwargs):
        super().__init__(seed=seed, factor=factor)
        self.mean = mean
        self.variance = variance

    def forward(self, x, training=None):
        if not self._training(training):
            return x
        n = self._n(x)
        contrast_factor = self._normal(n, self.mean, self.variance)
        contrast_factor = self._gate(contrast_factor, self._coin(n), identity=1.0)
        return Fdr.adjust_contrast(x, contrast_factor)

    def get_config(self):
        return {"mean": self.mean, "variance": self.variance, "seed": self.seed, "factor": self.factor}


# possible to define class RandomCrop(DRLayer), not so useful.


class RandomHorizontallyFlip(DRLayer):
    def __init__(self, mean=0.5, variance=0.1, seed=None, factor=0.9, **kwargs):
        super().__init__(seed=seed, factor=factor)
        self.mean = mean
        self.variance = variance

    def forward(self, x, training=None):
        if not self._training(training):
            return x
        n = self._n(x)
        prob = self._normal(n, self.mean, self.variance)
        mask = self._coin(n) & (prob < 0.5)
        return self._where(mask, Fdr.flip_left_right(x), x)

    def get_config(self):
        return {"mean": self.mean, "variance": self.variance, "seed": self.seed, "factor": self.factor}


class RandomVerticallyFlip(DRLayer):
    def __init__(self, mean=0.5, variance=0.1, seed=None, factor=0.9, **kwargs):
        super().__init__(seed=seed, factor=factor)
        self.mean = mean
        self.variance = variance

    def forward(self, x, training=None):
        if not self._training(training):
            return x
        n = self._n(x)
        prob = self._normal(n, self.mean, self.variance)
        mask = self._coin(n) & (prob < 0.5)
        return self._where(mask, Fdr.flip_up_down(x), x)

    def get_config(self):
        return {"mean": self.mean, "variance": self.variance, "seed": self.seed, "factor": self.factor}


class RandomHue(DRLayer):
    def __init__(self, mean=0.0, variance=0.15, seed=None, factor=0.9, **kwargs):
        super().__init__(seed=seed, factor=factor)
        self.mean = mean
        self.variance = variance

    def forward(self, x, training=None):
        if not self._training(training):
            return x
        n = self._n(x)
        delta = self._normal(n, self.mean, self.variance).clamp(-1, 1)
        delta = self._gate(delta, self._coin(n), identity=0.0)
        return Fdr.adjust_hue(x, delta)

    def get_config(self):
        return {"mean": self.mean, "variance": self.variance, "seed": self.seed, "factor": self.factor}


class RandomJpegQuality(DRLayer):
    def __init__(self, mean=60, variance=25, seed=None, factor=0.9, **kwargs):
        super().__init__(seed=seed, factor=factor)
        self.mean = mean
        self.variance = variance

    def forward(self, x, training=None):
        if not self._training(training):
            return x
        n = self._n(x)
        jpeg_quality = self._normal(n, self.mean, self.variance).clamp(0, 100)
        return self._where(self._coin(n), Fdr.adjust_jpeg_quality(x, jpeg_quality), x)

    def get_config(self):
        return {"mean": self.mean, "variance": self.variance, "seed": self.seed, "factor": self.factor}


class RandomSaturation(DRLayer):
    def __init__(self, mean=1.25, variance=1.125, seed=None, factor=0.9, **kwargs):
        super().__init__(seed=seed, factor=factor)
        self.mean = mean
        self.variance = variance

    def forward(self, x, training=None):
        if not self._training(training):
            return x
        n = self._n(x)
        saturation_factor = self._normal(n, self.mean, self.variance).clamp(min=0)
        saturation_factor = self._gate(saturation_factor, self._coin(n), identity=1.0)
        return Fdr.adjust_saturation(x, saturation_factor)

    def get_config(self):
        return {"mean": self.mean, "variance": self.variance, "seed": self.seed, "factor": self.factor}
