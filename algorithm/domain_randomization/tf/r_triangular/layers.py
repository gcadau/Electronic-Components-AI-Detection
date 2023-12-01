from tensorflow import keras
import tensorflow as tf
import numpy as np


class RandomInvert(keras.layers.Layer):

    def __init__(self, factor=0.5, seed=None, **kwargs):
        super().__init__(**kwargs)
        self.factor = factor
        self.seed = seed


    @staticmethod
    def __random_invert_img(x, p=0.5):
        if tf.random.uniform([]) < p:
            x_tf = (255 - x)
        else:
            x_tf = x
        return x_tf

    def call(self, x, training=None):
        if training:
            return self.__random_invert_img(x, self.factor)
        else:
            return x

    def get_config(self):
        config = super().get_config()
        config.update({'factor': self.factor})
        return config


class RandomBrightness(keras.layers.Layer):
    def __init__(self, max_delta=0.2, mode=None, seed=None, factor=0.9, **kwargs):
        super().__init__(**kwargs)
        self.max_delta = max_delta
        self.seed = seed
        self.factor = factor
        self.mode = mode
        if self.mode is None:
            self.mode = 0


    def call(self, x, training=None):
        if (training) and (tf.random.uniform([]) <= self.factor):
            if self.seed is not None:
                tf.random.set_seed(self.seed)
                rng = np.random.default_rng(seed=self.seed)
                delta = rng.triangular(-self.max_delta, self.mode, self.max_delta)
            else:
                delta = np.random.triangular(-self.max_delta, self.mode, self.max_delta)
            return tf.image.adjust_brightness(x, delta)
        else:
            return x

    def get_config(self):
        config = super().get_config()
        config.update({'max_delta': self.max_delta, 'mode': self.mode, 'factor': self.factor})
        return config


class RandomContrast(keras.layers.Layer):
    def __init__(self, lower=0, upper=2.5, mode=None, seed=None, factor=0.9, **kwargs):
        super().__init__(**kwargs)
        self.lower = lower
        self.upper = upper
        self.seed = seed
        self.factor = factor
        self.mode = mode
        if self.mode is None:
            self.mode = (self.upper-self.lower)/2


    def call(self, x, training=None):
        if (training) and (tf.random.uniform([]) <= self.factor):
            if self.seed is not None:
                tf.random.set_seed(self.seed)
                rng = np.random.default_rng(seed=self.seed)
                contrast_factor = rng.triangular(self.lower, self.mode, self.upper)
            else:
                contrast_factor = np.random.triangular(self.lower, self.mode, self.upper)
            return tf.image.adjust_contrast(x, contrast_factor)
        else:
            return x

    def get_config(self):
        config = super().get_config()
        config.update({'low and upper': (self.lower, self.upper),  'mode': self.mode, 'factor': self.factor})
        return config


# possibile to define class RandomCrop(keras.layers.Layer), not so useful.


class RandomHorizontallyFlip(keras.layers.Layer):
    def __init__(self, seed=None, factor=0.9, **kwargs):
        super().__init__(**kwargs)
        self.seed = seed
        self.factor = factor
        self.mode = 0.5


    def call(self, x, training=None):
        if (training) and (tf.random.uniform([]) <= self.factor):
            prob = -1
            if self.seed is not None:
                tf.random.set_seed(self.seed)
                rng = np.random.default_rng(seed=self.seed)
                prob = rng.triangular(0, self.mode, 1)
            else:
                prob = np.random.triangular(0, self.mode, 1)
            if prob < 0.5:
                return tf.image.flip_left_right(x)
            else:
                return x
        else:
            return x

    def get_config(self):
        config = super().get_config()
        config.update({'factor': self.factor, 'mode': self.mode})
        return config


class RandomVerticallyFlip(keras.layers.Layer):
    def __init__(self, seed=None, factor=0.9, **kwargs):
        super().__init__(**kwargs)
        self.seed = seed
        self.factor = factor
        self.mode = 0.5


    def call(self, x, training=None):
        if (training) and (tf.random.uniform([]) <= self.factor):
            prob = -1
            if self.seed is not None:
                tf.random.set_seed(self.seed)
                rng = np.random.default_rng(seed=self.seed)
                prob = rng.triangular(0, self.mode, 1)
            else:
                prob = np.random.triangular(0, self.mode, 1)
            if prob < 0.5:
                return tf.image.flip_up_down(x)
            else:
                return x
        else:
            return x

    def get_config(self):
        config = super().get_config()
        config.update({'factor': self.factor, 'mode': self.mode})
        return config


class RandomHue(keras.layers.Layer):
    def __init__(self, max_delta=0.2, mode=None, seed=None, factor=0.9, **kwargs):
        super().__init__(**kwargs)
        self.max_delta = max_delta
        self.seed = seed
        self.factor = factor
        self.mode = mode
        if self.mode is None:
            self.mode = 0


    def call(self, x, training=None):
        if (training) and (tf.random.uniform([]) <= self.factor):
            if self.seed is not None:
                tf.random.set_seed(self.seed)
                rng = np.random.default_rng(seed=self.seed)
                delta = rng.triangular(-self.max_delta, self.mode, self.max_delta)
            else:
                delta = np.random.triangular(-self.max_delta, self.mode, self.max_delta)
            return tf.image.adjust_hue(x, delta)
        else:
            return x

    def get_config(self):
        config = super().get_config()
        config.update({'max_delta': self.max_delta, 'factor': self.factor, 'mode': self.mode})
        return config


class RandomJpegQuality(keras.layers.Layer):
    def __init__(self, lower=20, upper=100, mode=None, seed=None, factor=0.9, **kwargs):
        super().__init__(**kwargs)
        self.lower = lower
        self.upper = upper
        self.seed = seed
        self.factor = factor
        self.mode = mode
        if self.mode is None:
            self.mode = (self.upper-self.lower)/2


    def call(self, x, training=None):
        if (training) and (tf.random.uniform([]) <= self.factor):
            if self.seed is not None:
                tf.random.set_seed(self.seed)
                rng = np.random.default_rng(seed=self.seed)
                jpeg_quality = rng.triangular(self.lower, self.mode, self.upper)
            else:
                jpeg_quality = np.random.triangular(self.lower, self.mode, self.upper)
            return tf.image.adjust_jpeg_quality(x, jpeg_quality)
        else:
            return x

    def get_config(self):
        config = super().get_config()
        config.update({'low and upper': (self.lower, self.upper), 'factor': self.factor, 'mode': self.mode})
        return config


class RandomSaturation(keras.layers.Layer):
    def __init__(self, lower=0, upper=2, mode=None, seed=None, factor=0.9, **kwargs):
        super().__init__(**kwargs)
        self.lower = lower
        self.upper = upper
        self.seed = seed
        self.factor = factor
        self.mode = mode
        if self.mode is None:
            self.mode = (self.upper-self.lower)/2


    def call(self, x, training=None):
        if (training) and (tf.random.uniform([]) <= self.factor):
            if self.seed is not None:
                tf.random.set_seed(self.seed)
                rng = np.random.default_rng(seed=self.seed)
                saturation_factor = rng.triangular(self.lower, self.mode, self.upper)
            else:
                saturation_factor = np.random.triangular(self.lower, self.mode, self.upper)
            return tf.image.adjust_saturation(x, saturation_factor)
        else:
            return x

    def get_config(self):
        config = super().get_config()
        config.update({'low and upper': (self.lower, self.upper), 'factor': self.factor, 'mode': self.mode})
        return config
