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
    def __init__(self, seed=None, factor=0.9, **kwargs):
        # usual behaviour (not mandatory): lower=-{max_delta}, upper={max_delta} (-> delta sampled from
        # Tr[-delta, 0, delta], with delta value to be applyed to brightness)
        super().__init__(**kwargs)
        self.seed = seed
        self.factor = factor


    def call(self, x, lower=-0.2, mode=0, upper=0.2, training=None):
        if mode is None:
            mode = (upper-lower)/2
        if (training) and (tf.random.uniform([]) <= self.factor):
            if self.seed is not None:
                tf.random.set_seed(self.seed)
                rng = np.random.default_rng(seed=self.seed)
                delta = rng.triangular(lower, mode, upper)
            else:
                delta = np.random.triangular(lower, mode, upper)
            return tf.image.adjust_brightness(x, delta)
        else:
            return x

    def get_config(self):
        config = super().get_config()
        config.update({'factor': self.factor})
        return config


class RandomContrast(keras.layers.Layer):
    def __init__(self, seed=None, factor=0.9, **kwargs):
        super().__init__(**kwargs)
        self.seed = seed
        self.factor = factor


    def call(self, x, lower=0, upper=2.5, mode=None, training=None):
        if mode is None:
            mode = (upper-lower)/2
        if (training) and (tf.random.uniform([]) <= self.factor):
            if self.seed is not None:
                tf.random.set_seed(self.seed)
                rng = np.random.default_rng(seed=self.seed)
                contrast_factor = rng.triangular(lower, mode, upper)
            else:
                contrast_factor = np.random.triangular(lower, mode, upper)
            return tf.image.adjust_contrast(x, contrast_factor)
        else:
            return x

    def get_config(self):
        config = super().get_config()
        config.update({'factor': self.factor})
        return config


# possibile to define class RandomCrop(keras.layers.Layer), not so useful.


class RandomHorizontallyFlip(keras.layers.Layer):
    def __init__(self, seed=None, factor=0.9, **kwargs):
        super().__init__(**kwargs)
        self.seed = seed
        self.factor = factor


    def call(self, x, lower=0, upper=1, mode=0.5, training=None):
        if mode is None:
            mode = (upper-lower)/2
        if (training) and (tf.random.uniform([]) <= self.factor):
            prob = -1
            if self.seed is not None:
                tf.random.set_seed(self.seed)
                rng = np.random.default_rng(seed=self.seed)
                prob = rng.triangular(lower, mode, upper)
            else:
                prob = np.random.triangular(lower, mode, upper)
            if prob < 0.5:
                return tf.image.flip_left_right(x)
            else:
                return x
        else:
            return x

    def get_config(self):
        config = super().get_config()
        config.update({'factor': self.factor})
        return config


class RandomVerticallyFlip(keras.layers.Layer):
    def __init__(self, seed=None, factor=0.9, **kwargs):
        super().__init__(**kwargs)
        self.seed = seed
        self.factor = factor


    def call(self, x, lower=0, upper=1, mode=0.5, training=None):
        if mode is None:
            mode = (upper-lower)/2
        if (training) and (tf.random.uniform([]) <= self.factor):
            prob = -1
            if self.seed is not None:
                tf.random.set_seed(self.seed)
                rng = np.random.default_rng(seed=self.seed)
                prob = rng.triangular(lower, mode, upper)
            else:
                prob = np.random.triangular(lower, mode, upper)
            if prob < 0.5:
                return tf.image.flip_up_down(x)
            else:
                return x
        else:
            return x

    def get_config(self):
        config = super().get_config()
        config.update({'factor': self.factor})
        return config


class RandomHue(keras.layers.Layer):
    def __init__(self, seed=None, factor=0.9, **kwargs):
        # usual behaviour (not mandatory): lower=-{max_delta}, upper={max_delta} (-> delta sampled from
        # Tr[-delta, 0, delta], with delta value to be applyed to hue)
        super().__init__(**kwargs)
        self.seed = seed
        self.factor = factor


    def call(self, x, lower=-0.2, mode=0, upper=0.2, training=None):
        if mode is None:
            mode = (upper-lower)/2
        if (training) and (tf.random.uniform([]) <= self.factor):
            if self.seed is not None:
                tf.random.set_seed(self.seed)
                rng = np.random.default_rng(seed=self.seed)
                delta = rng.triangular(lower, mode, upper)
            else:
                delta = np.random.triangular(lower, mode, upper)
            return tf.image.adjust_hue(x, delta)
        else:
            return x

    def get_config(self):
        config = super().get_config()
        config.update({'factor': self.factor})
        return config


class RandomJpegQuality(keras.layers.Layer):
    def __init__(self, seed=None, factor=0.9, **kwargs):
        super().__init__(**kwargs)
        self.seed = seed
        self.factor = factor


    def call(self, x, lower=20, upper=100, mode=None, training=None):
        if mode is None:
            mode = (upper-lower)/2
        if (training) and (tf.random.uniform([]) <= self.factor):
            if self.seed is not None:
                tf.random.set_seed(self.seed)
                rng = np.random.default_rng(seed=self.seed)
                jpeg_quality = rng.triangular(lower, mode, upper)
            else:
                jpeg_quality = np.random.triangular(lower, mode, upper)
            ims = []
            for i in range(x.shape[0]):
                im = x[i,:,:,:]
                im = tf.image.adjust_jpeg_quality(im, jpeg_quality)
                ims.append(im)
            x = tf.stack(ims)
            return x
        else:
            return x

    def get_config(self):
        config = super().get_config()
        config.update({'factor': self.factor})
        return config


class RandomSaturation(keras.layers.Layer):
    def __init__(self, seed=None, factor=0.9, **kwargs):
        super().__init__(**kwargs)
        self.seed = seed
        self.factor = factor


    def call(self, x, lower=0, upper=2, mode=None, training=None):
        if mode is None:
            mode = (upper-lower)/2
        if (training) and (tf.random.uniform([]) <= self.factor):
            if self.seed is not None:
                tf.random.set_seed(self.seed)
                rng = np.random.default_rng(seed=self.seed)
                saturation_factor = rng.triangular(lower, mode, upper)
            else:
                saturation_factor = np.random.triangular(lower, mode, upper)
            return tf.image.adjust_saturation(x, saturation_factor)
        else:
            return x

    def get_config(self):
        config = super().get_config()
        config.update({'factor': self.factor})
        return config
