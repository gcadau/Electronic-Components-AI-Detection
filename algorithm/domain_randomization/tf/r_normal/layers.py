from tensorflow import keras
import tensorflow as tf


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
    def __init__(self, max_delta=0.2, seed=None, factor=0.9, **kwargs):
        super().__init__(**kwargs)
        self.max_delta = max_delta
        self.seed = seed
        self.factor = factor


    def call(self, x, training=None):
        if (training) and (tf.random.uniform([]) <= self.factor):
            if self.seed is not None:
                delta = tf.random.uniform(shape=[], minval=-self.max_delta, maxval=self.max_delta, seed=self.seed)
            else:
                delta = tf.random.uniform(shape=[], minval=-self.max_delta, maxval=self.max_delta)
            return tf.image.adjust_brightness(x, delta)
        else:
            return x

    def get_config(self):
        config = super().get_config()
        config.update({'max_delta': self.max_delta, 'factor': self.factor})
        return config


class RandomContrast(keras.layers.Layer):
    def __init__(self, lower=0, upper=2.5, seed=None, factor=0.9, **kwargs):
        super().__init__(**kwargs)
        self.lower = lower
        self.upper = upper
        self.seed = seed
        self.factor = factor


    def call(self, x, training=None):
        if (training) and (tf.random.uniform([]) <= self.factor):
            if self.seed is not None:
                contrast_factor = tf.random.uniform(shape=[], minval=self.lower, maxval=self.upper, seed=self.seed)
            else:
                contrast_factor = tf.random.uniform(shape=[], minval=self.lower, maxval=self.upper)
            return tf.image.adjust_contrast(x, contrast_factor)
        else:
            return x

    def get_config(self):
        config = super().get_config()
        config.update({'low and upper': (self.lower, self.upper), 'factor': self.factor})
        return config


# possibile to define class RandomCrop(keras.layers.Layer), not so useful.


class RandomHorizontallyFlip(keras.layers.Layer):
    def __init__(self, seed=None, factor=0.9, **kwargs):
        super().__init__(**kwargs)
        self.seed = seed
        self.factor = factor


    def call(self, x, training=None):
        if (training) and (tf.random.uniform([]) <= self.factor):
            prob = -1
            if self.seed is not None:
                prob = tf.random.uniform(shape=[], minval=0, maxval=1, seed=self.seed)
            else:
                prob = tf.random.uniform(shape=[], minval=0, maxval=1)
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


    def call(self, x, training=None):
        if (training) and (tf.random.uniform([]) <= self.factor):
            prob = -1
            if self.seed is not None:
                prob = tf.random.uniform(shape=[], minval=0, maxval=1, seed=self.seed)
            else:
                prob = tf.random.uniform(shape=[], minval=0, maxval=1)
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
    def __init__(self, max_delta=0.2, seed=None, factor=0.9, **kwargs):
        super().__init__(**kwargs)
        self.max_delta = max_delta
        self.seed = seed
        self.factor = factor


    def call(self, x, training=None):
        if (training) and (tf.random.uniform([]) <= self.factor):
            if self.seed is not None:
                delta = tf.random.uniform(shape=[], minval=-self.max_delta, maxval=self.max_delta, seed=self.seed)
            else:
                delta = tf.random.uniform(shape=[], minval=-self.max_delta, maxval=self.max_delta)
            return tf.image.adjust_hue(x, delta)
        else:
            return x

    def get_config(self):
        config = super().get_config()
        config.update({'max_delta': self.max_delta, 'factor': self.factor})
        return config


class RandomJpegQuality(keras.layers.Layer):
    def __init__(self, lower=20, upper=100, seed=None, factor=0.9, **kwargs):
        super().__init__(**kwargs)
        self.lower = lower
        self.upper = upper
        self.seed = seed
        self.factor = factor


    def call(self, x, training=None):
        if (training) and (tf.random.uniform([]) <= self.factor):
            if self.seed is not None:
                jpeg_quality = tf.random.uniform(shape=[], minval=self.lower, maxval=self.upper, seed=self.seed)
            else:
                jpeg_quality = tf.random.uniform(shape=[], minval=self.lower, maxval=self.upper)
            return tf.image.adjust_jpeg_quality(x, jpeg_quality)
        else:
            return x

    def get_config(self):
        config = super().get_config()
        config.update({'low and upper': (self.lower, self.upper), 'factor': self.factor})
        return config


class RandomSaturation(keras.layers.Layer):
    def __init__(self, lower=0, upper=2, seed=None, factor=0.9, **kwargs):
        super().__init__(**kwargs)
        self.lower = lower
        self.upper = upper
        self.seed = seed
        self.factor = factor


    def call(self, x, training=None):
        if (training) and (tf.random.uniform([]) <= self.factor):
            if self.seed is not None:
                saturation_factor = tf.random.uniform(shape=[], minval=self.lower, maxval=self.upper, seed=self.seed)
            else:
                saturation_factor = tf.random.uniform(shape=[], minval=self.lower, maxval=self.upper)
            return tf.image.adjust_saturation(x, saturation_factor)
        else:
            return x

    def get_config(self):
        config = super().get_config()
        config.update({'low and upper': (self.lower, self.upper), 'factor': self.factor})
        return config
