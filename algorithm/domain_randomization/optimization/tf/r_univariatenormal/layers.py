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
    def __init__(self, seed=None, factor=0.9, **kwargs):
        # usual behaviour (not mandatory): mean=-{0}, variance={sigma_delta} (-> delta sampled from N(0, sigma_delta),
        # with delta value to be applyed to brightness)
        super().__init__(**kwargs)
        self.seed = seed
        self.factor = factor


    def call(self, x, mean=0.0, variance=0.15, training=None):
        if (training) and (tf.random.uniform([]) <= self.factor):
            if self.seed is not None:
                delta = tf.random.normal(shape=[], mean=mean, stddev=variance, seed=self.seed)
                delta = tf.clip_by_value(delta, -1, 1)
            else:
                delta = tf.random.normal(shape=[], mean=mean, stddev=variance)
                delta = tf.clip_by_value(delta, -1, 1)
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


    def call(self, x, mean=1.25, variance=1, training=None):
        if (training) and (tf.random.uniform([]) <= self.factor):
            if self.seed is not None:
                contrast_factor = tf.random.normal(shape=[], mean=mean, stddev=variance, seed=self.seed)
            else:
                contrast_factor = tf.random.normal(shape=[], mean=mean, stddev=variance)
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


    def call(self, x, mean=0.5, variance=0.1, training=None):
        if (training) and (tf.random.uniform([]) <= self.factor):
            prob = -1
            if self.seed is not None:
                prob = tf.random.normal(shape=[], mean=mean, stddev=variance, seed=self.seed)
            else:
                prob = tf.random.normal(shape=[], mean=mean, stddev=variance)
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


    def call(self, x, mean=0.5, variance=0.1, training=None):
        if (training) and (tf.random.uniform([]) <= self.factor):
            prob = -1
            if self.seed is not None:
                prob = tf.random.normal(shape=[], mean=mean, stddev=variance, seed=self.seed)
            else:
                prob = tf.random.normal(shape=[], mean=mean, stddev=variance)
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
        # usual behaviour (not mandatory): mean=-{0}, variance={sigma_delta} (-> delta sampled from N(0, sigma_delta),
        # with delta value to be applyed to hue)
        super().__init__(**kwargs)
        self.seed = seed
        self.factor = factor


    def call(self, x, mean=0.0, variance=0.15, training=None):
        if (training) and (tf.random.uniform([]) <= self.factor):
            if self.seed is not None:
                delta = tf.random.normal(shape=[], mean=mean, stddev=variance, seed=self.seed)
                delta = tf.clip_by_value(delta, -1, 1)
            else:
                delta = tf.random.normal(shape=[], mean=mean, stddev=variance)
                delta = tf.clip_by_value(delta, -1, 1)
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


    def call(self, x, mean=60, variance=25, training=None):
        if (training) and (tf.random.uniform([]) <= self.factor):
            if self.seed is not None:
                jpeg_quality = tf.random.normal(shape=[], mean=mean, stddev=variance, seed=self.seed)
                jpeg_quality = tf.clip_by_value(jpeg_quality, 0, 100)
            else:
                jpeg_quality = tf.random.normal(shape=[], mean=mean, stddev=variance)
                jpeg_quality = tf.clip_by_value(jpeg_quality, 0, 100)
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


    def call(self, x, mean=1.25, variance=1.125, training=None):
        if (training) and (tf.random.uniform([]) <= self.factor):
            if self.seed is not None:
                saturation_factor = tf.random.normal(shape=[], mean=mean, stddev=variance, seed=self.seed)
                saturation_factor = tf.clip_by_value(saturation_factor, 0, float('inf'))
            else:
                saturation_factor = tf.random.normal(shape=[], mean=mean, stddev=variance)
                saturation_factor = tf.clip_by_value(saturation_factor, 0, float('inf'))
            return tf.image.adjust_saturation(x, saturation_factor)
        else:
            return x

    def get_config(self):
        config = super().get_config()
        config.update({'factor': self.factor})
        return config
