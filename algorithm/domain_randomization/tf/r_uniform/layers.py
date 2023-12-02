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
    def __init__(self, lower=-0.2, upper=0.2, seed=None, factor=0.9, **kwargs):
        # usual behaviour (not mandatory): lower=-{max_delta}, upper={max_delta} (-> delta sampled from [-delta, delta],
        # with delta value to be applyed to brightness)
        super().__init__(**kwargs)
        self.lower = lower
        self.upper = upper
        self.seed = seed
        self.factor = factor


    def call(self, x, training=None):
        if (training) and (tf.random.uniform([]) <= self.factor):
            if self.seed is not None:
                delta = tf.random.uniform(shape=[], minval=self.lower, maxval=self.upper, seed=self.seed)
            else:
                delta = tf.random.uniform(shape=[], minval=self.lower, maxval=self.upper)
            return tf.image.adjust_brightness(x, delta)
        else:
            return x

    def get_config(self):
        config = super().get_config()
        config.update({'lower and upper': (self.lower, self.upper), 'factor': self.factor})
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
        config.update({'lower and upper': (self.lower, self.upper), 'factor': self.factor})
        return config


# possibile to define class RandomCrop(keras.layers.Layer), not so useful.


class RandomHorizontallyFlip(keras.layers.Layer):
    def __init__(self, lower=0, upper=1, seed=None, factor=0.9, **kwargs):
        # usual behaviour (not mandatory): lower=0, upper=1 -> same probability to get
        # a value greater/lower than 0.5 (-> threshold between flip/don't flip)
        super().__init__(**kwargs)
        self.lower = lower
        self.upper = upper
        self.seed = seed
        self.factor = factor


    def call(self, x, training=None):
        if (training) and (tf.random.uniform([]) <= self.factor):
            prob = -1
            if self.seed is not None:
                prob = tf.random.uniform(shape=[], minval=self.lower, maxval=self.upper, seed=self.seed)
            else:
                prob = tf.random.uniform(shape=[], minval=self.lower, maxval=self.upper)
            if prob < 0.5:
                return tf.image.flip_left_right(x)
            else:
                return x
        else:
            return x

    def get_config(self):
        config = super().get_config()
        config.update({'lower and upper': (self.lower, self.upper), 'factor': self.factor})
        return config


class RandomVerticallyFlip(keras.layers.Layer):
    def __init__(self, lower=0, upper=1, seed=None, factor=0.9, **kwargs):
        # usual behaviour (not mandatory): lower=0, upper=1 -> same probability to get
        # a value greater/lower than 0.5 (-> threshold between flip/don't flip)
        super().__init__(**kwargs)
        self.lower = lower
        self.upper = upper
        self.seed = seed
        self.factor = factor


    def call(self, x, training=None):
        if (training) and (tf.random.uniform([]) <= self.factor):
            prob = -1
            if self.seed is not None:
                prob = tf.random.uniform(shape=[], minval=self.lower, maxval=self.upper, seed=self.seed)
            else:
                prob = tf.random.uniform(shape=[], minval=self.lower, maxval=self.upper)
            if prob < 0.5:
                return tf.image.flip_up_down(x)
            else:
                return x
        else:
            return x

    def get_config(self):
        config = super().get_config()
        config.update({'lower and upper': (self.lower, self.upper), 'factor': self.factor})
        return config


class RandomHue(keras.layers.Layer):
    def __init__(self, lower=-0.2, upper=0.2, seed=None, factor=0.9, **kwargs):
        # usual behaviour (not mandatory): lower=-{max_delta}, upper={max_delta} (-> delta sampled from [-delta, delta],
        # with delta value to be applyed to hue)
        super().__init__(**kwargs)
        self.lower = lower
        self.upper = upper
        self.seed = seed
        self.factor = factor


    def call(self, x, training=None):
        if (training) and (tf.random.uniform([]) <= self.factor):
            if self.seed is not None:
                delta = tf.random.uniform(shape=[], minval=self.lower, maxval=self.upper, seed=self.seed)
            else:
                delta = tf.random.uniform(shape=[], minval=self.lower, maxval=self.upper)
            return tf.image.adjust_hue(x, delta)
        else:
            return x

    def get_config(self):
        config = super().get_config()
        config.update({'lower and upper': (self.lower, self.upper), 'factor': self.factor})
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
        config.update({'lower and upper': (self.lower, self.upper), 'factor': self.factor})
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
        config.update({'lower and upper': (self.lower, self.upper), 'factor': self.factor})
        return config
