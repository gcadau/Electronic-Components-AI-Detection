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
    def __init__(self, max_delta, **kwargs):
        super().__init__(**kwargs)
        self.max_delta = max_delta


    def call(self, x, training=None):
        if training:
            return tf.image.random_brightness(x, max_delta=self.max_delta, seed=self.seed)
        else:
            return x

    def get_config(self):
        config = super().get_config()
        config.update({'max_delta': self.max_delta})
        return config