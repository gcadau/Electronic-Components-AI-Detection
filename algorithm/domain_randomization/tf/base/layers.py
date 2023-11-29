from tensorflow import keras
import tensorflow as tf


class RandomInvert(keras.layers.Layer):

    def __init__(self, factor=0.5, **kwargs):
        super().__init__(**kwargs)
        self.factor = factor


    @staticmethod
    def __random_invert_img(x, p=0.5):
        if tf.random.uniform([]) < p:
            x_tf = (255 - x)
        else:
            x_tf = x
        return x_tf

    def call(self, x):
        return self.__random_invert_img(x, self.factor)