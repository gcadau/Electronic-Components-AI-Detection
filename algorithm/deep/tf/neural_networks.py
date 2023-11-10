import tensorflow
from keras.src.layers import Conv2D, Flatten, Dense
from tensorflow import keras
from keras import Model


class ResNet1(keras.Model):

    def __init__(self):
        super(ResNet1, self).__init__()
        self.conv_1 = keras.layers.Conv2D(32, 3, activation="relu")
        self.conv_2 = keras.layers.Conv2D(64, 3, activation="relu")
        self.maxpool = keras.layers.MaxPooling2D(3)
        self.block_1 = ResNetBlock((64, 64))
        self.block_2 = ResNetBlock((64, 64))
        self.conv_3 = keras.layers.Conv2D(64, 3, activation='relu')
        self.glopool = keras.layers.GlobalAveragePooling2D()
        self.dense_1 = keras.layers.Dense(256, activation="relu")
        self.do = keras.layers.Dropout(0.5)
        self.dense_2 = keras.layers.Dense(10)

    def call(self, inputs):
        x = self.conv_1(inputs)
        x = self.conv_2(x)
        x = self.maxpool(x)
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.conv_3(x)
        x = self.glopool(x)
        x = self.dense_1(x)
        x = self.do(x)
        x = self.dense_2(x)
        return x


class ResNetBlock(keras.layers.Layer):
    def __init__(self, filters):
        super(ResNetBlock, self).__init__()
        filters1, filters2 = filters
        # first 3x3 conv
        self.conv2a = keras.layers.Conv2D(filters1, 3, activation='relu', padding='same')
        # second 3x3 conv
        self.conv2b = keras.layers.Conv2D(filters2, 3, activation='relu', padding='same')

    def call(self, inputs):
        x = self.conv2a(inputs)
        x = self.conv2b(x)
        x += inputs
        return x
