from tensorflow import keras
import tensorflow as tf
import numpy as np
import random


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


class Brightness(keras.layers.Layer):
    def __init__(self, par, **kwargs):
        super().__init__(**kwargs)
        if par<-1:
            par = -1
        if par>1:
            par = 1
        self.delta = par


    def call(self, x, training=None):
        if training:
            return tf.image.adjust_brightness(x, self.delta)
        else:
            return x

    def get_config(self):
        config = super().get_config()
        config.update({'par': self.delta})
        return config


class Contrast(keras.layers.Layer):
    def __init__(self, par, **kwargs):
        super().__init__(**kwargs)
        self.contrast_factor = par


    def call(self, x, training=None):
        if training:
            return tf.image.adjust_contrast(x, self.contrast_factor)
        else:
            return x

    def get_config(self):
        config = super().get_config()
        config.update({'par': self.contrast_factor})
        return config


# possibile to define class Crop(keras.layers.Layer), not so useful.


class HorizontallyFlip(keras.layers.Layer):
    def __init__(self, par, **kwargs):
        super().__init__(**kwargs)
        self.prob = par


    def call(self, x, training=None):
        if training:
            if self.prob < 0.5:
                return tf.image.flip_left_right(x)
            else:
                return x
        else:
            return x

    def get_config(self):
        config = super().get_config()
        config.update({'par': self.prob})
        return config


class VerticallyFlip(keras.layers.Layer):
    def __init__(self, par, **kwargs):
        super().__init__(**kwargs)
        self.prob = par


    def call(self, x, training=None):
        if training:
            if self.prob < 0.5:
                return tf.image.flip_up_down(x)
            else:
                return x
        else:
            return x

    def get_config(self):
        config = super().get_config()
        config.update({'par': self.prob})
        return config


class Hue(keras.layers.Layer):
    def __init__(self, par, **kwargs):
        super().__init__(**kwargs)
        if par<-1:
            par = -1
        if par>1:
            par = 1
        self.delta = par


    def call(self, x, training=None):
        if training:
            return tf.image.adjust_hue(x, self.delta)
        else:
            return x

    def get_config(self):
        config = super().get_config()
        config.update({'par': self.delta})
        return config


class JpegQuality(keras.layers.Layer):
    def __init__(self, par, **kwargs):
        super().__init__(**kwargs)
        if par<0:
            par = 0
        if par>100:
            par = 100
        self.jpeg_quality = par


    def call(self, x, training=None):
        if training:
            return tf.image.adjust_jpeg_quality(x, self.jpeg_quality)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({'par': self.jpeg_quality})
        return config


class Saturation(keras.layers.Layer):
    def __init__(self, par, **kwargs):
        super().__init__(**kwargs)
        if par<0:
            par = 0
        self.saturation_factor = par


    def call(self, x, training=None):
        if training:
            return tf.image.adjust_saturation(x, self.saturation_factor)
        else:
            return x

    def get_config(self):
        config = super().get_config()
        config.update({'par': self.saturation_factor})
        return config





class RandomParameters(keras.layers.Layer):
    def __init__(self, seed=None, factors=None, **kwargs):
        super().__init__(**kwargs)

        if factors is None:
            factors = [0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9]

        self.factor = factors
        self.seed = seed


    def call(self, x, mean=None, variance=None, values=None, training=None, rand=True):
        if training:
            if rand:
                self.randoms = []
                if self.seed is not None:
                    tf.random.set_seed(self.seed)
                    rng = np.random.default_rng(seed=self.seed)
                    random_parameters = rng.multivariate_normal(mean, variance, size=x.shape[0])
                else:
                    random_parameters = np.random.multivariate_normal(mean, variance, size=x.shape[0])

                random_brightness = []
                if x.shape[0] is not None:
                    for i in range(x.shape[0]):
                        if tf.random.uniform([]) <= self.factor[0]:
                            params = {'par': random_parameters[i][0]}
                            random_brightness.append(Brightness(**params))
                        else:
                            random_brightness.append(NoneTransformation())
                else: # no batches or eager execution not enabled
                    if random.random() <= self.factor[0]:
                        params = {'par': random_parameters[0]}
                        random_brightness.append(Brightness(**params))
                    else:
                        random_brightness.append(NoneTransformation())
                self.randoms.append(random_brightness)
                random_contrast = []
                if x.shape[0] is not None:
                    for i in range(x.shape[0]):
                        if tf.random.uniform([]) <= self.factor[1]:
                            params = {'par': random_parameters[i][1]}
                            random_contrast.append(Contrast(**params))
                        else:
                            random_contrast.append(NoneTransformation())
                else: # no batches or eager execution not enabled
                    if random.random() <= self.factor[1]:
                        params = {'par': random_parameters[1]}
                        random_contrast.append(Contrast(**params))
                    else:
                        random_contrast.append(NoneTransformation())
                self.randoms.append(random_contrast)
                random_horizontallyFlip = []
                if x.shape[0] is not None:
                    for i in range(x.shape[0]):
                        if tf.random.uniform([]) <= self.factor[2]:
                            params = {'par': random_parameters[i][2]}
                            random_horizontallyFlip.append(HorizontallyFlip(**params))
                        else:
                            random_horizontallyFlip.append(NoneTransformation())
                else: # no batches or eager execution not enabled
                    if random.random() <= self.factor[2]:
                        params = {'par': random_parameters[2]}
                        random_horizontallyFlip.append(HorizontallyFlip(**params))
                    else:
                        random_horizontallyFlip.append(NoneTransformation())
                self.randoms.append(random_horizontallyFlip)
                random_verticallyFlip = []
                if x.shape[0] is not None:
                    for i in range(x.shape[0]):
                        if tf.random.uniform([]) <= self.factor[3]:
                            params = {'par': random_parameters[i][3]}
                            random_verticallyFlip.append(VerticallyFlip(**params))
                        else:
                            random_verticallyFlip.append(NoneTransformation())
                else: # no batches or eager execution not enabled
                    if random.random() <= self.factor[3]:
                        params = {'par': random_parameters[3]}
                        random_verticallyFlip.append(VerticallyFlip(**params))
                    else:
                        random_verticallyFlip.append(NoneTransformation())
                self.randoms.append(random_verticallyFlip)
                random_hue = []
                if x.shape[0] is not None:
                    for i in range(x.shape[0]):
                        if tf.random.uniform([]) <= self.factor[4]:
                            params = {'par': random_parameters[i][4]}
                            random_hue.append(Hue(**params))
                        else:
                            random_hue.append(NoneTransformation())
                else: # no batches or eager execution not enabled
                    if random.random() <= self.factor[4]:
                        params = {'par': random_parameters[4]}
                        random_hue.append(Hue(**params))
                    else:
                        random_hue.append(NoneTransformation())
                self.randoms.append(random_hue)
                random_jpegQuality = []
                if x.shape[0] is not None:
                    for i in range(x.shape[0]):
                        if tf.random.uniform([]) <= self.factor[5]:
                            params = {'par': random_parameters[i][5]}
                            random_jpegQuality.append(JpegQuality(**params))
                        else:
                            random_jpegQuality.append(NoneTransformation())
                else: # no batches or eager execution not enabled
                    if random.random() <= self.factor[5]:
                        params = {'par': random_parameters[5]}
                        random_jpegQuality.append(JpegQuality(**params))
                    else:
                        random_jpegQuality.append(NoneTransformation())
                self.randoms.append(random_jpegQuality)
                random_saturation = []
                if x.shape[0] is not None:
                    for i in range(x.shape[0]):
                        if tf.random.uniform([]) <= self.factor[6]:
                            params = {'par': random_parameters[i][6]}
                            random_saturation.append(Saturation(**params))
                        else:
                            random_saturation.append(NoneTransformation())
                else: # no batches or eager execution not enabled
                    if random.random() <= self.factor[6]:
                        params = {'par': random_parameters[6]}
                        random_saturation.append(Saturation(**params))
                    else:
                        random_saturation.append(NoneTransformation())
                self.randoms.append(random_saturation)

                for i in range(len(self.factor)):
                    ran = self.randoms[i]
                    ims = []
                    if x.shape[0] is not None:
                        for j in range(x.shape[0]):
                            im = x[j,:,:,:]
                            im = ran[j](im, training=True)
                            ims.append(im)
                        x = tf.stack(ims)
                    else: # no batches or eager execution not enabled
                        x = tf.map_fn(lambda x: ran[0](x, training=True), x)
            else:
                self.randoms = []
                random_brightness = []
                if x.shape[0] is not None:
                    for i in range(x.shape[0]):
                        if tf.random.uniform([]) <= self.factor[0]:
                            params = {'par': values[i][0]}
                            random_brightness.append(Brightness(**params))
                        else:
                            random_brightness.append(NoneTransformation())
                else: # no batches or eager execution not enabled
                    if random.random() <= self.factor[0]:
                        params = {'par': values[0]}
                        random_brightness.append(Brightness(**params))
                    else:
                        random_brightness.append(NoneTransformation())
                self.randoms.append(random_brightness)
                random_contrast = []
                if x.shape[0] is not None:
                    for i in range(x.shape[0]):
                        if tf.random.uniform([]) <= self.factor[1]:
                            params = {'par': values[i][1]}
                            random_contrast.append(Contrast(**params))
                        else:
                            random_contrast.append(NoneTransformation())
                else: # no batches or eager execution not enabled
                    if random.random() <= self.factor[1]:
                        params = {'par': values[1]}
                        random_contrast.append(Contrast(**params))
                    else:
                        random_contrast.append(NoneTransformation())
                self.randoms.append(random_contrast)
                random_horizontallyFlip = []
                if x.shape[0] is not None:
                    for i in range(x.shape[0]):
                        if tf.random.uniform([]) <= self.factor[2]:
                            params = {'par': values[i][2]}
                            random_horizontallyFlip.append(HorizontallyFlip(**params))
                        else:
                            random_horizontallyFlip.append(NoneTransformation())
                else: # no batches or eager execution not enabled
                    if random.random() <= self.factor[2]:
                        params = {'par': values[2]}
                        random_horizontallyFlip.append(HorizontallyFlip(**params))
                    else:
                        random_horizontallyFlip.append(NoneTransformation())
                self.randoms.append(random_horizontallyFlip)
                random_verticallyFlip = []
                if x.shape[0] is not None:
                    for i in range(x.shape[0]):
                        if tf.random.uniform([]) <= self.factor[3]:
                            params = {'par': values[i][3]}
                            random_verticallyFlip.append(VerticallyFlip(**params))
                        else:
                            random_verticallyFlip.append(NoneTransformation())
                else: # no batches or eager execution not enabled
                    if random.random() <= self.factor[3]:
                        params = {'par': values[3]}
                        random_verticallyFlip.append(VerticallyFlip(**params))
                    else:
                        random_verticallyFlip.append(NoneTransformation())
                self.randoms.append(random_verticallyFlip)
                random_hue = []
                if x.shape[0] is not None:
                    for i in range(x.shape[0]):
                        if tf.random.uniform([]) <= self.factor[4]:
                            params = {'par': values[i][4]}
                            random_hue.append(Hue(**params))
                        else:
                            random_hue.append(NoneTransformation())
                else: # no batches or eager execution not enabled
                    if random.random() <= self.factor[4]:
                        params = {'par': values[4]}
                        random_hue.append(Hue(**params))
                    else:
                        random_hue.append(NoneTransformation())
                self.randoms.append(random_hue)
                random_jpegQuality = []
                if x.shape[0] is not None:
                    for i in range(x.shape[0]):
                        if tf.random.uniform([]) <= self.factor[5]:
                            params = {'par': values[i][5]}
                            random_jpegQuality.append(JpegQuality(**params))
                        else:
                            random_jpegQuality.append(NoneTransformation())
                else: # no batches or eager execution not enabled
                    if random.random() <= self.factor[5]:
                        params = {'par': values[5]}
                        random_jpegQuality.append(JpegQuality(**params))
                    else:
                        random_jpegQuality.append(NoneTransformation())
                self.randoms.append(random_jpegQuality)
                random_saturation = []
                if x.shape[0] is not None:
                    for i in range(x.shape[0]):
                        if tf.random.uniform([]) <= self.factor[6]:
                            params = {'par': values[i][6]}
                            random_saturation.append(Saturation(**params))
                        else:
                            random_saturation.append(NoneTransformation())
                else: # no batches or eager execution not enabled
                    if random.random() <= self.factor[6]:
                        params = {'par': values[6]}
                        random_saturation.append(Saturation(**params))
                    else:
                        random_saturation.append(NoneTransformation())
                self.randoms.append(random_saturation)

                for i in range(len(self.factor)):
                    ran = self.randoms[i]
                    ims = []
                    if x.shape[0] is not None:
                        for j in range(x.shape[0]):
                            im = x[j,:,:,:]
                            im = ran[j](im, training=True)
                            ims.append(im)
                        x = tf.stack(ims)
                    else: # no batches or eager execution not enabled
                        x = tf.map_fn(lambda x: ran[0](x, training=True), x)

        return x

    def get_config(self):
        config = super().get_config()
        config.update({"seed": self.seed, "factors": self.factors})
        return config



class NoneTransformation():
    def __int__(self):
        pass

    def __call__(self, x, training=None):
        return x