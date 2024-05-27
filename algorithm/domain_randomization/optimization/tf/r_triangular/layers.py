from tensorflow import keras
import tensorflow as tf
import numpy as np
import random
from tf.keras.utils import register_keras_serializable


@register_keras_serializable(package="Custom", name="RandomInvertoptimizationtfr_triangularlayers")
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

	@classmethod
	def from_config(cls, config):
		return cls(**config)

@register_keras_serializable(package="Custom", name="RandomBrightnessoptimizationtfr_triangularlayers")
class RandomBrightness(keras.layers.Layer):
    def __init__(self, seed=None, factor=0.9, **kwargs):
        # usual behaviour (not mandatory): lower=-{max_delta}, upper={max_delta} (-> delta sampled from
        # Tr[-delta, 0, delta], with delta value to be applyed to brightness)
        super().__init__(**kwargs)
        self.seed = seed
        self.factor = factor


    def call(self, x, training=None, lower=-0.2, mode=0, upper=0.2, value=None, rand=True):
        if mode is None:
            mode = (upper-lower)/2
        if rand:
            if training:
                if x.shape[0] is not None: 
                    if self.seed is not None:
                        tf.random.set_seed(self.seed)
                        rng = np.random.default_rng(seed=self.seed)
                        delta = rng.triangular(lower, mode, upper, size=x.shape[0])
                    else:
                        delta = np.random.triangular(lower, mode, upper, size=x.shape[0])
                    ims = []
                    for i in range(x.shape[0]):
                        im = x[i,:,:,:]
                        if tf.random.uniform([]) <= self.factor:
                            im = tf.image.adjust_brightness(im, delta[i])
                        ims.append(im)
                    return tf.stack(ims)
                else: # no batches or eager execution not enabled
                    if self.seed is not None:
                        tf.random.set_seed(self.seed)
                        rng = np.random.default_rng(seed=self.seed)
                        delta = rng.triangular(lower, mode, upper, size=1)[0]
                    else:
                        delta = np.random.triangular(lower, mode, upper, size=1)[0]
                    if random.random() <= self.factor:
                        return tf.map_fn(lambda x: tf.image.adjust_brightness(x, delta), x)
                    else:
                        return x  
            else:
                return x
        else:
            if training:
                if x.shape[0] is not None: 
                    ims = []
                    for i in range(x.shape[0]):
                        im = x[i,:,:,:]
                        if tf.random.uniform([]) <= self.factor:
                            delta = value[i]
                            im = tf.image.adjust_brightness(im, delta)
                        ims.append(im)
                    return tf.stack(ims)
                else: # no batches or eager execution not enabled
                    if random.random() <= self.factor:
                        delta = value[0]
                        return tf.map_fn(lambda x: tf.image.adjust_brightness(x, delta), x)
                    else:
                        return x  
            else:
                return x

    def get_config(self):
        config = super().get_config()
        config.update({'seed': self.seed, 'factor': self.factor})
        return config

	@classmethod
	def from_config(cls, config):
		return cls(**config)

@register_keras_serializable(package="Custom", name="RandomContrastoptimizationtfr_triangularlayers")
class RandomContrast(keras.layers.Layer):
    def __init__(self, seed=None, factor=0.9, **kwargs):
        super().__init__(**kwargs)
        self.seed = seed
        self.factor = factor


    def call(self, x, training=None, lower=0, upper=2.5, mode=None, value=None, rand=True):
        if mode is None:
            mode = (upper-lower)/2
        if rand:
            if training:
                if x.shape[0] is not None: 
                    if self.seed is not None:
                        tf.random.set_seed(self.seed)
                        rng = np.random.default_rng(seed=self.seed)
                        contrast_factor = rng.triangular(lower, mode, upper, size=x.shape[0])
                    else:
                        contrast_factor = np.random.triangular(lower, mode, upper, size=x.shape[0])
                    ims = []
                    for i in range(x.shape[0]):
                        im = x[i,:,:,:]
                        if tf.random.uniform([]) <= self.factor:
                            im = tf.image.adjust_contrast(im, contrast_factor[i])
                        ims.append(im)
                    return tf.stack(ims)
                else: # no batches or eager execution not enabled
                    if self.seed is not None:
                        tf.random.set_seed(self.seed)
                        rng = np.random.default_rng(seed=self.seed)
                        contrast_factor = rng.triangular(lower, mode, upper, size=1)[0]
                    else:
                        contrast_factor = np.random.triangular(lower, mode, upper, size=1)[0]
                    if random.random() <= self.factor:
                        return tf.map_fn(lambda x: tf.image.adjust_contrast(x, contrast_factor), x)
                    else:
                        return x  
            else:
                return x
        else:
            if training:
                if x.shape[0] is not None: 
                    ims = []
                    for i in range(x.shape[0]):
                        im = x[i,:,:,:]
                        if tf.random.uniform([]) <= self.factor:
                            contrast_factor = value[i]
                            im = tf.image.adjust_contrast(im, contrast_factor)
                        ims.append(im)
                    return tf.stack(ims)
                else: # no batches or eager execution not enabled
                    if random.random() <= self.factor:
                        contrast_factor = value[0]
                        return tf.map_fn(lambda x: tf.image.adjust_contrast(x, contrast_factor), x)
                    else:
                        return x  
            else:
                return x

    def get_config(self):
        config = super().get_config()
        config.update({'seed': self.seed, 'factor': self.factor})
        return config

	@classmethod
	def from_config(cls, config):
		return cls(**config)

# possibile to define class RandomCrop(keras.layers.Layer), not so useful.


@register_keras_serializable(package="Custom", name="RandomHorizontallyFlipoptimizationtfr_triangularlayers")
class RandomHorizontallyFlip(keras.layers.Layer):
    def __init__(self, seed=None, factor=0.9, **kwargs):
        super().__init__(**kwargs)
        self.seed = seed
        self.factor = factor


    def call(self, x, training=None, lower=0, upper=1, mode=0.5, value=None, rand=True):
        if mode is None:
            mode = (upper-lower)/2
        if rand:
            if training:
                if x.shape[0] is not None: 
                    prob = None
                    if self.seed is not None:
                        tf.random.set_seed(self.seed)
                        rng = np.random.default_rng(seed=self.seed)
                        prob = rng.triangular(lower, mode, upper, size=x.shape[0])
                    else:
                        prob = np.random.triangular(lower, mode, upper, size=x.shape[0])
                    ims = []
                    for i in range(x.shape[0]):
                        im = x[i,:,:,:]
                        if tf.random.uniform([]) <= self.factor and prob[i] < 0.5:
                            im = tf.image.flip_left_right(im)
                        ims.append(im)
                    return tf.stack(ims)
                else: # no batches or eager execution not enabled
                    prob = None
                    if self.seed is not None:
                        tf.random.set_seed(self.seed)
                        rng = np.random.default_rng(seed=self.seed)
                        prob = rng.triangular(lower, mode, upper, size=1)[0]
                    else:
                        prob = np.random.triangular(lower, mode, upper, size=1)[0]
                    if random.random() <= self.factor and prob < 0.5:
                        return tf.map_fn(lambda x: tf.image.flip_left_right(x), x)
                    else:
                        return x  
            else:
                return x
        else:
            if training:
                if x.shape[0] is not None: 
                    ims = []
                    for i in range(x.shape[0]):
                        im = x[i,:,:,:]
                        prob = value[i]
                        if tf.random.uniform([]) <= self.factor and prob < 0.5:
                            im = tf.image.flip_left_right(im)
                        ims.append(im)
                    return tf.stack(ims)
                else: # no batches or eager execution not enabled
                    prob = value[0]
                    if random.random() <= self.factor and prob < 0.5:
                        return tf.map_fn(lambda x: tf.image.flip_left_right(x), x)
                    else:
                        return x  
            else:
                return x

    def get_config(self):
        config = super().get_config()
        config.update({'seed': self.seed, 'factor': self.factor})
        return config

	@classmethod
	def from_config(cls, config):
		return cls(**config)

@register_keras_serializable(package="Custom", name="RandomVerticallyFlipoptimizationtfr_triangularlayers")
class RandomVerticallyFlip(keras.layers.Layer):
    def __init__(self, seed=None, factor=0.9, **kwargs):
        super().__init__(**kwargs)
        self.seed = seed
        self.factor = factor


    def call(self, x, training=None, lower=0, upper=1, mode=0.5, value=None, rand=True):
        if mode is None:
            mode = (upper-lower)/2
        if rand:
            if training:
                if x.shape[0] is not None: 
                    prob = None
                    if self.seed is not None:
                        tf.random.set_seed(self.seed)
                        rng = np.random.default_rng(seed=self.seed)
                        prob = rng.triangular(lower, mode, upper, size=x.shape[0])
                    else:
                        prob = np.random.triangular(lower, mode, upper, size=x.shape[0])
                    ims = []
                    for i in range(x.shape[0]):
                        im = x[i,:,:,:]
                        if tf.random.uniform([]) <= self.factor and prob[i] < 0.5:
                            im = tf.image.flip_up_down(im)
                        ims.append(im)
                    return tf.stack(ims)
                else: # no batches or eager execution not enabled
                    prob = None
                    if self.seed is not None:
                        tf.random.set_seed(self.seed)
                        rng = np.random.default_rng(seed=self.seed)
                        prob = rng.triangular(lower, mode, upper, size=1)[0]
                    else:
                        prob = np.random.triangular(lower, mode, upper, size=1)[0]
                    if random.random() <= self.factor and prob < 0.5:
                        return tf.map_fn(lambda x: tf.image.flip_up_down(x), x)
                    else:
                        return x  
            else:
                return x
        else:
            if training:
                if x.shape[0] is not None: 
                    ims = []
                    for i in range(x.shape[0]):
                        im = x[i,:,:,:]
                        prob = value[i]
                        if tf.random.uniform([]) <= self.factor and prob < 0.5:
                            im = tf.image.flip_up_down(im)
                        ims.append(im)
                    return tf.stack(ims)
                else: # no batches or eager execution not enabled
                    prob = value[0]
                    if random.random() <= self.factor and prob < 0.5:
                        return tf.map_fn(lambda x: tf.image.flip_up_down(x), x)
                    else:
                        return x  
            else:
                return x

    def get_config(self):
        config = super().get_config()
        config.update({'seed': self.seed, 'factor': self.factor})
        return config

	@classmethod
	def from_config(cls, config):
		return cls(**config)

@register_keras_serializable(package="Custom", name="RandomHueoptimizationtfr_triangularlayers")
class RandomHue(keras.layers.Layer):
    def __init__(self, seed=None, factor=0.9, **kwargs):
        # usual behaviour (not mandatory): lower=-{max_delta}, upper={max_delta} (-> delta sampled from
        # Tr[-delta, 0, delta], with delta value to be applyed to hue)
        super().__init__(**kwargs)
        self.seed = seed
        self.factor = factor


    def call(self, x, training=None, lower=-0.2, mode=0, upper=0.2, value=None, rand=True):
        if mode is None:
            mode = (upper-lower)/2
        if rand:
            if training:
                if x.shape[0] is not None: 
                    if self.seed is not None:
                        tf.random.set_seed(self.seed)
                        rng = np.random.default_rng(seed=self.seed)
                        delta = rng.triangular(lower, mode, upper, size=x.shape[0])
                    else:
                        delta = np.random.triangular(lower, mode, upper, size=x.shape[0])
                    ims = []
                    for i in range(x.shape[0]):
                        im = x[i,:,:,:]
                        if tf.random.uniform([]) <= self.factor:
                            im = tf.image.adjust_hue(im, delta[i])
                        ims.append(im)
                    return tf.stack(ims)
                else: # no batches or eager execution not enabled
                    if self.seed is not None:
                        tf.random.set_seed(self.seed)
                        rng = np.random.default_rng(seed=self.seed)
                        delta = rng.triangular(lower, mode, upper, size=1)[0]
                    else:
                        delta = np.random.triangular(lower, mode, upper, size=1)[0]
                    if random.random() <= self.factor:
                        return tf.map_fn(lambda x: tf.image.adjust_hue(x, delta), x)
                    else:
                        return x  
            else:
                return x
        else:
            if training:
                if x.shape[0] is not None: 
                    ims = []
                    for i in range(x.shape[0]):
                        im = x[i,:,:,:]
                        delta = value[i]
                        if tf.random.uniform([]) <= self.factor:
                            im = tf.image.adjust_hue(im, delta)
                        ims.append(im)
                    return tf.stack(ims)
                else: # no batches or eager execution not enabled
                    if random.random() <= self.factor:
                        delta = value[0]
                        return tf.map_fn(lambda x: tf.image.adjust_hue(x, delta), x)
                    else:
                        return x  
            else:
                return x

    def get_config(self):
        config = super().get_config()
        config.update({'seed': self.seed, 'factor': self.factor})
        return config

	@classmethod
	def from_config(cls, config):
		return cls(**config)

@register_keras_serializable(package="Custom", name="RandomJpegQualityoptimizationtfr_triangularlayers")
class RandomJpegQuality(keras.layers.Layer):
    def __init__(self, seed=None, factor=0.9, **kwargs):
        super().__init__(**kwargs)
        self.seed = seed
        self.factor = factor


    def call(self, x, training=None, lower=20, upper=100, mode=None, value=None, rand=True):
        if mode is None:
            mode = (upper-lower)/2
        if rand:
            if training:
                if x.shape[0] is not None: 
                    if self.seed is not None:
                        tf.random.set_seed(self.seed)
                        rng = np.random.default_rng(seed=self.seed)
                        jpeg_quality = rng.triangular(lower, mode, upper, size=x.shape[0])
                    else:
                        jpeg_quality = np.random.triangular(lower, mode, upper, size=x.shape[0])
                    ims = []
                    for i in range(x.shape[0]):
                        im = x[i,:,:,:]
                        if tf.random.uniform([]) <= self.factor:
                            im = tf.image.adjust_jpeg_quality(im, jpeg_quality[i])
                        ims.append(im)
                    return tf.stack(ims)
                else: # no batches or eager execution not enabled
                    if self.seed is not None:
                        tf.random.set_seed(self.seed)
                        rng = np.random.default_rng(seed=self.seed)
                        jpeg_quality = rng.triangular(lower, mode, upper, size=1)[0]
                    else:
                        jpeg_quality = np.random.triangular(lower, mode, upper, size=1)[0]
                    if random.random() <= self.factor:
                        return tf.map_fn(lambda x: tf.image.adjust_jpeg_quality(x, jpeg_quality), x)
                    else:
                        return x  
            else:
                return x
        else:
            if training:
                if x.shape[0] is not None: 
                    ims = []
                    for i in range(x.shape[0]):
                        im = x[i,:,:,:]
                        jpeg_quality = value[i]
                        if tf.random.uniform([]) <= self.factor:
                            im = tf.image.adjust_jpeg_quality(im, jpeg_quality)
                        ims.append(im)
                    return tf.stack(ims)
                else: # no batches or eager execution not enabled
                    if random.random() <= self.factor:
                        jpeg_quality = value[0]
                        return tf.map_fn(lambda x: tf.image.adjust_jpeg_quality(x, jpeg_quality), x)
                    else:
                        return x  
            else:
                return x

    def get_config(self):
        config = super().get_config()
        config.update({'seed': self.seed, 'factor': self.factor})
        return config

	@classmethod
	def from_config(cls, config):
		return cls(**config)

@register_keras_serializable(package="Custom", name="RandomSaturationoptimizationtfr_triangularlayers")
class RandomSaturation(keras.layers.Layer):
    def __init__(self, seed=None, factor=0.9, **kwargs):
        super().__init__(**kwargs)
        self.seed = seed
        self.factor = factor


    def call(self, x, training=None, lower=0, upper=2, mode=None, value=None, rand=True):
        if mode is None:
            mode = (upper-lower)/2
        if rand:
            if training:
                if x.shape[0] is not None: 
                    if self.seed is not None:
                        tf.random.set_seed(self.seed)
                        rng = np.random.default_rng(seed=self.seed)
                        saturation_factor = rng.triangular(lower, mode, upper, size=x.shape[0])
                    else:
                        saturation_factor = np.random.triangular(lower, mode, upper, size=x.shape[0])
                    ims = []
                    for i in range(x.shape[0]):
                        im = x[i,:,:,:]
                        if tf.random.uniform([]) <= self.factor:
                            im = tf.image.adjust_saturation(im, saturation_factor[i])
                        ims.append(im)
                    return tf.stack(ims)
                else: # no batches or eager execution not enabled
                    if self.seed is not None:
                        tf.random.set_seed(self.seed)
                        rng = np.random.default_rng(seed=self.seed)
                        saturation_factor = rng.triangular(lower, mode, upper, size=1)[0]
                    else:
                        saturation_factor = np.random.triangular(lower, mode, upper, size=1)[0]
                    if random.random() <= self.factor:
                        return tf.map_fn(lambda x: tf.image.adjust_saturation(x, saturation_factor), x)
                    else:
                        return x  
            else:
                return x
        else:
            if training:
                if x.shape[0] is not None: 
                    ims = []
                    for i in range(x.shape[0]):
                        im = x[i,:,:,:]
                        saturation_factor = value[i]
                        if tf.random.uniform([]) <= self.factor:
                            im = tf.image.adjust_saturation(im, saturation_factor)
                        ims.append(im)
                    return tf.stack(ims)
                else: # no batches or eager execution not enabled
                    if random.random() <= self.factor:
                        saturation_factor = value[0]
                        return tf.map_fn(lambda x: tf.image.adjust_saturation(x, saturation_factor), x)
                    else:
                        return x  
            else:
                return x


    def get_config(self):
        config = super().get_config()
        config.update({'seed': self.seed, 'factor': self.factor})
        return config

	@classmethod
	def from_config(cls, config):
		return cls(**config)