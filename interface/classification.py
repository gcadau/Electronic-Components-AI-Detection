from algorithm.utils.data.tf.fullimage import FullImages, FullImage
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
tf.get_logger().setLevel('WARNING')
import warnings
from absl import logging
logging.set_verbosity(logging.ERROR)
import argparse
import tensorflow as tf
import keras
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import sys
tf.get_logger().setLevel('WARNING')
from enum import Enum, auto



def classify_image(image, class_names, model=None, model_path=None, shared_model=None):
    if model is None:
        if model_path is not None:
            model = tf.keras.models.load_model(model_path)
        if shared_model is not None:
            model = shared_model['model']
    prediction = model.predict(image, verbose=0)
    pred_labels = []
    probabilities = []
    for p in prediction:
        pred_lidx = np.argmax(p)
        pred_labels.append(class_names[pred_lidx])
        probabilities.append(np.max(p))
    return pred_labels[0], probabilities[0]