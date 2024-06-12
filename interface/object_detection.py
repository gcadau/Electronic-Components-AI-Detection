from algorithm.utils.data.tf.fullimage import FullImages, FullImage
from algorithm.utils.data.tf.fullimage import sliding_window, preprocess_windows
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


class Concurrency(Enum):
    Threads = auto()
    Processes = auto()

def classify_windows__batch(batch, class_names, model=None, model_path=None, shared_model=None):
    #global counter
    if model is None:
        if model_path is not None:
            model = tf.keras.models.load_model(model_path)
        if shared_model is not None:
            model = shared_model['model']
    windows_batched, positions_batched = batch
    prediction = model.predict(windows_batched, verbose=0)
    pred_labels = []
    probabilities = []
    for p in prediction:
        #sys.stdout.write('\rAnalized window: {}/{}'.format(counter + 1, total_windows))
        #sys.stdout.flush()
        #counter += 1
        pred_lidx = np.argmax(p)
        pred_labels.append(class_names[pred_lidx])
        probabilities.append(np.max(p))
    return positions_batched, pred_labels, probabilities

# Non maximum suppression
def non_maximum_suppression(detections, iou_threshold=0.5):
    if len(detections) == 0:
        return []

    # detections ordered in terms of confidencen score
    detections = sorted(detections, key=lambda x: x[5], reverse=True)
    nms_detections = []

    while detections:
        # Select detection with the highest confidence score
        best_detection = detections.pop(0)
        nms_detections.append(best_detection)
        
        remaining_detections = []
        
        for det in detections:
            if iou(best_detection, det) < iou_threshold:
                remaining_detections.append(det)
        
        detections = remaining_detections

    return nms_detections

# Compute Intersection over Union (IoU)
def iou(box1, box2):
    x1, y1, w1, h1 = box1[:4]
    x2, y2, w2, h2 = box2[:4]
    
    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)
    
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = w1 * h1
    box2_area = w2 * h2
    
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area != 0 else 0

def add_centroid(detection):
    x, y, w, h, label, confidence = detection

    x2, y2 = x + w, y + h
 
    centroid_x = (x + x2) / 2
    centroid_y = (y + y2) / 2
    
    detection_with_centroid = (x, y, w, h, centroid_x, centroid_y, label, confidence)
    
    return detection_with_centroid

