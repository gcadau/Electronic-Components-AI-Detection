from algorithm.utils.data.tf.fullimage import FullImage
from algorithm.utils.data.tf.fullimage import preprocess_image
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


def classify_image(image, model, class_names):
    prediction = model.predict(image, verbose=0)
    pred_labels = []
    probabilities = []
    for p in prediction:
        pred_lidx = np.argmax(p)
        pred_labels.append(class_names[pred_lidx])
        probabilities.append(np.max(p))
    return pred_labels[0], probabilities[0]



parser = argparse.ArgumentParser(description='prediction.py')
parser.add_argument('--data_path', type=str, help='Path of the directory in which input data are stored\n(default: Input\\sections)', default="Input\\sections")
parser.add_argument('--name', type=str, help="Name of the dataset\n(default: 'Electronic components dataset')", default="Electronic components dataset")
parser.add_argument('--format', type=str, help="Image format (e.g.: RGB, Grayscale, RGBA). If not present, automatically deduced from images. \n(default: None)", default=None)
parser.add_argument('--model_path', type=str, help="Path of the directory from which the trained model has to be loaded.\n(default: in\model)", default="in\\model")
args = parser.parse_args()

# for gpus.
# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#     for gpu in gpus:
#         tf.config.experimental.set_memory_growth(gpu, True)


sections = FullImage(data_path=args.data_path)

single_images = sections.get_set()
single_image = single_images[0]

#get first image and print it
# printable_object = sections.print_item(single_images)
# plt.imshow(printable_object)
# plt.axis('off')  

class_names = sections.labels


model_dir = "in\\model"
tf.get_logger().setLevel('ERROR')

model = tf.keras.models.load_model(model_dir)

for layer in model.layers:
    if hasattr(layer, 'input_shape'):
        input_shape = layer.input_shape
        break
model__input_shape=input_shape[-3:-1]

resize = True
normalize = True
img_height, img_width = (128, 128) 
mean = [float(1/255), float(1/255), float(1/255)]
num_channels = 3
printable_object = tf.constant("-1", dtype=tf.string)



image = preprocess_image(
                                single_image, 
                                resize=resize, 
                                normalize=normalize, 
                                img_height=img_height, 
                                img_width=img_width, 
                                mean=mean, 
                                printable_object=printable_object
                               )

classification = classify_image(image, model=model, class_names=class_names)
label, confidence = classification


# print(f"Image classification: {label} with confidence {confidence}")

#Plot image classification and probabilities
# plt.figure(figsize=(8, 8))
# plt.title(label, fontsize=30)
# plt.axis("off")
# printable_object = sections.print_item(single_image)
# plt.imshow(printable_object)
# plt.show()
