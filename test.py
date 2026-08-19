from algorithm.utils.data.tf.dataimage import DataImage
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
from absl import logging
logging.set_verbosity(logging.ERROR)
import argparse
import tensorflow as tf
import keras
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser(description='test.py')
parser.add_argument('--data_path', type=str, help='Path of the directory in which input data are stored\n(default: Input\dataset)', default="Input/dataset")
parser.add_argument('--name', type=str, help="Name of the dataset\n(default: 'Electronic components dataset')", default="Electronic components dataset")
parser.add_argument('--format', type=str, help="Image format (e.g.: RGB, Grayscale, RGBA). If not present, automatically deduced from images. \n(default: None)", default=None)
parser.add_argument('--model_path', type=str, help="Path of the directory from which the trained model has to be loaded.\n(default: in\model)", default="in/model")
args = parser.parse_args()

# for gpus.
# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#     for gpu in gpus:
#         tf.config.experimental.set_memory_growth(gpu, True)


dataset = DataImage(
    data_path=args.data_path,
    split=1,
    transform=None,
    normalize=True,
    mean='auto',
    std=None,
    resize=True,
    height='auto',
    width='auto',
    one_hot_encoding=True,
    name=args.name,
    format=args.format,
    buffer_size=500,
    batch_size=16
)

validset = dataset.get_set(split="validation")
class_names = dataset.labels

# get some random images and print them
# image_batch, label_batch = next(iter(trainset))
# plt.figure(figsize=(60, 60))
# for i in range(9):
#     ax = plt.subplot(3, 3, i + 1)
#     printable_object = dataset.print_item(image_batch['print_object'][i])
#     plt.imshow(printable_object)
#     label = label_batch[i].numpy().decode("utf-8")
#     plt.title(label, fontsize=20)
#     plt.axis("off")
# plt.show()


validset = dataset.apply_one_hot_encoding(validset)
model_dir = args.model_path

tf.get_logger().setLevel('ERROR')
model = keras.models.load_model(model_dir)

good = 0
tot = 0
batches = list(iter(validset))
for j in range(len(batches)):
    extr = batches[j]
    img, lab = extr
    for i in range(len(lab)):
        im = tf.expand_dims(img['data'][i], axis=0)
        printable_object = dataset.print_item(img['print_object'][i])
        po = tf.expand_dims(img['print_object'][i], axis=0)
        pred = model.predict({'data': im, 'print_object': po})
        pred_lidx = tf.argmax(pred[0])
        pred_label = class_names[pred_lidx]
        print("Predicted label: {}".format(pred_label))
        act_lidx = tf.argmax(lab[i])
        act_label = class_names[act_lidx]
        print("Real label: {}".format(act_label))
        print("Probability of the prediction:", pred[0][pred_lidx])
        print()
        if pred_lidx == act_lidx:
            good += 1
        tot += 1
        # To plot images and probabilities
        #plt.figure(figsize=(8, 8))
        label = pred_label
        #plt.title(label, fontsize=30)
        #plt.axis("off")
        #plt.imshow(printable_object)
        #plt.show()
        #plt.figure(figsize=(35, 10))
        #plt.bar(range(len(class_names)), pred[0], tick_label=class_names)
        #plt.xticks(rotation=45, fontsize=18)
        #plt.show()
        print("Partial score:", good, "/", tot, f"({(good/tot)*100}%)")
print("\n\n")
print("Score:", good, "/", tot, f"({(good/tot)*100}%)")