# tensorflow imports
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Model
import tensorflow_datasets as tfds
# standard imports
import matplotlib as mpl
import matplotlib.pyplot as mplpy
import PIL
import PIL.Image
import json
import os
# file imports
from fn import comma_addremove, configure_for_performance, safeinput

image_count = None

list_ds = tf.data.Dataset.list_files(str(), shuffle=False)
list_ds = list_ds.shuffle(image_count, reshuffle_each_iteration=False)

val_size = int(image_count * 0.2)
test_ds = list_ds.skip(val_size)
val_ds = list_ds.take(val_size)