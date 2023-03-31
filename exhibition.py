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
from fn import configure_for_performance, get_optimizer

image_count = None

list_ds = tf.data.Dataset.list_files(str(), shuffle=False)
list_ds = list_ds.shuffle(image_count, reshuffle_each_iteration=False)

val_size = int(image_count * 0.2)
test_ds = list_ds.skip(val_size)
val_ds = list_ds.take(val_size)

model = tf.keras.models.load_model('model_save/model.h5')
temp_layer_storage = None
train_layers = False

if(len(model) == 10):

    temp_layer_storage = model[:2]
    model = model[2:]
    train_layers = True


if(train_layers == True):
    model = temp_layer_storage + model


