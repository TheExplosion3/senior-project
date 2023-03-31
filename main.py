# tensorflow imports
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_datasets as tfds
# standard imports
import matplotlib.pyplot as mplpy
import numpy as np
import datetime
import time
import json
import os
# file imports
from fn import configure_for_performance, get_optimizer


# Variable initialization

# timer start, will show full elapsed time that the program took to execute.
start_time = datetime.datetime.now()

# dataset init
(train_ds, test_ds), metadata = tfds.load(
    'cifar100',
    split=['train[:80%]', 'test[90%:]'],
    with_info=True,
    as_supervised=True,
)

# other important stuff
num_classes = metadata.features['label'].num_classes
epochs = 10
batch_size = 5
steps_per_epoch = len(train_ds)//batch_size

# empty variables due to how python clears memory based upon scope
f, model = None, None

# randomly rotates, zooms, and flips images to introduce more realistic cases for the image, i don't know how it integrates yet though.
data_augmentation = keras.Sequential(
  [
    tf.keras.layers.RandomFlip("horizontal", input_shape=(32,32,3)),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
  ]
)


# checks if the model has been created in the past, if not then it creates it on the spot.
# this network is convolutional, as shown by the first parts.
# stats:
# amt of nodes on input layer: 4
# amt of nodes per hidden layer: 37/38
# amt of nodes on output layer: 100
if os.stat("model_save/model.h5").st_size == 0:
  model = tf.keras.Sequential([
      data_augmentation,
      tf.keras.layers.Conv2D(4, (3, 3), padding='same', activation='elu', input_shape=(32, 32, 3)),
      tf.keras.layers.MaxPooling2D((2, 2)),
      tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='elu'),
      tf.keras.layers.MaxPooling2D((2, 2)),
      tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='elu'),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dropout(0.5),
      tf.keras.layers.Dense(37, activation='leaky_relu'),
      tf.keras.layers.Dense(38, activation='leaky_relu'),
      tf.keras.layers.Dense(num_classes, activation='sigmoid')
  ])
else:
  model = tf.keras.models.load_model('model_save/model.h5')

print("Compiling Model...")

lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
  0.001,
  decay_steps=steps_per_epoch*1000,
  decay_rate=1,
  staircase=False)

model.compile(optimizer=get_optimizer(lr_schedule),
loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy', 'mae'])

print("Model Compiled.")

if os.stat("model_save/model.h5").st_size != 0:
  print("Current Model Statistics:")
  model.summmary()

print(f"Training beginning, running {epochs} epoch(s)")

image, label = next(iter(train_ds))
train_ds = configure_for_performance(train_ds)
test_ds = configure_for_performance(test_ds)

# training loop. this code starts with an image, then it takes a label as well, and checks whether or not the image is that label. after looping through, it shows avg. accuracy, then waits 5 seconds, then proceeds.


model.fit(x=train_ds, epochs=epochs, steps_per_epoch=steps_per_epoch, validation_data=[test_ds])
acc=model.evaluate(image, label, verbose=2)
print(f'\naverage accuracy: {(100 * np.mean(acc)):5.2f}\n')
time.sleep(5)

print("Training ended. Creating savepoints.")

### saving phase ###
# this code is self explanatory
print("Saving data...")
model.save('model_save/model.h5')
print("Save complete.")

# timer concludes, showing total time elapsed for training run
elapsed_time = datetime.datetime.now()

print(f"Total run time: {start_time - elapsed_time}")