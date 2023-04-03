# tensorflow imports
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_datasets as tfds
import keras_tuner as kt
# standard imports
import matplotlib.pyplot as mplpy
import numpy as np
import datetime
import time
import json
import os
# file imports
from fn import configure_for_performance, get_optimizer, safeinput


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

print("Which version would you like to use, or would you like to recompile the model? Type a number, or n for new.")
usr = None
while usr != "n" or "1" or "2":
  usr = safeinput('s')
# checks if the model has been created in the past, if not then it creates it on the spot.
# this network is convolutional, as shown by the first parts.
# stats:
# amt of nodes on input layer: 4
# amt of nodes per hidden layer: 37/38
# amt of nodes on output layer: 100
  if os.stat("model_save/model.h5").st_size == 0 and usr == "n":

    data_augmentation = keras.Sequential(
      [
        tf.keras.layers.RandomFlip("horizontal", input_shape=(32,32,3)),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
      ]
    ) 

    model = tf.keras.Sequential([
        data_augmentation,
        tf.keras.layers.Input(shape=(32, 32, 3)),
        tf.keras.layers.Conv2D(4, (3, 3), padding='same', activation='elu',),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='elu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='elu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(37, activation='leaky_relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(38, activation='leaky_relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation='sigmoid')
        ])
    break
  elif usr == "1" or "2":
    model = tf.keras.models.load_model(f'model_save/modelv{usr}.h5')
    model.summary()
    break
  else:
    print("Invalid input, please try again.")


print("Compiling Model...")

lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
  0.001,
  decay_steps=steps_per_epoch*1000,
  decay_rate=1,
  staircase=False)

model.compile(optimizer=get_optimizer(lr_schedule),
loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy', 'mae'])

print("Model Compiled.")

print(f"Training beginning, running {epochs} epoch(s)")

image, label = next(iter(train_ds))
train_ds = configure_for_performance(train_ds)
test_ds = configure_for_performance(test_ds)

# training loop. this code starts with an image, then it takes a label as well, and checks whether or not the image is that label. after looping through, it shows avg. accuracy, then waits 5 seconds, then proceeds.

history = model.fit(x=train_ds, epochs=epochs, steps_per_epoch=steps_per_epoch, validation_data=[test_ds])
loss, acc, mse=model.evaluate(train_ds, verbose=2)
print(f'\naverage accuracy : {(np.mean(acc)):5.2f}\n')

mplpy.style.use('ggplot')
mplpy.plot(history.history['loss'], label = 'loss')
mplpy.plot(history.history['val_loss'], label='val loss')
mplpy.title("Loss vs Val_Loss")
mplpy.xlabel("Epochs")
mplpy.ylabel("Loss")
mplpy.legend()
mplpy.show()

### saving phase ###
# this code is self explanatory
print("Would you like to save this run? (y/n)")
usr = safeinput('s')

while usr != "y" or usr != "n":
  if usr == "y":
    print("Saving data...")
    model.save('model_save/model.h5')
    print("Save complete.")
    break
  elif usr == "n":
    print("Skipping saving process...")
    break
  else:
    "Invalid input, please try again."
    safeinput('s')

# timer concludes, showing total time elapsed for training run
elapsed_time = datetime.datetime.now()

print(f"Total run time: {elapsed_time - start_time}")