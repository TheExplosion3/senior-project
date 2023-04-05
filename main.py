# tensorflow imports
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_datasets as tfds
# standard imports
import matplotlib.pyplot as mplpy
import numpy as np
import datetime
import time
import os
# file imports
from fn import configure_for_performance, get_optimizer, safeinput


# Variable initialization

# timer start, will show full elapsed time that the program took to execute.
start_time = datetime.datetime.now()

# dataset init
(train_ds, test_ds), metadata = tfds.load(
    'cifar100',
    split=['train[:90%]', 'test[90%:]'],
    with_info=True,
    as_supervised=True,
)

# other important stuff
num_classes = metadata.features['label'].num_classes
epochs = 10
batch_size = 8
steps_per_epoch = len(train_ds)//batch_size

# empty variables due to how python clears memory based upon scope
f, model, usr = None, None, None

print("Which version would you like to use, or would you like to recompile the model? Type a number, or n for new.")

while usr != "n" or "1" or "2":
  usr = safeinput('s')

# checks if the model has been created in the past, if not then it creates it on the spot.
# this network is convolutional, as shown by the first parts.
  if os.stat("model_save/model.h5").st_size == 0 and usr == "n":

    # augmentation helps to prevent against overfitting
    data_augmentation = keras.Sequential(
      [
        tf.keras.layers.RandomFlip("horizontal", input_shape=(32,32,3)),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
      ]
    )

    # main model section, first it augments data to make it just a bit harder for itself (will most likely eventually remove it when actually utilizing it.), then it goes through the convolutional section to extract features, then finally uses the hidden layers to determine its confidence in which label it is, and then returns the answer with softmax.
    model = tf.keras.Sequential([
        data_augmentation,
        tf.keras.layers.Conv2D(96, (3, 3), padding='same', activation='elu', input_shape=(32, 32, 3)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(96, (3, 3), padding='same', strides=2),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.Conv2D(192, (3, 3), padding='same', activation='swish'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(192, (3, 3), padding='same', strides=2),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(512, activation='leaky_relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(256, activation='leaky_relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(128, activation='leaky_relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation='softmax')
        ])
    break
  elif usr == "1" or "2":
    model = tf.keras.models.load_model(f'model_save/modelv{usr}.h5')
    model.summary()
    time.sleep(2)
    break
  else:
    print("Invalid input, please try again.")

# this will be important later
mdln = usr

print("Compiling Model...")

# learning schedule, decreases the rate at which learning occurs to prevent overfitting, slowly tapering off as it progresses through the epoch
lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
  0.001,
  decay_steps=steps_per_epoch*1000,
  decay_rate=1,
  staircase=False)

# compiles model, with NAdam optimizer as defined in the fn.py file, as well as defining loss via SCC.
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

# shows a graph of training statistics
mplpy.style.use('ggplot')
mplpy.plot(history.history['loss'], label = 'loss')
mplpy.plot(history.history['val_loss'], label='val loss')
mplpy.plot(history.history['val_accuracy'], label= "val_accuracy")
mplpy.title("Statistics")
mplpy.xlabel("Epochs")
mplpy.ylabel("Loss/Accuracy")
mplpy.legend()
mplpy.show()

### saving phase ###
# this code is self explanatory
print("Would you like to save this run? (y/n)")

while usr != "y" or usr != "n":
  usr = safeinput('s')
  if usr == "y":
    print("Saving data...")
    if mdln == "n":
      model.save('model_save/model.h5')
    elif mdln == "1":
      model.save('model_save/modelv1.h5')
    elif mdln == "2":
      model.save('model_save/modelv2.h5')
    print("Save complete.")
    break
  elif usr == "n":
    print("Skipping saving process...")
    break
  else:
    print("Invalid input, please try again.")
    safeinput('s')

# timer concludes, showing total time elapsed for training run
elapsed_time = datetime.datetime.now()

print(f"Total run time: {elapsed_time - start_time}")