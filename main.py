# tensorflow imports
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Model
import tensorflow_datasets as tfds
# standard imports
import matplotlib as mpl
import matplotlib.pyplot as mplpy
import numpy as np
import json
import PIL
import PIL.Image
import os

# Object with iteration of training, and times training has been run, along with the weight data from the model
class checkpointsave:
  def __init(self, data, iteration, times_run):
    self.data = data
    self.iteration = iteration
    self.times_run = times_run

def configure_for_performance(ds):
  ds = ds.cache()
  ds = ds.shuffle(buffer_size=1000)
  ds = ds.batch(5)
  ds = ds.prefetch(buffer_size="AUTOTUNE")
  return ds

def comma_addremove(closer, f):
  f = open("storage.json", "w")
  lines = f.read()
  if [-1] is ",":
    f.write(lines[:-1])
  else:
    f.write(lines + ",")
  if closer is True:
    f.close()
  else:
    return f

# Variable initialization along with object
(train_ds, val_ds, test_ds), metadata = tfds.load(
    'cifar-100',
    split=['train[:80%]', 'train[80%:90%]', 'train[90%:]', 'test[:50%]'],
    with_info=True,
    as_supervised=True,
)

iterator = 0
print("Iterate for how many times? ")

while True:
  try:
    uinput = int(input())
    break
  except TypeError:
      print("Invalid input. Try again.")
  except Exception:
      print("Unknown exception occurred.")
# empty variables due to how python clears memory based upon scope

f = None
model = None
savept = None

if os.stat("storage.json").st_size == 0:
  model = tf.keras.Sequential([
      tf.keras.layers.Flatten(input_shape=(28, 28)),
      tf.keras.layers.Dense(128, activation='elu'),
      tf.keras.layers.Dense(128, activation='elu'),
      tf.keras.layers.Dense(64, activation='leaky_relu'),
      tf.keras.layers.Dense(32, activation='leaky_relu'),
      tf.keras.layers.Dense(10, activation='softmax')
  ])
  savept = checkpointsave(None, 0, 0)
else:
  model = tf.keras.models.load_model('model.h5')
  f = comma_addremove(False, f)
  json.load(f)
  f.close()

print("Compiling Model...")

model.compile(optimizer='adam',
loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

print("Model Compiled.")

print("Current Model Statistics:")
model.summmary()
print(f"Training beginning, running {uinput} time(s)")

image, label = next(iter(train_ds))
get_label_name = metadata.features['label'].int2str
_ = mplpy.imshow(image)
_ = mplpy.title(get_label_name(label))

train_ds = configure_for_performance(train_ds)
val_ds = configure_for_performance(val_ds)
test_ds = configure_for_performance(test_ds)

while iterator != uinput:

  model.fit(image, label, epochs=10)

  iterator+=1

  if image.next() != None:
    image, label = next(image), next(label)

    get_label_name = metadata.features['label'].int2str
  _ = mplpy.imshow(image)
  _ = mplpy.title(get_label_name(label))
# Adds the correct amount of iterations to the count, amount of times run increases by 1 as well.
checkpointsave.runner += 1
checkpointsave.iteration = checkpointsave.iteration + iterator
print("Training ended. Creating savepoint.")


# save method, first step announces it is opening file, and saving
print("Saving data...")
model.save('model.h5')
checkpointsave.savept = "model.h5"
f = open("storage.json", "w")
# attempts to dump py object into a json object in the file
try:
  json.dump(savept, f, indent = 6)
# if it fails it excepts any exceptions and tries 4 more times, announcing exception and try attempt each time
except Exception:
  for i in range(4):
    try:
      print(f"Save failed, trying again.\nAttempt number: {i}\nException: {Exception}")
      json.dump(savept, f, indent = 6)
      f.write(',')
      break
    except:
      print("Max number of attempts reached, exiting without save.")
# Announces the process complete, closes the file, program terminates.
finally:
  print("Saving process complete.")
  f.close()