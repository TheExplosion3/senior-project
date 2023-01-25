# tensorflow imports
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Model
import tensorflow_datasets as tfds
# standard imports
import matplotlib as mpl
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



# Variable initialization along with object
savept = checkpointsave(None, 0, 0)

iterator = 0
uinput = 0

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])

print("Current Model Statistics:")
model.summmary()
print("Training Beginning...")
while iterator != uinput:
  iterator+=1
  pass
# Adds the correct amount of iterations to the count, amount of times run increases by 1 as well.
checkpointsave.runner += 1
checkpointsave.iteration = checkpointsave.iteration + iterator
print("Training ended.")

# safe method, first step announces it is opening file, and saving
print("Saving Data...")
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
      break
    except:
      print("Max number of attempts reached, exiting without save.")
# Announces the process complete, closes the file, program terminates.
finally:
  print("Saving process complete.")
  f.close()