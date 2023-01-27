# tensorflow imports
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Model
import tensorflow_datasets as tfds
# standard imports
import matplotlib as mpl
import matplotlib.pyplot as mplpy
import json
import os
# file imports
from fn import checkpointsave, comma_addremove, configure_for_performance, safeinput

# Variable initialization along with object
(train_ds, val_ds, test_ds), metadata = tfds.load(
    'cifar-100',
    split=['train[:80%]', 'val[80%:90%]', 'test[90%:]'],
    with_info=True,
    as_supervised=True,
)

# set up known vars
iterator = 0
iterations = 0
saveattempts = 0

# user input grabber, with exception handling
print("Iterate for how many times? ")
safeinput(iterations, "i")
print("Amount of times to attempt saving? ")
safeinput(saveattempts, "i")
    
# empty variables due to how python clears memory based upon scope

f = None
model = None
savept = None

# checks if the model has been created in the past, if not then it grabs it elsewhere
if os.stat("storage.json").st_size == 0:
  model = tf.keras.Sequential([
      tf.keras.layers.Flatten(input_shape=(32, 32)),
      tf.keras.layers.Dense(128, activation='elu'),
      tf.keras.layer.Dropout(0.5),
      tf.keras.layers.Dense(128, activation='elu'),
      tf.keras.layer.Dropout(0.5),
      tf.keras.layers.Dense(64, activation='leaky_relu'),
      tf.keras.layer.Dropout(0.5),
      tf.keras.layers.Dense(32, activation='leaky_relu'),
      tf.keras.layer.Dropout(0.5),
      tf.keras.layers.Dense(10, activation='leaky_relu'),
      tf.keras.layer.Dropout(0.5),
      tf.keras.layers.Dense(5, activation ='softmax')
  ])
  savept = checkpointsave(None, 0, 0)
else:
  model = tf.keras.models.load_model('model.h5')
  f = comma_addremove(False, f)
  json.load(f)
  f.close()

print("Compiling Model...")

model.compile(optimizer='nadam',
loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

print("Model Compiled.")

print("Current Model Statistics:")
model.summmary()
print(f"Training beginning, running {iterations} time(s)")


image, label = next(iter(train_ds))
get_label_name = metadata.features['label'].int2str
_ = mplpy.imshow(image)
_ = mplpy.title(get_label_name(label))

train_ds = configure_for_performance(train_ds)
val_ds = configure_for_performance(val_ds)
test_ds = configure_for_performance(test_ds)

while iterator != iterations:

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
f = open("storage.json", "a")
# attempts to dump py object into a json object in the file
try:
  json.dump(savept, f, indent = 6)
# if it fails it excepts any exceptions and tries 4 more times, announcing exception and try attempt each time
except Exception:
  if saveattempts:
    for i in range(saveattempts - 1):
      try:
        print(f"Save failed, trying again.\nAttempt number: {i}\nException: {Exception}")
        json.dump(savept, f, indent = 6)
        break
      except:
        pass
      print("Max number of attempts reached, exiting without save.")
else:
  print("Save successful.")
# Announces the process complete, closes the file, program terminates.
finally:
  f.write(",")
  print("Saving process complete.")
  f.close()