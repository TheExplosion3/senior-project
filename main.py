# tensorflow imports
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_datasets as tfds
# standard imports
import matplotlib.pyplot as mplpy
import json
import os
# file imports
from fn import comma_addremove, configure_for_performance, safeinput


# Variable initialization
(train_ds, test_ds), metadata = tfds.load(
    'cifar100',
    split=['train[:80%]', 'test[90%:]'],
    with_info=True,
    as_supervised=True,
)

num_classes = metadata.features['label'].num_classes
epochs = 0

# user input grabber, with exception handling
print("How many epochs?")
safeinput(epochs, "i")

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

resize_and_rescale = tf.keras.Sequential([
  tf.keras.layers.Resizing(180, 180),
  tf.keras.layers.Rescaling(1./255)
])

# checks if the model has been created in the past, if not then it grabs it elsewhere
# amt of nodes on input layer: 4
# amt of nodes per hidden layer: 37/38
# amt of nodes on output layer: 100
if os.stat("model_save/model.h5").st_size == 0:
  model = tf.keras.Sequential([
      resize_and_rescale,
      data_augmentation,
      tf.keras.layers.Conv2D(4, (3, 3), activation='elu', input_shape=(32, 32, 3)),
      tf.keras.layers.MaxPooling2D((2, 2)),
      tf.keras.layers.Conv2D(32, (3, 3), activation='elu'),
      tf.keras.layers.MaxPooling2D((2, 2)),
      tf.keras.layers.Conv2D(32, (3, 3), activation='elu'),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dropout(0.5),
      tf.keras.layers.Dense(37, activation='leaky_relu'),
      tf.keras.layers.Dense(38, activation='leaky_relu'),
      tf.keras.layers.Dense(num_classes)
  ])
else:
  model = tf.keras.models.load_model('model_save/model.h5')
  f = comma_addremove(False, f)
  json.load(f)
  f.close()

print("Compiling Model...")

model.compile(optimizer='nadam',
loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

print("Model Compiled.")

if os.stat("model_save/model.h5").st_size != 0:
  print("Current Model Statistics:")
  model.summmary()

print(f"Training beginning, running {epochs} epoch(s)")


image, label = next(iter(train_ds))
get_label_name = metadata.features['label'].int2str
_ = mplpy.imshow(image)
_ = mplpy.title(get_label_name(label))

train_ds = configure_for_performance(train_ds)
test_ds = configure_for_performance(test_ds)
### training phase ###
while True:

  model.fit(image, label, epochs)

  if image.next() != None:
    image, label = next(image), next(label)

    get_label_name = metadata.features['label'].int2str
    _ = mplpy.imshow(image)
    _ = mplpy.title(get_label_name(label))
  else:
    break

loss, acc = model.evaluate(test_ds)
print("Accuracy: ", acc)
# Adds the correct amount of iterations to the count, amount of times run increases by 1 as well.
print("Training ended. Creating savepoint.")

### saving phase ###
# save method, first step announces it is opening file, and saving
print("Saving data...")
model.save('model_save/model.h5')
print("Save complete.")