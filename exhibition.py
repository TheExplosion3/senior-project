# tensorflow imports
import tensorflow as tf
# standard imports
import matplotlib.pyplot as mplpy
import numpy as np
import PIL
import PIL.Image
# file imports
from fn import safeinput

# user input for loading model version
print ("Which model would you like to use? Select a version, or hit enter to get through individually")
usr = safeinput('s')
model = None
if usr is "":
    model = tf.keras.models.load_model('model_save/model.h5')
else:
    model = tf.keras.models.load_model(f'model_save/modelv{usr}.h5')

# user input for grabbing the image location
print("Select an image, in the format of /x/y/z.png")
usr = safeinput('s')

# variables and stuff
arridx = None
high = 0
current = 0
highdx = 0
image = PIL.Image.open(usr)
tf_payload = tf.keras.preprocessing.image.load_img(usr)

# actual prediction of what the image is
arr = model.predict(tf_payload, verbose=2)

# sorts for the highest prediction, and its index
for i in arr:
    if i < high:
        high = i
        highdx = current
        arridx = None
    elif i == high:
        if arridx is None:
            arridx = np.arr([highdx, current])
        else:
            np.append(arridx, current)
    current += 1

# displays the image, the network's prediction, and its confidence.
if arridx is None:
    mplpy.figure(figsize=(10, 10))
    mplpy.imshow(image.numpy().astype("uint8"))
    mplpy.title(image.filename + " Network Prediction")
    mplpy.suptitle(f"Network states: {arr[highdx]}, confidence: {arr[high]}")
    mplpy.axis("off")
else:
    mplpy.figure(figsize=(10, 10))
    mplpy.imshow(image.numpy().astype("uint8"))
    mplpy.title(image.filename + " Network Prediction")
    mplpy.suptitle(f"Network states: Multiple answers, as follows: {arridx}, confidence: {high}")
    mplpy.axis("off")