# tensorflow imports
import tensorflow as tf
# standard imports
import matplotlib.pyplot as mplpy
import numpy as np
import PIL
import PIL.Image

model = tf.keras.models.load_model('model_save/model.h5')

print("Select an image, in the format of /x/y/z.png")
usr = input()

image = PIL.Image.open(usr)
tf_payload = tf.keras.preprocessing.image.load_img(usr)

arr = model.predict(tf_payload, verbose=2)
arridx = None

high = 0
current = 0
highdx = 0
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