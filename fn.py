import tensorflow as tf
import tensorflow.keras as keras

# quite literally just makes the model's optimizer, gives it a learning schedule defined in the passthrough, then returns it.
def get_optimizer(lr_schedule):
  return tf.keras.optimizers.experimental.Nadam(lr_schedule)

# configures the dataset for performance
def configure_for_performance(ds):
  ds = ds.cache()
  ds = ds.shuffle(buffer_size=1000)
  ds = ds.batch(8)
  ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
  return ds

# a function that makes sure all user input is safe, and is not of an invalid type. the code will handle it by erroring out.
def safeinput(var, vartype):
  while True:
    try:
      if vartype == "s":
        var = input()
      elif vartype == "i":
        var = int(input())
      elif vartype == "f":
        var = float(input())
      elif vartype == "b":
        var = input()
        var = bool(var[0].upper() + var[1:])
      elif vartype == "c":
        var = complex(input())
      else:
        raise TypeError
      break
    except TypeError:
        print("Invalid input. Try again.")
    except Exception:
        print("Unknown exception occurred.")
  return var

# a function that makes sure all user input is safe, and is not of an invalid type. if invalid, the process will error out. this type does not require an input of a variable, for use in defining another variable.
def safeinput(vartype):
  temp = None
  while True:
    try:
      if vartype == "s":
        temp = input()
      elif vartype == "i":
        temp = int(input())
      elif vartype == "f":
        temp = float(input())
      elif vartype == "b":
        temp = input()
        temp = bool(temp[0].upper() + temp[1:])
      elif vartype == "c":
        temp = complex(input())
      else:
        raise TypeError
      break
    except TypeError:
        print("Invalid input. Try again.")
    except Exception:
        print("Unknown exception occurred.")
  return temp