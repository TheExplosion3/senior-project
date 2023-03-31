import tensorflow as tf

def get_optimizer(lr_schedule):
  return tf.keras.optimizers.experimental.Nadam(lr_schedule)

def configure_for_performance(ds):
  ds = ds.cache()
  ds = ds.shuffle(buffer_size=1000)
  ds = ds.batch(5)
  ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
  return ds

def comma_addremove(closer, f):
  f = open("storage.json", "w")
  lines = f.read()
  if [-1] == ",":
    f.write(lines[:-1])
  else:
    f.write(lines + ",")
  if closer is True:
    f.close()
  else:
    return f

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