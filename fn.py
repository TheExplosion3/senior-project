class checkpointsave:
  def __init__(self, data, iteration, times_run):
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