def count(f, it):
  n = 0
  for x in it:
    if f(x):
      n += 1
  return n

def is_not_none(x):
  return x is not None

def not_none(it):
  for x in it:
    if x is not None:
      yield x