from io import StringIO
import numpy as np
import sys

sys.path.append("D:\\MLCode\\horch")

import horch

def dm(s, **kwargs):
  if isinstance(s, list):
      return np.array(s)
  f = StringIO(s.replace(';', '\n'))
  return np.genfromtxt(f, **kwargs)

# def gtensor(data):
#   horch.tensor(data, requires_grad=True)

def gtensor(data):
  return horch.tensor(data, requires_grad=True)
