from io import StringIO
import numpy as np
import sys
import importlib
import decorator
import functools

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

def module_exists(module_name):
  spec = importlib.util.find_spec(module_name)
  return spec is None

# def torch_test(func):
#   def wrapper(func, *args, **kwargs):
#     if module_exists("torch"):
#       import torch
#       import torch.nn.functional as F
#       return func(*args, **kwargs)
#     else:
#       return None
#   return decorator.decorator(wrapper, func)

# def torch_test(func):
#   @functools.wraps(func)
#   def wrapper(*args, **kwargs):
#     if module_exists("torch"):
#       import torch
#       import torch.nn.functional as F
#       return func(*args, **kwargs)
#     else:
#       pass
#   return wrapper