import numpy as _np
import horch as _H

def uniform(low=0.0, high=1.0, size=(), requires_grad=False):
  data = _np.random.uniform(low, high, size)
  return _H.tensor(data, requires_grad=requires_grad)

def rand(*shapes, requires_grad=False):
  return uniform(size=shapes, requires_grad=requires_grad)

def normal(loc=0, scale=1, size=(), requires_grad=False):
  data = _np.random.normal(loc, scale, size)
  return _H.tensor(data, requires_grad=requires_grad)

def randn(*shapes, requires_grad=False):
  return normal(size=shapes, requires_grad=requires_grad)