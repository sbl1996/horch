import numpy as np

from .operator import Operator

class Mean(Operator):

  def __init__(self, parents, *args):
    super().__init__(parents, args)

  def forward(self, x, axis, keepdims):
    m = np.mean(x, axis=axis, keepdims=keepdims)
    if not isinstance(m, np.ndarray):
      m = np.array(m)
    return m

  def backward(self, acc, x, axis, keepdims):
    if axis is not None:
      if not keepdims:
        acc = np.expand_dims(acc, axis)
      return acc * np.ones_like(x) / x.shape[axis]
    return acc * np.ones_like(x) / x.size