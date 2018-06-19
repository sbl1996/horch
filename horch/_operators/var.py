import numpy as np

from .operator import Operator

class Var(Operator):

  def __init__(self, parents, *args):
    super().__init__(parents, args)

  def forward(self, x, axis, keepdims):
    m = np.var(x, axis=axis, keepdims=keepdims, ddof=1)
    if not isinstance(m, np.ndarray):
      m = np.array(m)
    return m

  def backward(self, acc, x, axis, keepdims):
    if axis is not None:
      if not keepdims:
        acc = np.expand_dims(acc, axis)
      m = np.mean(x, axis=axis, keepdims=True)
      n = x.shape[axis]
    else:
      m = np.mean(x)
      n = x.size
    return (x - m) * 2 / (n - 1) * acc