import numpy as np

from .operator import Operator

class Std(Operator):

  def __init__(self, parents, *args):
    super().__init__(parents, args)

  def forward(self, x, axis, keepdims):
    m = np.std(x, axis=axis, keepdims=keepdims, ddof=1)
    if not isinstance(m, np.ndarray):
      m = np.array(m)
    return m

  def backward(self, acc, x, axis, keepdims):
    s = self.tensor.data.copy()
    if axis is not None:
      if not keepdims:
        acc = np.expand_dims(acc, axis)
        s = np.expand_dims(s, axis)
      m = np.mean(x, axis=axis, keepdims=True)
      n = x.shape[axis]
    else:
      m = np.mean(x)
      n = x.size
    return (x - m) / (n - 1) / s * acc