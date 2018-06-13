import numpy as np

from .operator import Operator

class Max(Operator):

  def __init__(self, parents, *args):
    super(Max, self).__init__(parents, args)

  def forward(self, a, axis=None, keepdims=None):
    m = np.max(a, axis=axis, keepdims=keepdims)
    if not isinstance(m, np.ndarray):
      m = np.array(m)
    return m

  def backward(self, acc, a, axis, keepdims):
    if axis is None:
      ind = np.unravel_index(np.argmax(a, axis=None), a.shape)
      grad = np.zeros(a.shape)
      grad[ind] = acc
    else:
      ind = np.argmax(a, axis=axis)
      grad = np.zeros(a.shape)
      grad[ind, np.arange(a.ndim)] = acc
    return grad