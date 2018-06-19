import numpy as np

from .operator import Operator

class Max(Operator):

  def __init__(self, parents, *args):
    super().__init__(parents, args)

  def forward(self, a, axis, keepdims):
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
      if isinstance(axis, int):
        axis = (axis,)
      n = len(axis)
      for ax in axis:
        a = np.moveaxis(a, ax, -1)
      shape = a.shape
      shape1 = a.shape[:-n]
      shape2 = a.shape[-n:]
      size1 = np.prod(shape1)
      size2 = np.prod(shape2)
      a = a.reshape(-1, size2)
      ind = np.argmax(a, axis=-1)
      grad = np.zeros((size1, size2))
      grad[np.arange(size1), ind] = acc.reshape(-1)
      grad = grad.reshape(shape)
      for ax in reversed(axis):
        grad = np.moveaxis(grad, -1, ax)
    return grad