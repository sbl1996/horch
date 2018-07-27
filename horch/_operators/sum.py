import numpy as np

from .operator import Operator
from .util import expand_dims

class Sum(Operator):

  def __init__(self, parents, *args):
    super().__init__(parents, args)

  def forward(self, x, axis, keepdims):
    m = np.sum(x, axis=axis, keepdims=keepdims)
    if not isinstance(m, np.ndarray):
      m = np.array(m)
    return m

  def backward(self, acc, x, axis, keepdims):
    if axis is not None and not keepdims:
      acc = expand_dims(acc, axis)
    return acc * np.ones_like(x)
