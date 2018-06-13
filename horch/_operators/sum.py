import numpy as np

from .operator import Operator

class Sum(Operator):

  def __init__(self, parents, *args):
    super(Sum, self).__init__(parents, args)

  def forward(self, x, axis):
    return x.sum(axis=axis, keepdims=True).squeeze()

  def backward(self, acc, x, axis):
    if axis is not None:
      acc = np.expand_dims(acc, axis)
    return acc * np.ones_like(x)
