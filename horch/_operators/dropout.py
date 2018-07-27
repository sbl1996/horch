import numpy as np

from .operator import Operator

class Dropout(Operator):

  def __init__(self, parents, *args):
    super().__init__(parents, args)

  def forward(self, x, p, training):
    if training:
      mask = (np.random.uniform(size=x.shape) > p) / (1 - p)
      self.mask = mask
      x = x * mask
    return x

  def backward(self, acc, x, p, training):
    return acc * self.mask