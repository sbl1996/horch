import numpy as np

from .operator import Operator

class ReLU(Operator):

  def __init__(self, parents, *args):
    super().__init__(parents, args)

  def forward(self, x):
    return np.maximum(x, 0)

  def backward(self, acc, x):
    acc[x <= 0] = 0
    return acc
