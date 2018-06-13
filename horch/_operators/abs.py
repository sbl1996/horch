import numpy as np

from .operator import Operator

class Abs(Operator):

  def __init__(self, parents, *args):
    super(Abs, self).__init__(parents, args)

  def forward(self, x):
    return np.abs(x)

  def backward(self, acc, x):
    grad = np.zeros(x.shape)
    grad[x > 0] = 1
    grad[x < 0] = -1
    return acc * grad
