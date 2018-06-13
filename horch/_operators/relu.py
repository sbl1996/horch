import numpy as np

from .operator import Operator

class ReLU(Operator):

  def __init__(self, parents, *args):
    super(ReLU, self).__init__(parents, args)

  def forward(self, x):
    return np.maximum(x, 0)

  def backward(self, acc, x):
    grad = np.zeros(x.shape)
    grad[x > 0] = 1
    return acc * grad
