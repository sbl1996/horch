import numpy as np

from .operator import Operator

class Reshape(Operator):

  def __init__(self, parents, *args):
    super(Exp, self).__init__(parents, args)

  def forward(self, x, shape):
    return x.reshape(shape)

  def backward(self, acc, x, shape):
    return acc.reshape(x.shape)