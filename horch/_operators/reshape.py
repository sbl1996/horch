import numpy as np

from .operator import Operator

class Reshape(Operator):

  def __init__(self, parents, *args):
    super(Reshape, self).__init__(parents, args)

  def forward(self, x, shapes):
    return x.reshape(*shapes)

  def backward(self, acc, x, shape):
    return acc.reshape(x.shape)