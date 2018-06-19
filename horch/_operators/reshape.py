import numpy as np

from .operator import Operator

class Reshape(Operator):

  def __init__(self, parents, *args):    
    super().__init__(parents, args)

  def forward(self, x, *args):
    return x.reshape(*args)

  def backward(self, acc, x, *args):
    return acc.reshape(x.shape)