import numpy as np

from .operator import Operator

class Exp(Operator):

  def __init__(self, parents, *args):
    super(Exp, self).__init__(parents, args)

  def forward(self, x):
    return np.exp(x)

  def backward(self, acc, x):
    return acc * np.exp(x)