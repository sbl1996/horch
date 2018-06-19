import numpy as np

from .operator import Operator

class Log(Operator):

  def __init__(self, parents, *args):
    super().__init__(parents, args)

  def forward(self, x):
    return np.log(x)

  def backward(self, acc, x):
    return acc / x
