import numpy as np

from .operator import Operator

class Neg(Operator):

  def __init__(self, parents, *args):
    super().__init__(parents, args)

  def forward(self, x):
    return -x

  def backward(self, acc, x):
    return -acc
