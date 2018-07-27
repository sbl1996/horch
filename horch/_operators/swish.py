import numpy as np
from scipy.special import expit

from .operator import Operator

class Swish(Operator):

  def __init__(self, parents, *args):
    super().__init__(parents, args)

  def forward(self, x, beta):
    return x * expit(beta * x)

  def backward(self, acc, x, beta):
    y = self.tensor.data
    by = beta * y
    s = expit(beta * x)
    g = by + s * (1 - by)
    return g