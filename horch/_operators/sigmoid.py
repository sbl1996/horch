import numpy as np
from scipy.special import expit

from .operator import Operator

class Sigmoid(Operator):

  def __init__(self, parents, *args):
    super().__init__(parents, args)

  def forward(self, x):
    return expit(x)

  def backward(self, acc, x):
    # d = self.tensor.data
    d = expit(x)
    return acc * (1 - d) * d