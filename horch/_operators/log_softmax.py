import numpy as np

from .operator import Operator

class LogSoftmax(Operator):

  def __init__(self, parents, *args):
    super().__init__(parents, args)

  def forward(self, x, axis):
    m = x.max(axis=axis, keepdims=True)
    x = x - m
    return x / x.sum(axis=axis, keepdims=True)

  def backward(self, acc, x, axis):
    d = self.tensor.data
    return (1 - d) * acc