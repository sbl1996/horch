from .operator import Operator

class Transpose(Operator):

  def __init__(self, parents, *args):
    super().__init__(parents, args)

  def forward(self, t):
    return t.T

  def backward(self, acc, t):
    return acc.T