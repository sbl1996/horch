from .operator import Operator
from ._broadcast import broadcast_backward_2
class Add(Operator):

  def __init__(self, parents, *args):
    super().__init__(parents, args)

  def forward(self, l, r):
    return l + r

  def backward(self, acc, l, r):
    lgrad = acc
    rgrad = acc
    return broadcast_backward_2(l, r, lgrad, rgrad)