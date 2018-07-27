import numpy as np

from .operator import Operator
from ._broadcast import broadcast_backward_2

class Maximum(Operator):

  def __init__(self, parents, *args):
    super().__init__(parents, args)

  def forward(self, l, r):
    return np.maximum(l, r)

  def backward(self, acc, l, r):
    lgrad = acc * (l > r)
    rgrad = acc * (l <= r)
    return broadcast_backward_2(l, r, lgrad, rgrad)