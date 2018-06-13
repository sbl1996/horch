import numpy as np

from .operator import Operator

class MatMul(Operator):

  def __init__(self, parents, *args):
    super(MatMul, self).__init__(parents, args)

  def forward(self, l, r):
    return l @ r

  def backward(self, acc, l, r):
    lshape = l.shape
    rshape = r.shape
    if l.ndim == 1 and r.ndim == 1:
      acc = acc.reshape(1, 1)
    if l.ndim == 1:
      l = l[np.newaxis, :]
      acc = acc[np.newaxis, :]
    if r.ndim == 1:
      r = r[:, np.newaxis]
      acc = acc[:, np.newaxis]
    lgrad = (acc @ r.T).reshape(lshape)
    rgrad = (l.T @ acc).reshape(rshape)
    return lgrad, rgrad
