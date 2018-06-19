import numpy as np

from .operator import Operator

class Maximum(Operator):

  def __init__(self, parents, *args):
    super().__init__(parents, args)

  def forward(self, l, r):
    return np.maximum(l, r)

  def backward(self, acc, l, r):
    lgrad = acc * (l > r)
    rgrad = acc * (l <= r)
    if l.shape != r.shape:
      if l.ndim == r.ndim:
        if l.shape[0] == 1: # (1,3) (2,3)
          lgrad = lgrad.sum(axis=0).reshape(l.shape)
        else:               # (2,3) (1,3)
          rgrad = rgrad.sum(axis=0).reshape(r.shape)
      elif l.ndim < r.ndim:
        if l.ndim == 0: # () (2,3)
          lgrad = lgrad.sum(keepdims=True).squeeze()
        else:           # (3,) (2,3)
          lgrad = lgrad.sum(axis=0).reshape(l.shape)
      else:
        if r.ndim == 0: # (2,3) ()
          rgrad = rgrad.sum(keepdims=True).squeeze()
        else:           # (2,3) (3,)
          rgrad = rgrad.sum(axis=0).reshape(r.shape)
    return lgrad, rgrad
