import numpy as np

from .operator import Operator
from ..tensor import Tensor

class GetItem(Operator):

  def __init__(self, parents, *args):
    super().__init__(parents, args)

  def _transform_indices(self, ind):
    if isinstance(ind, tuple):
      ind = list(ind)
      for i in range(len(ind)):
        if isinstance(ind[i], Tensor):
          ind[i] = ind[i].data
      ind = tuple(ind)
    elif isinstance(ind, Tensor):
      ind = ind.data
    return ind

  def forward(self, x, ind):
    ind = self._transform_indices(ind)
    res = x[ind]
    if not isinstance(res, np.ndarray):
      res = np.array(res)
    return res

  def backward(self, acc, x, ind):
    ind = self._transform_indices(ind)
    grad = np.zeros_like(x)
    grad[ind] = acc
    return grad