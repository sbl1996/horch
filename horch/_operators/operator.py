from typing import List, Generator, Optional, Any
import numpy as _np

from horch.tensor import Tensor
from horch._utils import count, is_not_none, not_none

class Operator(object):

  def __init__(self, parents: List[Optional['Operator']], args) -> None:
    super().__init__()
    self.parents = parents
    self.args = args
    self.requires_grad = any(p.requires_grad for p in self.parents)

    self.out_degree = 0
    self.acc_grad = 0

    if self.parents: # not leaf
      for p in not_none(self.parents):
        p.out_degree += 1
      result = self.forward(*self._get_parents_data(), *args)
      self.tensor = Tensor(result)
      self.tensor.op = self

  def _get_parents_data(self) -> Generator[Optional[_np.ndarray], Any, None]:
    for p in self.parents:
      if p is not None:
        yield p.tensor.data
      else:
        yield None

  def _backward(self, acc: _np.ndarray, root: bool = False) -> None:
    self.out_degree -= 1
    self.acc_grad += acc
    if self.out_degree != 0:
      if not root:
        return
    grads = self.backward(self.acc_grad, *self._get_parents_data(), *self.args)
    self.acc_grad = None
    if not isinstance(grads, tuple):
      grads = (grads,)
    for p, grad in zip(self.parents, grads):
      if p is not None:
        p._backward(grad)

  def zero_grad(self) -> None:
    self.tensor.grad = None
    for p in self.parents:
      p.zero_grad()