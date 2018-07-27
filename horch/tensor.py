from typing import Tuple, Any

import numpy as np


class Tensor(object):

  def __init__(self, data: np.ndarray, requires_grad: bool = False) -> None:
    super().__init__()
    self.data = data
    self.grad = None
    self.requires_grad = requires_grad
    self.op = Leaf(self, requires_grad=requires_grad)

  def torch(self):
    import torch
    return torch.tensor(self.data.copy(), requires_grad=self.requires_grad)

  def size(self) -> Tuple[int, ...]:
    return self.data.shape

  def backward(self) -> None:
    if self.data.size != 1:
      raise RuntimeError("grad can be created only for scalar outputs")
    self.op._backward(np.array(1, dtype=self.data.dtype), root=True)

  def zero_grad(self) -> None:
    self.op.zero_grad()

  @property
  def shape(self) -> Tuple[int, ...]:
    return self.data.shape

  def item(self) -> Any:
    return self.data.item()

  def __truediv__(self, rt):
    return _op.div(self, rt)

  def __rtruediv__(self, lt):
    return _op.div(lt, self)

  def __len__(self):
    return len(self.data)

  def __str__(self):
    return 'Tensor containing:\n%s' % self.data

  def __repr__(self):
    return self.__str__()

  def __add__(self, rt):
    return _op.add(self, rt)

  def __radd__(self, lt):
    return _op.add(lt, self)

  def __mul__(self, rt):
    return _op.mul(self, rt)    

  def __rmul__(self, lt):
    return _op.mul(lt, self)

  def __sub__(self, rt):
    return _op.sub(self, rt)

  def __rsub__(self, lt):
    return _op.sub(lt, self)

  def __neg__(self):
    return _op.neg(self)

  def __matmul__(self, rt):
    return _op.matmul(self, rt)

  def __getitem__(self, *args):
    return _op.getitem(self, *args)

  def log(self):
    return _op.log(self)  

  def abs(self):
    return _op.abs(self)

  def exp(self):
    return _op.exp(self)

  def maximum(self, rt):
    return _op.maximum(self, rt)

  def mean(self, axis=None, keepdims=False):
    return _op.mean(self, axis, keepdims)

  def max(self, axis=None, keepdims=False):
    return _op.max(self, axis, keepdims)

  def reshape(self, *args):
    return _op.reshape(self, *args)

  def view(self, *args):
    return self.reshape(*args)

  def sqrt(self):
    return _op.sqrt(self)

  def std(self, axis=None, keepdims=False):
    return _op.std(self, axis=axis, keepdims=keepdims)

  def sum(self, axis=None, keepdims=False):
    return _op.sum(self, axis=axis, keepdims=keepdims)

  @property
  def T(self):
    return _op.transpose(self)

  def var(self, axis=None, keepdims=False):
    return _op.var(self, axis=axis, keepdims=keepdims)

import horch._op as _op
from horch._operators import Leaf