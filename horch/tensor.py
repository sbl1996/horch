import numpy as np

class Tensor:

  def __init__(self, data, requires_grad=False):
    self.data = data
    self.grad = None
    self.requires_grad = requires_grad
    from ._operators import Leaf
    self.op = Leaf(self, requires_grad=requires_grad)

  def torch(self):
    import torch
    return torch.tensor(self.data.copy(), requires_grad=self.requires_grad)

  def size(self):
    return self.data.shape

  def backward(self):
    size = self.size()
    if size != ():
      raise RuntimeError("grad can be created only for scalar outputs")
    self.op._backward(np.array(1, dtype=self.data.dtype))

  def zero_grad(self):
    self.op.zero_grad()

  def item(self):
    return self.data.item()

  def __len__(self):
    return len(self.data)

  def __str__(self):
    return 'Tensor containing:\n%s' % self.data

  def __repr__(self):
    return self.__str__()

  def __add__(self, rt):
    from ._op import add
    return add(self, rt)

  def __radd__(self, lt):
    from ._op import add
    return add(lt, self)

  def __mul__(self, rt):
    from ._op import mul
    return mul(self, rt)    

  def __rmul__(self, lt):
    from ._op import mul
    return mul(lt, self)

  def __sub__(self, rt):
    from ._op import sub
    return sub(self, rt)

  def __rsub__(self, lt):
    from ._op import sub
    return sub(lt, self)

  def __neg__(self):
    from ._op import neg
    return neg(self)

  def __matmul__(self, rt):
    from ._op import matmul
    return matmul(self, rt)

  def __getitem__(self, *args):
    from ._op import getitem
    return getitem(self, *args)

  def sum(self, axis=0):
    from ._op import sum
    return sum(self, axis=axis)

  def relu(self):
    from ._op import relu
    return relu(self)

  def sigmoid(self):
    from ._op import sigmoid
    return sigmoid(self)

  def log(self):
    from ._op import log
    return log(self)  

  def abs(self):
    from ._op import abs
    return abs(self)

  def exp(self):
    from ._op import exp
    return exp(self)

  def maximum(self, rt):
    from ._op import maximum
    return maximum(self, rt)

  def mean(self, axis=None):
    from ._op import mean
    return mean(self, axis)

  def max(self, axis=None, keepdims=False):
    from ._op import max
    return max(self, axis, keepdims)

  def reshape(self, *shapes):
    from ._op import reshape
    return reshape(self, shapes)