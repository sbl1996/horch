from ..tensor import Tensor

class Operator:

  def __init__(self, parents, args, requires_grad=False):
    self.parents = parents
    self.args = args
    self.requires_grad = requires_grad
    if self.parents:
      data = [ p.tensor.data for p in parents ]
      result = self.forward(*data, *args)
      self.tensor = Tensor(result)
      self.tensor.op = self
    else: # Leaf
      self.tensor = None

  def forward(self):
    pass

  def _backward(self, acc):
    data = [ p.tensor.data for p in self.parents ]
    grads = self.backward(acc, *data, *self.args)
    if not isinstance(grads, tuple):
      grads = (grads,)
    if self.parents:
      for p, grad in zip(self.parents, grads):
        p._backward(grad)

  def backward(self, acc):
    pass

  def zero_grad(self):
    self.tensor.grad = None
    for p in self.parents:
      p.zero_grad()
