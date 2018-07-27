from .operator import Operator

class Leaf(Operator):

  def __init__(self, tensor, requires_grad):
    super().__init__([], [])
    self.requires_grad = requires_grad
    self.tensor = tensor

  def _backward(self, acc, root=False):
    if self.requires_grad:
      if self.tensor.grad is None:
        self.tensor.grad = acc
      else:
        self.tensor.grad += acc

