from .operator import Operator

class Leaf(Operator):

  def __init__(self, tensor, requires_grad):
    super(Leaf, self).__init__([], [], requires_grad)
    self.tensor = tensor

  def backward(self, acc):
    if self.requires_grad:
      if self.tensor.grad is None:
        self.tensor.grad = acc
      else:
        self.tensor.grad += acc

