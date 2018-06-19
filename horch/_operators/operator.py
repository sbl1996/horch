from horch.tensor import Tensor

class Operator(object):

  def __init__(self, parents, args, requires_grad=False):
    super().__init__()
    self.parents = parents
    self.args = args
    self.requires_grad = requires_grad
    if self.parents:
      data = map(get_tensor_data, self.parents)
      result = self.forward(*data, *args)
      self.tensor = Tensor(result)
      self.tensor.op = self
    else: # Leaf
      self.tensor = None

  def _backward(self, acc):
    data = map(get_tensor_data, self.parents)
    grads = self.backward(acc, *data, *self.args)
    if not isinstance(grads, tuple):
      grads = (grads,)
    if self.parents:
      for p, grad in zip(self.parents, grads):
        if p is not None:
          p._backward(grad)

  def zero_grad(self):
    self.tensor.grad = None
    for p in self.parents:
      p.zero_grad()

def get_tensor_data(p):
  if p is None:
    return None
  return p.tensor.data