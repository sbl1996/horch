from horch.tensor import Tensor
from horch._op import *
from horch._parameter import Parameter
from horch._loss import *

def tensor(data, requires_grad=False):
  return Tensor(data, requires_grad=requires_grad)