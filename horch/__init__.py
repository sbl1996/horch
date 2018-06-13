from .tensor import Tensor
from ._op import *
from ._parameter import Parameter
from ._loss import *

def tensor(data, requires_grad=False):
  return Tensor(data, requires_grad=requires_grad)