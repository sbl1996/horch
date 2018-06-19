from horch.tensor import Tensor
from horch._op import *
from horch._parameter import Parameter
from horch._loss import *
from horch._create import *
import numpy as _np

def tensor(data: _np.ndarray, requires_grad: bool = False) -> Tensor:
  return Tensor(data, requires_grad=requires_grad)