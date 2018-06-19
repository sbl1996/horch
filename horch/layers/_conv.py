import numpy as np

import horch as H
from horch import Parameter

from ._module import Module

def kaiming_normal(shape):
  n_in = shape[1]
  return np.random.randn(*shape) / np.sqrt(n_in / 2)

def detuple(x, repeat = 2):
  if isinstance(x, tuple):
    return x
  return tuple([x] * repeat)

class Conv2d(Module):

  def __init__(self, n_in, n_out, kernel_size, stride=1, padding=0, bias=True):
    super().__init__()
    self.n_in = n_in
    self.n_out = n_out
    self.kernel_size = detuple(kernel_size)
    self.stride = detuple(stride)
    self.padding = detuple(padding)
    kH, kW = self.kernel_size
    self.weight = Parameter(kaiming_normal((n_out, n_in, kH, kW)))
    if bias:
      self.bias = Parameter(np.random.randn(n_out) / np.sqrt(n_out / 2))
    else:
      self.bias = None

  def forward(self, x):
    x = H.conv2d(x, self.weight, self.bias, stride=self.stride, padding=self.padding)
    return x