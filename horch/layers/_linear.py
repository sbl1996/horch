import numpy as np
from ._module import Module
from .._parameter import Parameter

def kaiming_normal(n_in, n_out):
  return np.random.randn(n_in, n_out) / np.sqrt(n_in / 2)

class Linear(Module):

  def __init__(self, n_in, n_out, bias=True):
    super(Linear, self).__init__()
    self.n_in = n_in
    self.n_out = n_out
    self.weight = Parameter(kaiming_normal(n_in, n_out))
    if bias:
      self.bias = Parameter(np.zeros(n_out))
    else:
      self.bias = None

  def forward(self, x):
    out = x @ self.weight
    if self.bias is not None:
      out = out + self.bias
    return out