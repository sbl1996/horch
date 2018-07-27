import numpy as np
from ._module import Module
from horch import Parameter

def kaiming_normal(n_in, n_out):
  std = np.sqrt(2 / n_in)
  return np.random.normal(scale=std, size=(n_in, n_out))

class Linear(Module):

  def __init__(self, n_in, n_out, bias=True):
    super().__init__()
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