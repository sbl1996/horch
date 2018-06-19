import numpy as np
from ._module import Module
from horch import Parameter

class Sequential(Module):

  def __init__(self, *layers):
    super().__init__()
    for i, layer in enumerate(layers):
      self.register_child(str(i), layer)

  def forward(self, x):
    for layer in self.children():
      x = layer(x)
    return x