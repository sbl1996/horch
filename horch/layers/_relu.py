import numpy as np

import horch as H

from ._module import Module

class ReLU(Module):

  def __init__(self):
    super().__init__()

  def forward(self, x):
    return H.relu(x)