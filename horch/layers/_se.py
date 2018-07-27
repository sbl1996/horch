import horch as H

from ._linear import Linear
from ._module import Module

class SE(Module):
  """Squeeze and Excitation Module
  """

  def __init__(self, channels, reduction_ratio=16):
    super().__init__()
    self.channels = channels
    self.reduction_ratio = reduction_ratio

    reduced_channels = channels // reduction_ratio
    self.fc1 = Linear(channels, reduced_channels)
    self.fc2 = Linear(reduced_channels, channels)

  def forward(self, x):
    z = H.mean(x, axis=(2,3))
    z = self.fc1(z)
    z = H.relu(z)
    z = self.fc2(z)
    s = H.sigmoid(z)
    n, c = s.shape
    s = s.reshape(n, c, 1, 1)
    x = x * s
    return x