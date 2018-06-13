import numpy as np
from .optimizer import Optimizer

class SGD(Optimizer):

  def __init__(self, params, lr, momentum=0):
    super(SGD, self).__init__()
    self.lr = lr
    self.momentum = momentum
    self.group = {}
    self.group['params'] = list(params)
    self.group['momentum_buffer'] = {}

  def step(self):
    momentum_buffer = self.group['momentum_buffer']
    params = self.group['params']
    momentum = self.momentum
    lr = self.lr

    for p in params:
      if p.grad is None:
        continue
      if momentum != 0:
        if p not in momentum_buffer:
          momentum_buffer[p] = np.zeros(p.data.shape)
        buf = momentum_buffer[p]
        buf *= momentum
        buf += lr * p.grad
        p.data -= buf
      else:
        p.data -= lr * p.grad