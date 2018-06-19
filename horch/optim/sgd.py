import numpy as np
from .optimizer import Optimizer

class SGD(Optimizer):

  def __init__(self, params, lr, momentum=0, weight_decay=0):
    super(SGD, self).__init__()
    self.lr = lr
    self.momentum = momentum
    self.weight_decay = weight_decay
    self.group = {}
    self.group['params'] = list(params)
    self.group['momentum_buffer'] = {}

  def reduce(self, gamma):
    self.lr *= gamma

  def step(self):
    momentum_buffer = self.group['momentum_buffer']
    params = self.group['params']
    momentum = self.momentum
    weight_decay = self.weight_decay
    lr = self.lr

    for p in params:
      if p.grad is None:
        continue
      grad = p.grad
      if weight_decay != 0:
        grad += weight_decay * p.data
      if momentum != 0:
        if p not in momentum_buffer:
          momentum_buffer[p] = np.zeros(p.data.shape)
        buf = momentum_buffer[p]
        buf *= momentum
        buf += lr * grad
        p.data -= buf
      else:
        p.data -= lr * grad