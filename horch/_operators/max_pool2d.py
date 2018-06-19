import numpy as np

from .operator import Operator
from .util import detuple, patch, unpatch

class MaxPool2d(Operator):

  def __init__(self, parents, *args):
    super().__init__(parents, args)

  def forward(self, input, kernel_size, stride, padding, return_indices):
    if return_indices:
      x, indices = max_pool2d(input, kernel_size, stride, padding, return_indices)
      self.indices = indices
    else:
      x = max_pool2d(input, kernel_size, stride, padding, return_indices)
    return x
    

  def backward(self, acc, input, kernel_size, stride, padding, return_indices):
    g = max_pool2d_backward(acc, input, kernel_size, stride, padding, self.indices)
    return g

def _max_pool2d_no_indices(x, kH, kW, sH, sW, pH, pW, dH=1, dW=1):
  x = patch(x, kH, kW, sH, sW, pH, pW, dH, dW)
  return np.max(x, axis=(4,5))

def _max_pool2d(x, kH, kW, sH, sW, pH, pW, dH=1, dW=1):
  x = patch(x, kH, kW, sH, sW, pH, pW, dH, dW)
  n, c, oH, oW = x.shape[:4]
  size = n * c * oH * oW
  x = x.reshape(size, kH * kW)
  indices = np.argmax(x, axis=-1)
  x = x[np.arange(size), indices].reshape(n, c, oH, oW)
  return x, indices


# def _max_pool2d(x, kH, kW, sH, sW, pH, pW, dH=1, dW=1):
#   x = patch(x, kH, kW, sH, sW, pH, pW, dH, dW)
#   n, c, oH, oW = x.shape[:4]
#   indices = np.argmax(x.reshape(n * c * oH * oW, kH * kW), axis=-1)
#   x = np.max(x, axis=(4,5))
#   return x, indices

def max_pool2d(input, kernel_size, stride, padding, return_indices):
  kH, kW = detuple(kernel_size)
  sH, sW = detuple(stride)
  pH, pW = detuple(padding)
  if return_indices:
    return _max_pool2d(input, kH, kW, sH, sW, pH, pW, 1, 1)
  else:
    return _max_pool2d_no_indices(x, kH, kW, sH, sW, pH, pW, dH=1, dW=1)

def _max_pool2d_backward(g, x, kH, kW, sH, sW, pH, pW, dH, dW, ind):
  h, w = x.shape[2:]
  n, c, oH, oW = g.shape
  size = n * c * oH * oW
  gx = np.zeros((size, kH * kW), dtype=x.dtype)
  gx[np.arange(size), ind] = g.reshape(-1)
  gx = gx.reshape(n, c, oH, oW, kH, kW)
  return unpatch(gx, h, w, sH, sW, pH, pW, dH, dW)

def max_pool2d_backward(acc, input, kernel_size, stride, padding, ind):
  kH, kW = detuple(kernel_size)
  sH, sW = detuple(stride)
  pH, pW = detuple(padding)
  return _max_pool2d_backward(acc, input, kH, kW, sH, sW, pH, pW, 1, 1, ind)