import numpy as np
from .operator import Operator

class Conv2d(Operator):

  def __init__(self, parents, *args):
    super(Conv2d, self).__init__(parents, args)

  def _valid_conv(self, inputs, weight, stride=1):
    s = stride
    b, n_in, h, w = inputs.shape
    _, n_out, kH, kW = weight.shape
    oH = ceil((h - kH + 1) / s)
    oW = ceil((w - kW + 1) / s)
    strides = tuple(np.multiply(inputs.strides[2:], s)) + inputs.strides
    shapes = (oH, oW, b, n_out, kH, kW)
    a = np.lib.stride_tricks.as_strided(inputs, shape=shapes, strides=strides)
    return np.einsum('mnbipq,jipq->bjmn', a, weight, optimize=True)

  def _same_conv(self, inputs, weight, stride=1):
    # intermediate array leads to 2x memory usage
    s = stride
    b, n_in, h, w = inputs.shape
    _, n_out, kH, kW = weight.shape
    pH = kH // 2
    pW = kW // 2
    inputs = np.pad(inputs, pad_width=((0,0),(0,0),(pH,pH),(pW,pW)), mode='constant')
    strides = tuple(np.multiply(inputs.strides[2:], s)) + inputs.strides
    oH = (h - kH + 2*pH) // s + 1
    oW = (w - kW + 2*pW) // s + 1
    shapes = (oH, oW, b, n_out, kH, kW)
    a = np.lib.stride_tricks.as_strided(inputs, shape=shapes, strides=strides)
    return np.einsum('mnbipq,jipq->bjmn', a, weight, optimize='optimal')

  def forward(self, inputs, weight, stride, mode):
    if mode == 'same':
      return self._same_conv(inputs, weight, stride)
    elif mode == 'valid':
      return self._valid_conv(inputs, weight, stride)
    raise ValueError("mode must be one of 'same' or 'valid'")

  def backward(self, acc, inputs, weight, stride, mode):
    if mode == 'same':
      
    elif mode == 'valid':
      