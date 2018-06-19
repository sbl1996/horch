import numpy as np
from .operator import Operator

class Conv2d(Operator):

  def __init__(self, parents, *args):
    super().__init__(parents, args)

  def forward(self, input, weight, bias, stride, padding):
    return conv2d(input, weight, bias, stride, padding)

  def backward(self, acc, input, weight, bias, stride, padding):
    gx, gW, gb = conv2d_backward(acc, input, weight, bias, stride, padding)
    return gx, gW, gb
    

def detuple(x, repeat = 2):
  if isinstance(x, tuple):
    return x
  return tuple([x] * repeat)

def _conv(x, W, b, sH, sW, pH, pW, dH, dW):
  n, n_in, h, w = x.shape
  n_out, n_in, kH, kW = W.shape
  oH = (h + 2*pH - kH) // sH + 1
  oW = (w + 2*pW - kW) // sW + 1
  if pH != 0 or pW != 0:
    x = np.pad(x, pad_width=((0,0),(0,0),(pH,pH),(pW,pW)), mode='constant')
  strides = tuple(np.multiply(x.strides[2:], (sH, sW))) + x.strides
  shape = (oH, oW, n, n_in, kH, kW)
  x = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)
  x = np.einsum('mnbipq,jipq->bjmn', x, W, optimize=True)
  if b is not None:
    b = b.reshape(1, n_out, 1, 1)
    x += b
  return x

def conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1):
  sH, sW = detuple(stride)
  pH, pW = detuple(padding)
  dH, dW = detuple(dilation)
  return _conv(input, weight, bias, sH, sW, pH, pW, dH, dW)

def _conv_backward(g, x, W, b, sH, sW, pH, pW, dH, dW):
  n, n_in, h, w = x.shape
  n_out, n_in, kH, kW = W.shape
  oH, oW = g.shape[2:]
  if pH != 0 or pW != 0:
    x = np.pad(x, pad_width=((0,0),(0,0),(pH,pH),(pW,pW)), mode='constant')
  strides = tuple(np.multiply(x.strides[2:], (sH, sW))) + x.strides
  shape = (oH, oW, n, n_in, kH, kW)
  x = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)
  gW = np.einsum('mnbipq,bjmn->jipq', x, g, optimize=True)

  # (n, n_in, oH, oW, kH, kW)
  z = np.einsum('bjmn,jipq->bimnpq', g, W, optimize=True)
  gx = np.zeros((n, n_in, h+2*pH, w+2*pW), dtype=x.dtype)
  for i in range(kH):
    for j in range(kW):
      gx[:, :, i:(i+oH*sH):sH, j:(j+oW*sW):sW] += z[:, :, :, :, i, j]
  gx = gx[:, :, pH:h+pH, pW:w+pW]
  if b is None:
    gb = None
  else:
    gb = g.sum(axis=(0, 2, 3))
  return gx, gW, gb

def conv2d_backward(acc, input, weight, bias=None, stride=1, padding=0, dilation=1):
  sH, sW = detuple(stride)
  pH, pW = detuple(padding)
  dH, dW = detuple(dilation)
  return _conv_backward(acc, input, weight, bias, sH, sW, pH, pW, dH, dW)

def conv2d_group(input, weight, bias, stride, padding):
  pass
