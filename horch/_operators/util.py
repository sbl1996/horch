import numpy as np

def detuple(x, repeat = 2):
  if isinstance(x, tuple):
    return x
  return tuple([x] * repeat)

def patch(x, kH, kW, sH, sW, pH, pW, dH=1, dW=1):
  n, c, h, w = x.shape
  oH = (h + 2*pH - kH) // sH + 1
  oW = (w + 2*pW - kW) // sW + 1
  p = np.zeros((n, c, oH, oW, kH ,kW), dtype=x.dtype)
  if pH != 0 or pW != 0:
    x = np.pad(x, pad_width=((0,0),(0,0),(pH,pH),(pW,pW)), mode='constant')
  for i in range(kH):
    for j in range(kW):
      p[:, :, :, :, i, j] = x[:, :, i:(i+oH*sH):sH, j:(j+oW*sW):sW]
  return p

def unpatch(p, h, w, sH, sW, pH, pW, dH=1, dW=1):
  n, c, oH, oW, kH, kW = p.shape
  x = np.zeros((n, c, h+2*pH, w+2*pW), dtype=p.dtype)
  for i in range(kH):
    for j in range(kW):
      x[:, :, i:(i+oH*sH):sH, j:(j+oW*sW):sW] += p[:, :, :, :, i, j]
  x = x[:, :, pH:h+pH, pW:w+pW]
  return x

def expand_dims(a, dims):
  s = list(a.shape)
  if isinstance(dims, int):
    s.insert(dims, 1)
  else:
    for d in dims:
      s.insert(d, 1)
  return a.reshape(tuple(s))