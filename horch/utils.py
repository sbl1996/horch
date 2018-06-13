import numpy as _np
import horch as _H

def check_tensor(t):
  from .tensor import Tensor
  if not isinstance(t, Tensor):
    if not isinstance(t, _np.ndarray):
      t = _np.array(t)
    t = Tensor(t)
  return t

def evaluate(net, x, y, binary=False):
  """
  Args:
      net: model
      x: (num_samples, num_features)
      y: (num_samples,) for binary label, or
         (num_samples, num_classes) for multiclass label
  """
  x = _H.tensor(x)
  out = net(x)
  if binary:
    pred = _np.sign(out.data)
    pred[pred == -1] = 0
    criterian = _H.BCELoss
  else:
    pred = _np.argmax(out.data, axis=1)
    criterian = _H.CrossEntropyLoss
  loss = criterian(out, y).item()
  acc = _np.mean(pred == y)
  return loss, acc

def split(x, y, batch_size, shuffle=True):
  m = len(x)
  assert m == len(y)
  indices = _np.random.permutation(m) if shuffle else _np.arange(m)
  batches = int(_np.ceil(m / batch_size))
  for i in range(batches - 1):
    ind = indices[i*batch_size : (i+1)*batch_size]
    yield (x[ind], y[ind])
  ind = indices[(i+1)*batch_size:]
  yield (x[ind], y[ind])

def standardize(x, axis=0):
  """
  Args:
      x: (num_samples, num_features)
      axis:

  Returns:
      x_normalized: (num_samples, num_features)
      mean: (num_features,)
      std: (num_features,)
  """
  mean = _np.mean(x, axis=axis)
  std = _np.std(x, axis=axis)
  return (x - mean) / std, mean, std
