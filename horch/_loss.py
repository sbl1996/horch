import numpy as _np
import horch as _H

def BCELoss(logits, target):
  y = target
  z = logits
  return _H.mean(_H.maximum(z, 0) + _H.log(1 + _H.exp(-_H.abs(z))) - z*y)

def CrossEntropyLoss(inputs, target):
  """
  Args:
    inputs: (N, C) where C = number of classes
    target: (N) where each value is 0 <= c <= C - 1
  """
  N = len(inputs)
  return _H.mean(-inputs[_np.arange(N), target] + _H.log(_H.sum(_H.exp(inputs), axis=1)))
