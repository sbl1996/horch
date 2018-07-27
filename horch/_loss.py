import numpy as _np
import horch as _H

def BCELoss(logits, target):
  y = target
  z = logits
  return _H.mean(_H.maximum(z, 0) + _H.log(1 + _H.exp(-_H.abs(z))) - z*y)

def CrossEntropyLoss(logits, target):
  """
  Args:
    logits: (N, C) where C = number of classes
    target: (N) where each value is 0 <= c <= C - 1
  """
  return _H.cross_entropy(logits, target)