from .operator import Operator

class Mul(Operator):

  def __init__(self, parents, *args):
    super().__init__(parents, args)

  def forward(self, l, r):
    return l * r

  def backward(self, acc, l, r):
    lgrad = acc * r
    rgrad = acc * l
    ls = l.shape
    rs = r.shape
    lsize = l.size
    rsize = r.size
    if ls != rs:
      ldim = l.ndim
      rdim = r.ndim
      if ldim == rdim:
        diff_axes = tuple(filter(lambda i: ls[i] != rs[i], range(ldim)))
        if lsize > rsize: # (10, 2, 3, 5, 6, 7) - (10, 2, 1, 5, 1, 7)
            rgrad = rgrad.sum(axis=diff_axes, keepdims=True)
        else: # (10, 2, 1, 5, 1, 7) - (10, 2, 3, 5, 6, 7)
            lgrad = lgrad.sum(axis=diff_axes, keepdims=True)
      else:
        diff_axes = tuple(range(abs(ldim - rdim)))
        if ldim > rdim: # (10, 2, 3, 5, 6, 7) - (3, 5, 6, 7)
          rgrad = rgrad.sum(axis=diff_axes)
        else: # (3, 5, 6, 7) - (10, 2, 3, 5, 6, 7)
          lgrad = lgrad.sum(axis=diff_axes)
    return lgrad, rgrad