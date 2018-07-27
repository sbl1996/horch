def broadcast_backward_2(l, r, lgrad, rgrad):
  ls = l.shape
  rs = r.shape
  if ls != rs:
    ldim = l.ndim
    rdim = r.ndim
    if ldim == rdim:
      diff_axes = tuple(i for i in range(ldim) if ls[i] != rs[i])
      if l.size > r.size:
        # (10, 2, 3, 5, 6, 7) + (10, 2, 1, 5, 1, 7)
        rgrad = rgrad.sum(axis=diff_axes, keepdims=True)
      else:
        # (10, 2, 1, 5, 1, 7) + (10, 2, 3, 5, 6, 7)
        lgrad = lgrad.sum(axis=diff_axes, keepdims=True)
    else:
      diff_axes = tuple(range(abs(ldim - rdim)))
      if ldim > rdim: 
        # (10, 2, 3, 5, 6, 7) + (3, 5, 6, 7)
        rgrad = rgrad.sum(axis=diff_axes)
      else:
        # (3, 5, 6, 7) + (10, 2, 3, 5, 6, 7)
        lgrad = lgrad.sum(axis=diff_axes)
  return lgrad, rgrad
