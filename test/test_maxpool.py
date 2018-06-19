import numpy as np
import pytest

import horch as H

from utils import gtensor, dm

def test_max_pool2d():
  kernel_size = 3
  stride = 2
  padding = 0
  a = [
    [1, 2, 3, 2, 1],
    [2, 1, 4, 4, 2],
    [3, 2, 5, 4, 1],
    [1, 3, 2, 2, 2],
    [2, 1, 3, 4, 1],
  ]
  k = [
    [1, 2],
    [3, 4],
  ]
  c = [
    [5, 5],
    [5, 5],
  ]
  a = dm(a, dtype=np.double).reshape(1, 1, 5, 5)
  k = dm(k, dtype=np.double).reshape(1, 1, 2, 2)
  c = dm(c, dtype=np.double).reshape(1, 1, 2, 2)
  ta = gtensor(a)
  tk = gtensor(k)

  tc = H.max_pool2d(ta, kernel_size, stride=stride, padding=padding)
  (tc * tk).sum().backward()

  ga = [
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 10, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
  ]
  ga = dm(ga, dtype=np.double).reshape(1, 1, 5, 5)

  np.testing.assert_allclose(tc.data, c)
  np.testing.assert_allclose(ta.grad, ga)

# def test_max_pool2d():
#   kernel_size = 3
#   stride = 1
#   padding = 0
#   a = [
#     [1, 2, 3, 5],
#     [2, 1, 3, 4],
#     [3, 2, 1, 4],
#     [4, 3, 2, 8],
#   ]
#   k = [
#     [1, 2],
#     [3, 4],
#   ]
#   a = dm(a, dtype=np.double).reshape(1, 1, 4, 4)
#   k = dm(k, dtype=np.double).reshape(1, 1, 2, 2)
#   ta = gtensor(a)
#   tk = gtensor(k)

#   tc = H.max_pool2d(ta, kernel_size, stride=stride, padding=padding)


#   (tc * tk).sum().backward()

#   c = dm([[[
#     [ 3.,  5.],
#     [ 4.,  8.]
#   ]]])
#   ga = dm([[[
#     [  1.,   3.,   3.,   2.],
#     [  4.,  11.,  12.,   6.],
#     [  4.,  13.,  16.,  10.],
#     [  3.,   7.,  13.,  12.]
#   ]]])
#   gb = dm([[[
#     [ 15.,  23.,  36.],
#     [ 21.,  17.,  30.],
#     [ 31.,  21.,  19.]
#   ]]])
#   np.testing.assert_allclose(tc.data, c)
#   np.testing.assert_allclose(ta.grad, ga)
#   np.testing.assert_allclose(tb.grad, gb)

@pytest.mark.torch
def test_any():
  import torch
  import torch.nn.functional as F
  kernel_size = 3
  stride = 2
  padding = 1
  a = np.random.rand(2, 2, 8, 8)
  ta = torch.tensor(a, requires_grad=True)
  ga = gtensor(a)

  tc = F.max_pool2d(ta, kernel_size, stride=stride, padding=padding)
  gc = H.max_pool2d(ga, kernel_size, stride=stride, padding=padding)

  k = np.random.rand(*tc.shape)
  tk = torch.tensor(k, requires_grad=True)
  gk = gtensor(k)
  (tc * tk).sum().backward()
  (gc * gk).sum().backward()
  
  c = tc.detach().numpy()
  g = ta.grad.numpy()
  np.testing.assert_allclose(gc.data, c)
  np.testing.assert_allclose(ga.grad, g)