import pytest

import numpy as np

import horch as H

from utils import gtensor, dm, module_exists

def test_conv2d():
  a = [
    [1, 2, 3, 4],
    [2, 1, 3, 4],
    [3, 2, 1, 4],
    [4, 3, 2, 1],
  ]
  b = [
    [1, 1, 1],
    [1, 2, 1],
    [1, 1, 3],
  ]
  k = [
    [1, 2],
    [3, 4],
  ]
  a = dm(a, dtype=np.double).reshape(1, 1, 4, 4)
  b = dm(b, dtype=np.double).reshape(1, 1, 3, 3)
  k = dm(k, dtype=np.double).reshape(1, 1, 2, 2)
  ta = gtensor(a)
  tb = gtensor(b)
  tk = gtensor(k)

  tc = H.conv2d(ta, tb)

  (tc * tk).sum().backward()

  c = dm([[[
    [ 21.,  35.],
    [ 27.,  24.]
  ]]])
  ga = dm([[[
    [  1.,   3.,   3.,   2.],
    [  4.,  11.,  12.,   6.],
    [  4.,  13.,  16.,  10.],
    [  3.,   7.,  13.,  12.]
  ]]])
  gb = dm([[[
    [ 15.,  23.,  36.],
    [ 21.,  17.,  30.],
    [ 31.,  21.,  19.]
  ]]])
  np.testing.assert_allclose(tc.data, c)
  np.testing.assert_allclose(ta.grad, ga)
  np.testing.assert_allclose(tb.grad, gb)


def test_conv2d_with_padding():
  a = [
    [1, 2, 3, 4],
    [2, 1, 3, 4],
    [3, 2, 1, 4],
    [4, 3, 2, 1],
  ]
  b = [
    [1, 1, 1],
    [1, 2, 1],
    [1, 1, 3],
  ]
  k = [
    [1, 2, 3, 4],
    [3, 4, 5, 2],
    [6, 1, 2, 1],
    [2, 3, 1, 6],
  ]
  a = dm(a, dtype=np.double).reshape(1, 1, 4, 4)
  b = dm(b, dtype=np.double).reshape(1, 1, 3, 3)
  k = dm(k, dtype=np.double).reshape(1, 1, 4, 4)
  ta = gtensor(a)
  tb = gtensor(b)
  tk = gtensor(k)

  tc = H.conv2d(ta, tb, padding=1)

  (tc * tk).sum().backward()

  c = dm([[[
    [  9.,  20.,  28.,  18.],
    [ 17.,  21.,  35.,  23.],
    [ 24.,  27.,  24.,  19.],
    [ 16.,  18.,  15.,   9.],
  ]]])
  ga = dm([[[
    [ 11.,  20.,  23.,  18.],
    [ 20.,  33.,  33.,  25.],
    [ 25.,  34.,  35.,  28.],
    [ 14.,  30.,  17.,  20.],
  ]]])
  gb = dm([[[
    [  44.,   94.,   66.],
    [  74.,  114.,   89.],
    [  55.,   91.,   71.],
  ]]])
  np.testing.assert_allclose(tc.data, c)
  np.testing.assert_allclose(ta.grad, ga)
  np.testing.assert_allclose(tb.grad, gb)

def test_conv2d_with_stride():
  a = [
    [1, 2, 3, 4, 5],
    [2, 1, 3, 4, 5],
    [3, 2, 1, 4, 5],
    [4, 3, 2, 1, 5],
    [5, 2, 1, 3, 4],
  ]
  b = [
    [1, 1, 1],
    [1, 2, 1],
    [1, 1, 3],
  ]
  k = [
    [1, 2],
    [3, 4],
  ]
  a = dm(a, dtype=np.double).reshape(1, 1, 5, 5)
  b = dm(b, dtype=np.double).reshape(1, 1, 3, 3)
  k = dm(k, dtype=np.double).reshape(1, 1, 2, 2)
  ta = gtensor(a)
  tb = gtensor(b)
  tk = gtensor(k)

  tc = H.conv2d(ta, tb, stride=2)

  (tc * tk).sum().backward()

  c = dm([[[
    [ 21.,  48.],
    [ 28.,  35.],
  ]]])
  ga = dm([[[
    [  1.,   1.,   3.,   2.,   2.],
    [  1.,   2.,   3.,   4.,   2.],
    [  4.,   4.,  12.,   6.,  10.],
    [  3.,   6.,   7.,   8.,   4.],
    [  3.,   3.,  13.,   4.,  12.],
  ]]])
  gb = dm([[[
    [ 20.,  32.,  36.],
    [ 28.,  22.,  39.],
    [ 24.,  28.,  30.],
  ]]])
  np.testing.assert_allclose(tc.data, c)
  np.testing.assert_allclose(ta.grad, ga)
  np.testing.assert_allclose(tb.grad, gb)

@pytest.mark.torch
def test_any():
  import torch
  import torch.nn.functional as F
  stride = 3
  padding = 5
  a = np.random.rand(2, 2, 5, 5)
  w = np.random.rand(3, 2, 3, 3)
  b = np.random.rand(3)
  ta = torch.tensor(a, requires_grad=True)
  tw = torch.tensor(w, requires_grad=True)
  tb = torch.tensor(b, requires_grad=True)
  ha = gtensor(a)
  hw = gtensor(w)
  hb = gtensor(b)

  tc = F.conv2d(ta, tw, stride=stride, padding=padding, bias=tb)
  hc = H.conv2d(ha, hw, stride=stride, padding=padding, bias=hb)
  k = np.random.rand(*tc.shape)
  tk = torch.tensor(k, requires_grad=True)
  hk = gtensor(k)
  (tc * tk).sum().backward()
  (hc * hk).sum().backward()
  gta = ta.grad.numpy()
  gtw = tw.grad.numpy()
  gtb = tb.grad.numpy()
  np.testing.assert_allclose(tc.clone().detach().numpy(), hc.data)
  np.testing.assert_allclose(gta, ha.grad)
  np.testing.assert_allclose(gtw, hw.grad)
  np.testing.assert_allclose(gtb, hb.grad)