import horch as H

import numpy as np

from utils import gtensor, dm

def test_axis_none():
  x = dm('1 2 3; 4 5 6')
  hx = gtensor(x)
  k = 3
  hs = hx.std()
  (hs * k).sum().backward()
  grad = (x - x.mean()) / 5 / x.std(ddof=1) * k
  np.testing.assert_allclose(hx.grad, grad)

def test_axis():
  x = dm('1 2 3; 4 5 6')
  hx = gtensor(x)
  k = dm('3 6')
  hk = gtensor(k)
  hs = hx.std(axis=1)
  (hs * hk).sum().backward()
  grad = dm('-1.5 0 1.5; -3 0 3')
  np.testing.assert_allclose(hx.grad, grad)