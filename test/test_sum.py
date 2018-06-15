import horch as H

import numpy as np

from utils import gtensor, dm

def test_axis_none():
  t1 = gtensor(dm('1 2 3; 4 5 6'))
  k = 3
  (H.sum(t1) * k).backward()
  grad = dm('3 3 3; 3 3 3')
  np.testing.assert_allclose(t1.grad, grad)

def test_axis():
  t1 = gtensor(dm('1 2 3; 4 5 6'))
  k = gtensor(dm('3 6'))
  (H.sum(t1, axis=1) * k).sum().backward()
  grad = dm('3 3 3 ; 6 6 6')
  np.testing.assert_allclose(t1.grad, grad)