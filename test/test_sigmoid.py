import numpy as np

import horch as H

from utils import gtensor, dm

def test_sigmoid():
  a = gtensor(dm('-2 3'))
  k = gtensor(dm('2 3'))
  b = H.sigmoid(a)
  (b * k).sum().backward()
  grad = dm('0.20998717 0.13552998')
  np.testing.assert_allclose(a.grad, grad)