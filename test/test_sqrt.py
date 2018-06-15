import numpy as np

from utils import gtensor, dm

def test_sqrt():
  t1 = gtensor(dm('1 4'))
  t2 = t1.sqrt()
  k = gtensor(dm('2 3'))
  (t2 * k).sum().backward()
  grad = dm('1 .75')
  np.testing.assert_allclose(t1.grad, grad)