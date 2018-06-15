import numpy as np

from utils import gtensor, dm

def test_exp():
  t1 = gtensor(dm('-1 0 1 2'))
  t2 = t1.abs()
  t2.sum().backward()
  grad = dm('-1 0 1 1')
  np.testing.assert_allclose(t1.grad, grad)