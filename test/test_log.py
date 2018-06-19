import numpy as np

from utils import gtensor, dm

def test_log():
  t1 = gtensor(dm('1 2'))
  t2 = t1.log()
  t2.sum().backward()
  grad = 1 / t1.data
  np.testing.assert_allclose(t1.grad, grad)