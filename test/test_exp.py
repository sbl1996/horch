import numpy as np

from utils import gtensor, dm

def test_exp():
  t1 = gtensor(dm('1 2'))
  t2 = t1.exp()
  t2.sum().backward()
  grad = t2.data.copy()
  np.testing.assert_allclose(t1.grad, grad)