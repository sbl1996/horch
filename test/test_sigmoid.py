import numpy as np

from utils import gtensor, dm

def test_sigmoid():
  t1 = gtensor(dm('1 2'))
  t2 = t1.sigmoid()
  t2.sum().backward()
  grad = t2.data * (1 - t2.data)
  np.testing.assert_allclose(t1.grad, grad)