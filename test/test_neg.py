import numpy as np

from utils import gtensor, dm

def test_neg():
  t1 = gtensor(dm('1 2'))
  t2 = -t1
  t2.sum().backward()
  grad = dm('-1 -1')
  assert np.array_equal(t1.grad, grad)