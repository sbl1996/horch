import numpy as np

from utils import gtensor, dm

def test_maximum():
  # vector add vector
  t1 = gtensor(dm('3 2'))
  t2 = gtensor(dm('1 4'))
  k = gtensor(dm('4 6'))
  # (t1.max(t2) * k).sum().backward()
  (t1.maximum(t2) * k).sum().backward()
  grad1 = dm('4 0')
  grad2 = dm('0 6')
  np.testing.assert_allclose(t1.grad, grad1)
  np.testing.assert_allclose(t2.grad, grad2)

  # matrix add matrix
  t1 = gtensor(dm('1 2 3; 5 6 7'))
  t2 = gtensor(dm('3 4 5; 2 3 4'))
  k = gtensor(dm('1 2 3; 4 5 6'))
  # (t1.max(t2) * k).sum().backward()
  (t1.maximum(t2) * k).sum().backward()
  grad1 = dm('0 0 0; 4 5 6')
  grad2 = dm('1 2 3; 0 0 0')
  np.testing.assert_allclose(t1.grad, grad1)
  np.testing.assert_allclose(t2.grad, grad2)

  # matrix add vector
  t1 = gtensor(dm('2 2 2; 3 4 5'))
  t2 = gtensor(dm('2 6 4'))
  k = gtensor(dm('1 2 3; 4 5 6'))
  (t1.maximum(t2) * k).sum().backward()
  grad1 = dm('0 0 0; 4 0 6')
  grad2 = dm('1 7 3')
  np.testing.assert_allclose(t1.grad, grad1)
  np.testing.assert_allclose(t2.grad, grad2)

  # matrix add scalar
  t1 = gtensor(dm('1 2 3; 3 4 5'))
  t2 = gtensor(dm('2'))
  k = gtensor(dm('1 2 3; 4 5 6'))
  # (t1.max(t2) * k).sum().backward()
  (t1.maximum(t2) * k).sum().backward()
  grad1 = dm('0 0 3; 4 5 6')
  grad2 = dm('3')
  np.testing.assert_allclose(t1.grad, grad1)
  np.testing.assert_allclose(t2.grad, grad2)

  # vector add scalar
  t1 = gtensor(dm('1 2 3'))
  t2 = gtensor(dm('2'))
  k = gtensor(dm('4 5 6'))
  # (t1.max(t2) * k).sum().backward()
  (t1.maximum(t2) * k).sum().backward()
  grad1 = dm('0 0 6')
  grad2 = dm('9')
  np.testing.assert_allclose(t1.grad, grad1)
  np.testing.assert_allclose(t2.grad, grad2)