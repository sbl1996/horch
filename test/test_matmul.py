import numpy as np

from utils import gtensor, dm

def test_matmul_matrix():
  # (2, 3) @ (3, 4)
  t1 = gtensor(dm('1 2 3; 3 4 5'))
  t2 = gtensor(dm('3 4 5 6; 5 6 7 8; 1 2 3 4'))
  (t1 @ t2).sum().backward()
  grad1 = dm('18 26 10; 18 26 10')
  grad2 = dm('4 4 4 4; 6 6 6 6; 8 8 8 8')
  assert np.array_equal(t1.grad, grad1)
  assert np.array_equal(t2.grad, grad2)

def test_matrix_vector():
  # (2, 3) @ (3,)
  t1 = gtensor(dm('1 2 3; 3 4 5'))
  t2 = gtensor(dm('4 4 6'))
  (t1 @ t2).sum().backward()
  grad1 = dm('4 4 6; 4 4 6')
  grad2 = dm('4 6 8')
  assert np.array_equal(t1.grad, grad1)
  assert np.array_equal(t2.grad, grad2)

def test_vector_matrix():
  # (2, 3) @ (3,)
  t1 = gtensor(dm('4 9'))
  t2 = gtensor(dm('1 2 3; 3 4 5'))
  (t1 @ t2).sum().backward()
  grad1 = dm('6 12')
  grad2 = dm('4 4 4; 9 9 9')
  assert np.array_equal(t1.grad, grad1)
  assert np.array_equal(t2.grad, grad2)

def test_vector_vector():
  # (3,) @ (3,)
  t1 = gtensor(dm('4 9 3'))
  t2 = gtensor(dm('8 2 5'))
  (t1 @ t2).sum().backward()
  grad1 = dm('8 2 5')
  grad2 = dm('4 9 3')
  assert np.array_equal(t1.grad, grad1)
  assert np.array_equal(t2.grad, grad2)  