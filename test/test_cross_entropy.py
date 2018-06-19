import numpy as np

import horch as H
from utils import gtensor, dm

def test_cross_entropy():
  logit = [
    [  6,  -9, -11, -11,  13],
    [  2,   7,  -8,  11,   2],
    [  0,  17,   9,  11,  -5],
  ]
  target = [ 2, 3, 4 ]
  logit = np.array(logit, dtype=np.double)
  target = np.array(target, dtype=np.longlong)
  tlogit = gtensor(logit)
  tloss = H.cross_entropy(tlogit, target)
  tloss.backward()
  loss = np.array(15.34070468499769)
  grad = dm([
    [ 3.03683731e-04,  9.28975581e-11, -3.33333333e-01,  1.25723173e-11,  3.33029649e-01],
    [ 4.03869206e-05,  5.99395047e-03,  1.83356336e-09, -6.07472615e-03,  4.03869206e-05],
    [ 1.37610652e-08,  3.32397880e-01,  1.11507066e-04,  8.23931970e-04, -3.33333333e-01],
  ])
  np.testing.assert_allclose(tloss.data, loss)
  np.testing.assert_allclose(tlogit.grad, grad)