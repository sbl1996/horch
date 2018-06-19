import numpy as _np
import horch as _H

def check_tensor(t):
  if t is None:
    return None
  if not isinstance(t, _H.Tensor):
    if not isinstance(t, _np.ndarray):
      t = _np.array(t)
    t = _H.Tensor(t)
  return t

def evaluate(net, x, y, binary=False):
  """
  Args:
      net: model
      x: (num_samples, num_features)
      y: (num_samples,) for binary label, or
         (num_samples, num_classes) for multiclass label
  """
  net.eval()
  x = _H.tensor(x)
  out = net(x)
  if binary:
    pred = _np.sign(out.data)
    pred[pred == -1] = 0
    criterian = _H.BCELoss
  else:
    pred = _np.argmax(out.data, axis=1)
    criterian = _H.CrossEntropyLoss
  loss = criterian(out, y).item()
  acc = _np.mean(pred == y)
  net.train()
  return loss, acc

def evaluate_dataset(net, dataset, batch_size=32):
  """
  Args:
      net: model
      x: (num_samples, num_features)
      y: (num_samples,) for binary label, or
         (num_samples, num_classes) for multiclass label
  """
  net.eval()
  from torch.utils.data import DataLoader
  data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
  n_correct = 0
  loss_avg = 0
  i = 0
  for batch in data_loader:
    i += 1
    inputs, labels = batch
    inputs = _H.tensor(inputs.numpy())
    labels = _H.tensor(labels.numpy())

    net.zero_grad()
    output = net(inputs)
    loss = _H.CrossEntropyLoss(output, labels).item()
    loss_avg = (loss_avg * (i - 1) + loss) / i
    pred = _np.argmax(output.data, axis=1)
    n_correct += (pred == labels.data).sum()
  net.train()
  return n_correct / len(dataset), loss_avg

def split(x, y, batch_size, shuffle=True):
  m = len(x)
  assert m == len(y)
  indices = _np.random.permutation(m) if shuffle else _np.arange(m)
  batches = int(_np.ceil(m / batch_size))
  for i in range(batches - 1):
    ind = indices[i*batch_size : (i+1)*batch_size]
    yield (x[ind], y[ind])
  ind = indices[(batches - 1) * batch_size:]
  yield (x[ind], y[ind])

def standardize(x, axis=0):
  """
  Args:
      x: (num_samples, num_features)
      axis:

  Returns:
      x_normalized: (num_samples, num_features)
      mean: (num_features,)
      std: (num_features,)
  """
  mean = _np.mean(x, axis=axis)
  std = _np.std(x, axis=axis)
  return (x - mean) / std, mean, std
