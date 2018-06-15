import horch as H
import horch.layers as L

class HiddenNet(L.Module):

  def __init__(self, n_input, n_hidden, n_output, batch_norm=True):
    super(HiddenNet, self).__init__()
    self.n_input = n_input
    self.n_hidden = n_hidden
    self.n_output = n_output
    self.batch_norm = batch_norm

    self.l1 = L.Linear(n_input, n_hidden)
    if self.batch_norm:
        self.bn = L.BatchNorm1d(n_hidden)
    self.l2 = L.Linear(n_hidden, n_output)

  def forward(self, x):
    x = x.reshape(x.shape[0], -1)
    x = self.l1(x)
    if self.batch_norm:
        x = self.bn(x)
    x = H.relu(x)
    x = self.l2(x)

    if self.n_output == 1:
      return x.reshape(-1)
    
    return x