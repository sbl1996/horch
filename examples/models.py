import horch as H
import horch.layers as L

class MLP(L.Module):

  def __init__(self, units):
    super().__init__()
    self.units = units

    layers = []
    in_features = units[0]
    for i in range(1, len(units)):
      l = units[i]
      if isinstance(l, int):
        layers.append(L.Linear(in_features, l))
        in_features = l
      elif l == 'bn':
        layers.append(L.BatchNorm1d(in_features))
      elif l == 'relu':
        layers.append(L.ReLU())
    self.seq = L.Sequential(*layers)

  def forward(self, x):
    x = self.seq(x)
    if x.shape[1] == 1:
      x = x.reshape(-1)
    return x

class LeNetPlus(L.Module):

  def __init__(self, in_channels=3, num_classes=10):
    super().__init__()
    self.conv1 = L.Conv2d(in_channels, 6, kernel_size=5)
    self.bn1 = L.BatchNorm2d(6)
    self.conv2 = L.Conv2d(6, 16, kernel_size=5)
    self.bn2 = L.BatchNorm2d(16)
    self.fc1 = L.Linear(400, 120)
    self.fc2 = L.Linear(120, 84)
    self.fc3 = L.Linear(84, num_classes)

  def forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = H.relu(x)
    x = H.max_pool2d(x, kernel_size=2, stride=2)
    x = self.conv2(x)
    x = self.bn2(x)
    x = H.relu(x)
    x = H.max_pool2d(x, kernel_size=2, stride=2)
    x = x.view(x.shape[0], -1)
    x = self.fc1(x)
    x = H.dropout(x, p=0.5, training=self.training)
    x = H.relu(x)
    x = self.fc2(x)
    x = H.dropout(x, p=0.5, training=self.training)
    x = H.relu(x)
    x = self.fc3(x)

    return x

class LeNet(L.Module):

  def __init__(self, in_channels=3, num_classes=10):
    super().__init__()
    self.conv1 = L.Conv2d(in_channels, 6, kernel_size=5)
    self.conv2 = L.Conv2d(6, 16, kernel_size=5)
    self.fc1 = L.Linear(400, 120)
    self.fc2 = L.Linear(120, 84)
    self.fc3 = L.Linear(84, num_classes)

  def forward(self, x):
    x = self.conv1(x)
    x = H.relu(x)
    x = H.max_pool2d(x, kernel_size=2, stride=2)
    x = self.conv2(x)
    x = H.relu(x)
    x = H.max_pool2d(x, kernel_size=2, stride=2)
    x = x.view(x.shape[0], -1)
    x = self.fc1(x)
    x = H.relu(x)
    x = self.fc2(x)
    x = H.relu(x)
    x = self.fc3(x)

    return x