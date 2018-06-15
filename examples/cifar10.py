import numpy as np
from torchvision import datasets
from torchvision.transforms import ToTensor, Normalize, Compose, RandomCrop, RandomHorizontalFlip
from torch.utils.data import DataLoader, Dataset

import horch as H
from horch.utils import evaluate_dataset
from horch.optim import SGD

from hidden_net import HiddenNet

class Subset(Dataset):
  def __init__(self, dataset, indices):
    self.dataset = dataset
    if isinstance(indices, int):
      self.indices = np.arange(indices)
    else:
      self.indices = indices

  def __len__(self):
    return len(self.indices)

  def __getitem__(self, idx):
    return self.dataset[self.indices[idx]]

train_transform = Compose([
  RandomCrop(32, padding=4),
  RandomHorizontalFlip(),
  ToTensor(),
  Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

test_transform = Compose([
  ToTensor(),
  Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

CIFAR_DATA_HOME = 'D:\MLCode\datasets\CIFAR10'
cifar_train = datasets.CIFAR10(CIFAR_DATA_HOME, train=True, transform=train_transform, download=True)
cifar_test = datasets.CIFAR10(CIFAR_DATA_HOME, train=False, transform=test_transform, download=True)

m = 5000
train_data = Subset(cifar_train, m)
val_data = Subset(cifar_train, np.arange(m, m + 500))
test_data = cifar_test

net = HiddenNet(3072, 128, 10, args.batch_norm)
optimizer = SGD(net.parameters(), lr=0.01, momentum=0.9)

batch_size = 32
data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
epochs = 5
for epoch in range(epochs):
  print(epoch)
  for batch in data_loader:
    inputs, labels = batch
    inputs = H.tensor(inputs.numpy())
    labels = H.tensor(labels.numpy())

    net.zero_grad()
    output = net(inputs)
    loss = H.CrossEntropyLoss(output, labels)
    loss.backward()
    optimizer.step()
evaluate_dataset(net, cifar_train, 64)