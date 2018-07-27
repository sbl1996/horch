import numpy as np
from torchvision import datasets
from torchvision.transforms import ToTensor, Normalize, Compose, RandomCrop, RandomHorizontalFlip, Pad
from torch.utils.data import DataLoader, Dataset

import horch as H
from horch.utils import evaluate_dataset
from horch.optim import SGD

from models import MLP

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
  ToTensor(),
])

test_transform = Compose([
  ToTensor(),
])

MNIST_DATA_HOME = 'D:\MLCode\datasets\MNIST'
mnist_train = datasets.MNIST(MNIST_DATA_HOME, train=True, transform=train_transform, download=True)
mnist_test = datasets.MNIST(MNIST_DATA_HOME, train=False, transform=test_transform, download=True)

m = 59000
train_data = Subset(mnist_train, m)
val_data = Subset(mnist_train, np.arange(m, m + 1000))
test_data = mnist_test

net = MLP([784, 'bn', 'relu', 100, 'bn', 'relu', 10])
optimizer = SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

batch_size = 32
data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
epochs = 5
for epoch in range(epochs):
  print("Epoch %d" % (epoch + 1))
  for i, batch in enumerate(data_loader):
    input, labels = batch
    input = H.tensor(input.numpy())
    labels = H.tensor(labels.numpy())

    net.zero_grad()
    output = net(input)
    loss = H.CrossEntropyLoss(output, labels)
    loss.backward()
    optimizer.step()
  optimizer.reduce(0.95)
  val_acc, val_loss = evaluate_dataset(net, val_data, 64)
  print("val_loss: %.4f   val_acc: %.2f" % (val_loss, val_acc))

train_acc, train_loss = evaluate_dataset(net, train_data, 64)
print("%d Epochs." % epochs)
print("Training set:")
print("Loss: %.4f   Accuracy: %.2f" % (train_loss, train_acc))
test_acc, test_loss = evaluate_dataset(net, test_data, 64)
print("Test set:")
print("Loss: %.4f   Accuracy: %.2f" % (test_loss, test_acc))