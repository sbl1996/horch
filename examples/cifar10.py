import numpy as np
from torchvision import datasets
from torchvision.transforms import ToTensor, Normalize, Compose, RandomCrop, RandomHorizontalFlip
from torch.utils.data import DataLoader, Dataset

import horch as H
from horch.utils import evaluate_dataset
from horch.optim import SGD

from models import LeNetPlus

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
val_data = Subset(cifar_train, np.arange(m, m + 200))
val_data2 = Subset(cifar_train, np.arange(m - 200, m))
test_data = cifar_test

net = LeNetPlus()
optimizer = SGD(net.parameters(), lr=0.0003, momentum=0.9)

batch_size = 64
data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
epochs = 100
for epoch in range(epochs):
  print(epoch)
  for i, batch in enumerate(data_loader):
    inputs, labels = batch
    inputs = H.tensor(inputs.numpy())
    labels = H.tensor(labels.numpy())

    net.zero_grad()
    output = net(inputs)
    loss = H.CrossEntropyLoss(output, labels)
    loss.backward()
    optimizer.step()
  optimizer.reduce(0.95)
  print(evaluate_dataset(net, val_data2, 64))
  print(evaluate_dataset(net, val_data, 64))

# import networkx as nx

# def draw(g, **kwargs):
#   fig, ax = plt.subplots(1, 1, figsize=(8, 8))
#   pos = nx.spring_layout(g, k=2, **kwargs)
#   labels = dict(g.nodes('name'))
#   nx.draw(g, labels=labels, pos=pos, ax=ax,
#           font_size=12, node_color='skyblue')

# def op_name(op):
#   return type(op).__name__

# def computation_graph(root):
#   g = nx.DiGraph()
#   def run(op):
#     for p in op.parents:
#       g.add_node(p, name=op_name(p))
#       g.add_edge(p, op)
#       run(p)
#   g.add_node(root, name=op_name(root))
#   run(root)
#   return g

# g = computation_graph(loss.op)
# labels = dict(g.nodes('name'))
# pos = nx.spring_layout(g, k=0.1)
# nx.draw(g, labels=labels, pos=pos, node_size=600, font_size=8, node_color='skyblue')