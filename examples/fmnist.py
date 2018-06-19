import numpy as np
from torchvision import datasets
from torchvision.transforms import ToTensor, Normalize, Compose, RandomCrop, RandomHorizontalFlip, Pad
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
  Pad(2),
  ToTensor(),
])

test_transform = Compose([
  Pad(2),
  ToTensor(),
])

FMNIST_DATA_HOME = 'D:\MLCode\datasets\FashionMNIST'
fmnist_train = datasets.FashionMNIST(FMNIST_DATA_HOME, train=True, transform=train_transform, download=True)
fmnist_test = datasets.FashionMNIST(FMNIST_DATA_HOME, train=False, transform=test_transform, download=True)

m = 5000
train_data = Subset(fmnist_train, m)
val_data = Subset(fmnist_train, np.arange(m, m + 200))
val_data2 = Subset(fmnist_train, np.arange(m - 200, m))
test_data = fmnist_test

net = LeNetPlus(in_channels=1)
optimizer = SGD(net.parameters(), lr=0.003, momentum=0.9, weight_decay=5e-4)

batch_size = 32
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

# nx.draw(g, labels=labels, pos=pos, font_size=12, node_color='skyblue')