from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

import horch as H
from horch.utils import standardize, split, evaluate
from horch.optim import SGD

from models import MLP

bre = load_breast_cancer()
X = bre.data
y = bre.target

x = X

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33)

net = MLP([30, 'bn', 'relu', 10, 'bn', 'relu', 1])
optimizer = SGD(net.parameters(), lr=0.01, momentum=0.9)

epochs = 20
for epoch in range(epochs):
  batches = split(x_train, y_train, 8)
  for batch in batches:
    inputs, target = batch
    inputs = H.tensor(inputs)
    target = H.tensor(target)

    net.zero_grad()
    output = net(inputs)
    loss = H.BCELoss(output, target)
    loss.backward()
    optimizer.step()
train_loss, train_acc = evaluate(net, x_train, y_train, binary=True)
test_loss, test_acc = evaluate(net, x_test, y_test, binary=True)
print("%d Epochs." % epochs)
print("Training set:")
print("Loss: %.4f   Accuracy: %.2f" % (train_loss, train_acc))
print("Test set:")
print("Loss: %.4f   Accuracy: %.2f" % (test_loss, test_acc))