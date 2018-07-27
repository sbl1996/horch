from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

import horch as H
from horch.utils import standardize, split, evaluate
from horch.optim import SGD

from models import MLP

iris = load_iris()
X = iris.data
y = iris.target

x, mean, std = standardize(X)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33)

net = MLP([4, 'bn', 'relu', 10, 'bn', 'relu', 3])
optimizer = SGD(net.parameters(), lr=0.03, momentum=0.9)

epochs = 50
for epoch in range(epochs):
  batches = split(x_train, y_train, 8)
  for batch in batches:
    inputs, target = batch
    inputs = H.tensor(inputs)

    net.zero_grad()
    output = net(inputs)
    loss = H.CrossEntropyLoss(output, target)
    loss.backward()
    optimizer.step()
train_acc, train_loss = evaluate(net, x_train, y_train)
test_acc, test_loss = evaluate(net, x_test, y_test)
print("%d Epochs." % epochs)
print("Training set:")
print("Loss: %.4f   Accuracy: %.2f" % (train_loss, train_acc))
print("Test set:")
print("Loss: %.4f   Accuracy: %.2f" % (test_loss, test_acc))