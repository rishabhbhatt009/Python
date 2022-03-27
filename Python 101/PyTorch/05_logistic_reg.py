# Design Philosophy
#
# 1) Design Model
#       - input size
#       - output size
#       - fwd pass
# 2) Construct Loss and Optimizer
# 3) Training Loop
#       - fwd pass : compute prediction and loss
#       - bwd pass : compute gradient
#       - update : update weights

import numpy as np
import torch
import torch.nn as nn
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 0) Data Prep ---------------------------------------------------------------------------------
bc = datasets.load_breast_cancer()
X, y = bc.data, bc.target

print(f'X : {X[:5]}',
      f'y : {y[:5]}',
      f'y : {np.unique(y, return_counts=True)}\n',
      sep='\n'
      )
n_samples, n_features = X.shape

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

# Scale
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Creating Tensors
X_train = torch.tensor(X_train.astype(np.float32))
X_test = torch.tensor(X_test.astype(np.float32))
y_train = torch.tensor(y_train.astype(np.float32))
y_test = torch.tensor(y_test.astype(np.float32))

# Reshaping y
y_train = y_train.view(-1, 1)
y_test = y_test.view(-1, 1)


# 1) Model Design ------------------------------------------------------------------------------
class LogRegression(nn.Module):

    def __init__(self, n_input_features):
        super(LogRegression, self).__init__()
        self.linear = nn.Linear(n_input_features, 1)
        # layer2
        # layer3 ...

    def forward(self, x):
        y_predicted = torch.sigmoid(self.linear(x))
        # y_predicted = layer2(y_predicted)
        # y_predicted = layer3(y_predicted) ...
        return y_predicted


model = LogRegression(n_features)

# 2) Construct Loss and Optimizer --------------------------------------------------------------
loss_fn = nn.BCELoss()
learning_rate = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# 3) Training Loop -----------------------------------------------------------------------------
num_epochs = 100
for epoch in range(num_epochs):
    # fwd pass
    y_hat = model(X_train)
    loss = loss_fn(y_hat, y_train)
    # bwd pass
    loss.backward()
    # update
    optimizer.step()
    # reset
    optimizer.zero_grad()

    if (epoch + 1) % 10 == 0:
        print(f'epoch: {epoch + 1}, loss = {loss.item():.4f}')

# 3) Model Evaluation --------------------------------------------------------------------------
with torch.no_grad():
    y_hat = model(X_test)
    y_hat_classes = y_hat.round()

    TP = ((y_hat_classes == 1) & (y_test == 1)).sum()
    FP = ((y_hat_classes == 1) & (y_test == 0)).sum()
    TN = ((y_hat_classes == 0) & (y_test == 0)).sum()
    FN = ((y_hat_classes == 0) & (y_test == 1)).sum()

    print('',
          f'TP = {TP}\t, FP = {FP}',
          f'FN = {FN}\t, TN = {TN}',
          sep='\n')

    acc = y_hat_classes.eq(y_test).sum() / float(y_test.shape[0])
    print(f'acc = {acc:4f}')
