# Design Philosophy
#
# 1) Design Model
#       - input size
#       - output size
#       - fwd pass
# 2) Construct Loss and Optimizer
# 3) Training Loop
#       - fwd pass: compute prediction and loss
#       - bwd pass: gradients
#       - update weights


import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

print('Import Completed', '-' * 50)

# 0) Data Prep ------------------------------------------------------------------------------------
# Generating Data
X_numpy, y_numpy = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=1)
print(f'Shape of the numpy data set X = {X_numpy.shape}, y = {y_numpy.shape}')
print(X_numpy[0:5], '\n', y_numpy[0:5])

# Creating Tensors and Reshaping
X = torch.from_numpy(X_numpy.astype(np.float32))
# .reshape(y_numpy.shape[0], 1)
y = torch.from_numpy(y_numpy.astype(np.float32))
y = y.view(y.shape[0], 1)
print(f'Shape of tensors X = {X_numpy.shape}, y = {y_numpy.shape}')
print(X[0:5], '\n', y[0:5])

n_samples, n_features = X.shape

# 1) Design Model ---------------------------------------------------------------------------------
input_size = n_features
output_size = 1
model = nn.Linear(in_features=n_features, out_features=1)

# 2) Construct Loss and Optimizer -----------------------------------------------------------------
loss_fn = nn.MSELoss()

learning_rate = 0.05
optimizer = torch.optim.SGD(params=model.parameters(), lr=learning_rate)

# 3) Training Loop --------------------------------------------------------------------------------
epochs = 100
for epoch in range(epochs):
    # fwd pass and loss
    y_hat = model(X)
    loss = loss_fn(y_hat, y)

    # bwd pass
    loss.backward()

    # update
    optimizer.step()
    optimizer.zero_grad()

    if (epoch + 1) % 10 == 0:
        print(f'epoch : {epoch + 1}, loss = {loss.item():.4f}')

# Plotting Results -------------------------------------------------------------------------------
predicted = model(X).detach().numpy()
plt.plot(X_numpy, y_numpy, 'ro')
plt.plot(X_numpy, predicted, 'b')
plt.show()
