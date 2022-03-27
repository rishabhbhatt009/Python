import torch
import torch.nn as nn

# Linear Regression from numpy to torch
#
# Step - 1  Replaced gradient calculation with Autograd
#           - used backward() with requires_grad = True
#           - weight update with torch.no_grad()
#           - reset/zero grad after each epoch
#
# Step - 2  Replaced loss calculation with PyTorch Loss
#           - added a loss function
#
# Step - 3  Replaced param updates with PyTorch Optimizer
#           - added an optimizer
#           - weight update with optimizer.step()
#           - grad reset/zero with optimizer.zero_grad()
#
# Step - 4  Replaced forward fun with a PyTorch Model
#           - added a model (single layer model)
#           - drop weights - replace with model.parameters()
#           - modified the shape of input (X,Y)
#           - added n_samples, n_features, X_Test
#           - replaced forward() with model
#
# Step - 5  Creating custom Model
#           - create model class
#           - __init__ : define layers
#           - define forward ()
# -------------------------------------------------------------

# Data Prep
X = torch.tensor([[1], [2], [3], [4], [5]], dtype=torch.float32)
Y = torch.tensor([[2], [4], [6], [8], [10]], dtype=torch.float32)
X_Test = torch.tensor([5], dtype=torch.float32)

# wt = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)

n_samples, n_features = X.shape

print(f'Training Data : {X}',
      f'Number of sample : {n_samples} ',
      f'Number of features : {n_features} ',
      f'Testing Data : {Y}',
      sep='\n')

# Model Prediction
input_size = n_features
output_size = n_features


class LinearRegression(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(LinearRegression, self).__init__()
        # define Layer
        self.lin = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.lin(x)


# model = nn.Linear(input_size, output_size)
model = LinearRegression(input_size, output_size)

print(f'\nPrediction before training : f(5) = {model(X_Test).item():.3f}\n')

# Training
learning_rate = 0.01
iterations = 50

loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for epoch in range(iterations):
    # Prediction = fwd pass 
    y_pred = model(X)

    # calc Loss
    loss = loss_fn(Y, y_pred)

    # calc Gradient = bwd pass
    loss.backward()

    # Update weights
    # with torch.no_grad():
    #     wt -= learning_rate * wt.grad
    optimizer.step()

    # zero gradients
    # wt.grad.zero_()
    optimizer.zero_grad()

    if epoch % 5 == 0:
        [wt, b] = model.parameters()
        print(f'Epoch {epoch + 1} : w = {wt[0][0]:.3f}, loss = {loss:.8f}')

print(f'\nPrediction after training : f(5) = {model(X_Test)}')
