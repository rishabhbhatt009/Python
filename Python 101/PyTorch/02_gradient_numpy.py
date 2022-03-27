import numpy as np

# Linear Regression -------------------------------------------------------------

# Data Prep
X = np.array([1, 2, 3, 4, 5], dtype=np.float)
Y = np.array([2, 4, 6, 8, 10], dtype=np.float)
wt = 0.0
print(f'Training Data : {X}', f'Testing Data : {Y}', sep='\n')


# Model Prediction
def forward(x):
    return wt * x


# Loss Calculation
# MSE = 1/N * (w*x -y)**2
def loss_mse(y, y_hat):
    return ((y_hat - y) ** 2).mean()


# Gradient Calculation
# dJ/dw = 1/N * 2x * (w*x - y)
def grad(x, y, y_predicted):
    return np.dot(2 * x, y_predicted - y).mean()


print(f'Prediction before training : f(5) = {forward(5):.3f}')

# Training
learning_rate = 0.01
iterations = 10

for epoch in range(iterations):
    # Prediction = fwd pass
    y_pred = forward(X)

    # calc Loss
    loss = loss_mse(Y, y_pred)

    # calc Gradient = bwd pass
    dw = grad(X, Y, y_pred)

    # Update weights
    wt -= learning_rate * dw

    if epoch % 1 == 0:
        print(f'Epoch {epoch + 1} : w = {wt:.3f}, loss = {loss:.8f}')

print(f'Prediction after training : f(5) = {forward(5):.3f}')
