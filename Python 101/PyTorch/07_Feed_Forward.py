# Design Philosophy
#
# 0) Data Prep
#       - Data Loader
#       - Data Transformation
# 1) Model Design
# 2) Construct Loss Func and Optimizer
# 3) Training Loop
#       - fwd pass : compute prediction and loss
#       - bwd pass : compute gradients
#       - update + reset : update weights and empty gradients
# 4) Model Evaluation

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

# Device Configuration ------------------------------------------------------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'device = {device}')

# Define HyperParameters ----------------------------------------------------------------------------
input_size = 784  # images have dim : 28x28 = 784
hidden_size = 100
num_classes = 10
num_epochs = 1
batch_size = 100
learning_rate = 0.001

# 0) Data Prep --------------------------------------------------------------------------------------

# MNIST DATA
training_data = torchvision.datasets.MNIST(root='data', train=True,
                                           transform=transforms.ToTensor(),
                                           download=True
                                           )
testing_data = torchvision.datasets.MNIST(root='data', train=False,
                                          transform=transforms.ToTensor(),
                                          download=True
                                          )
# DataLoader
train_loader = torch.utils.data.DataLoader(dataset=training_data,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=testing_data,
                                          batch_size=batch_size,
                                          shuffle=False)


# 1) Model Design -----------------------------------------------------------------------------------
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.input_size = input_size
        self.l1 = nn.Linear(in_features=input_size, out_features=hidden_size)
        self.l2 = nn.ReLU()
        self.l3 = nn.Linear(in_features=hidden_size, out_features=num_classes)

    def forward(self, x):
        out = self.l1(x)
        out = self.l2(out)
        out = self.l3(out)
        # the cross entropy loss applies softmax for us, so we don't need to
        return out


model = NeuralNet(input_size, hidden_size, num_classes).to(device)

# 2) Construct Loss Func and Optimizer --------------------------------------------------------------
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 3) Training Loop ----------------------------------------------------------------------------------
for epoch in range(num_epochs):
    # Looping over batches
    for i, (images, labels) in enumerate(train_loader):
        images = images.reshape(-1, 28 * 28).to(device)
        labels = labels.to(device)

        # forward pass + compute loss
        outputs = model(images)
        loss = loss_func(outputs, labels)

        # calc gradients
        loss.backward()

        # update weights + reset gradient
        optimizer.step()
        optimizer.zero_grad()

        if (i + 1) % 100 == 0:
            print(f'Epoch {epoch + 1}/{num_epochs},\t Step {i + 1}/{len(train_loader)},\t loss = {loss.item():.4f}')

# 3) Model Evaluation -------------------------------------------------------------------------------
print('{"Model Evaluation":-<100}')
with torch.no_grad():
    n_correct = 0
    n_samples = 0

    for i, (images, labels) in enumerate(test_loader):
        if i == 0:
            print(images.shape, type(images))
            img = images.permute(0, 2, 3, 1)[0]

        images = images.reshape(-1, 28 * 28).to(device)
        labels = labels.to(device)
        outputs = model(images)

        if i == 0:
            print(outputs[0])
            _, prediction = torch.max(outputs, 1)
            print(_[0], prediction[0])
            plt.imshow(img, cmap='gray')
            plt.show(block=False)

        # value, index
        _, predictions = torch.max(outputs, 1)
        n_samples += labels.shape[0]
        n_correct += (predictions == labels).sum().item()

    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy = {acc}')

# So that window doesn't close
plt.show()