# Design Philosophy
#
# 0) Data Prep
# 1) Design Model
#       - input size -> hidden size -> output size
#       - forward pass
# 2) Construct Loss Function and Optimizer
# 3) Training Loop
#       - forward pass : compute predictions and loss
#       - backward pass : compute gradients
#       - update wrights and reset gradients
# 4) Evaluate Model

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# to download the data set from CIFAR
# import ssl
# ssl._create_default_https_context = ssl._create_unverified_context


# Device Config -----------------------------------------------------------------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device = {str(device)}')

# Hyper-parameters --------------------------------------------------------------------------------------
num_epochs = 4
batch_size = 10
learning_rate = 0.001

# Data Prep ---------------------------------------------------------------------------------------------
# dataset has PILImage images of range [0, 1]
# we transform them to Tensor of normalized range [-1, 1]

# Create Transform
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                ])
train_dataset = torchvision.datasets.CIFAR10(root='data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.CIFAR10(root='data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# Model Design ------------------------------------------------------------------------------------------
class ConvNet(nn.Module):
    def __init__(self, num_classes):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        out = self.pool(self.relu(self.conv1(x)))
        out = self.pool(self.relu(self.conv2(out)))
        out = out.view(-1, 16 * 5 * 5)
        out = self.relu(self.fc1(out))
        out = self.relu(self.fc2(out))
        out = self.fc3(out)

        return out


model = ConvNet(10).to(device)
# Construct Loss Func and Optimizer ---------------------------------------------------------------------
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Training Loop -----------------------------------------------------------------------------------------
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # forward pass + compute loss
        outputs = model(images)
        loss = loss_func(outputs, labels)

        # backward pass
        loss.backward()

        # update weights + reset grad
        optimizer.step()
        optimizer.zero_grad()

        if (i + 1) % 100 == 0:
            pp = '{:<20}{:<20}{:>20}'
            print(pp.format(f'Epoch {epoch + 1}/{num_epochs},',
                            f'Step {i + 1}/{len(train_loader)},',
                            f'loss = {loss.item():.4f}')
                  )

print(f'{"Finished Training":-<100}\n')

# Saving and Loading Model --------------------------------------
PATH = 'cnn'

# Method 1 (Lazy Method)
# Serialised model is bound to the classes and the dir structure
torch.save(model, PATH)
model = torch.load(PATH)
model.eval()

# Method 2 (Recommended Method)
# Saves the parameters
torch.save(model.state_dict(), PATH)
# Create model again with parameters
# model = Model(*args, **kwargs)
model.load_state_dict(torch.load(PATH))
model.eval()

# Model Evaluation --------------------------------------------------------------------------------------
print(f'{"Model Evaluation":-<100}')
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    n_class_correct = [0] * 10
    n_class_samples = [0] * 10

    for i, (images, labels) in enumerate(test_loader):

        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)

        # value, index
        _, predictions = torch.max(outputs, 1)
        n_samples += labels.shape[0]
        n_correct += (predictions == labels).sum().item()

        for i in range(batch_size):
            label = labels[i]
            pred = predictions[i]
            if label == pred:
                n_class_correct[label] += 1
            n_class_samples[label] += 1

    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy = {acc}')

    for i in range(10):
        acc = 100.0 * n_class_correct[i] / n_class_samples[i]
        print(f'Accuracy of {classes[i]}: {acc} %')


# Display images from test loader
def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# get some random training images
dataiter = iter(test_loader)
images, labels = dataiter.next()

# show images
imshow(torchvision.utils.make_grid(images))
