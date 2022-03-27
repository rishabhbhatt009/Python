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
num_epochs = 10
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
# 2) Construct Loss Func and Optimizer --------------------------------------------------------------
# 3) Training Loop ----------------------------------------------------------------------------------
