import math
import torch
import torchvision.transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np


# Dataset Class -----------------------------------------------------------------------------------
class WineDataset(Dataset):

    def __init__(self, transform=None):
        data = np.loadtxt('data/wine/wine.csv', delimiter=',', dtype=np.float32, skiprows=1)
        self.x = data[:, 1:]
        self.y = data[:, [0]]
        self.n_samples = self.x.shape[0]
        self.transform = transform

    def __getitem__(self, item):
        sample = self.x[item], self.y[item]
        if self.transform:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return self.n_samples


# Custom Transform Class --------------------------------------------------------------------------
class ToTensor:
    def __call__(self, sample):
        inputs, targets = sample
        # return torch.from_numpy(inputs), torch.from_numpy(targets)
        return torch.tensor(inputs), torch.tensor(targets)


class MulTransform:
    def __init__(self, factor):
        self.factor = factor

    def __call__(self, sample):
        inputs, targets = sample
        inputs *= self.factor
        return inputs, targets


# dataset = WineDataset(transform=None)
composed = torchvision.transforms.Compose([ToTensor(), MulTransform(1)])
dataset = WineDataset(transform=composed)

features, labels = dataset[0]
print(f'Data Shape : {features.shape}, {labels.shape}',
      f'Data Type : {type(features)}, {type(labels)}',
      f'Number of Samples : {len(dataset)}',
      f'X : {features}',
      f'y : {labels}',
      sep='\n')
batch_size = 5
dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

print(f'\n\nNumber of samples in data {len(dataset)}\n')
for i, (X, y) in enumerate(dataloader):
    print(f'{i+1}. X = {X.size()}, y = {y.size()}')

# Training Loop ---------------------------------------------------------------------------------
num_epochs = 2
total_samples = len(dataset)
n_iter = math.ceil(total_samples / batch_size)

for epoch in range(num_epochs):
    for i, (inputs, labels) in enumerate(dataloader):
        # fwd
        # bwd
        # update
        print(f'epoch {epoch + 1}/{num_epochs}, step {i}/{n_iter}, input = {inputs.shape}')
