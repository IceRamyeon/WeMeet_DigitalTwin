# %% 1. importing necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# %% 2. loading the MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)

testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=64, shuffle=False)

# %% 3. exploring the dataset
train_features, train_labels = next(iter(trainloader))
print("The shape of the training features is:", train_features.shape)
print("The shape of the training labels is:", train_labels.shape)

# %% 4.
test_features, test_labels = next(iter(testloader))
print("The shape of the testing features is:", test_features.shape)
print("The shape of the testing labels is:", test_labels.shape)

# %% 5. visualizing some samples from the dataset
img = train_features[5]
img.shape
torch.Size([1, 28, 28])

plt.figure(figsize=(4, 4))
plt.imshow(img.squeeze(0), 'gray')
plt.xticks([])
plt.yticks([])
plt.show()

train_labels[5]