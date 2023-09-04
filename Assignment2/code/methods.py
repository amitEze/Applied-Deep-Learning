import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# Setup device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
input_size = 784
hidden_size = 500
num_classes = 10
num_epochs = 5
batch_size = 100
learning_rate = 0.001

# Define subset sizes
train_subset_size = 6000
test_subset_size = 6000

# MNIST dataset
full_train_dataset = torchvision.datasets.MNIST(root='./data/',
                                                train=True,
                                                transform=transforms.ToTensor(),
                                                download=True)
full_test_dataset = torchvision.datasets.MNIST(root='./data/',
                                               train=False,
                                               transform=transforms.ToTensor())

def load_data():
    # Create subsets of train and test datasets
    train_subset = torch.utils.data.Subset(full_train_dataset, range(train_subset_size))
    test_subset = torch.utils.data.Subset(full_test_dataset, range(test_subset_size))

    # Data loaders
    train_loader = torch.utils.data.DataLoader(dataset=train_subset,
                                            batch_size=batch_size,
                                            shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_subset,
                                            batch_size=batch_size,
                                            shuffle=False)
    return train_loader, test_loader


from torch.utils.data import random_split

def load_data_with_validation(batch_size=batch_size):
    # Create subsets of train and test datasets
    train_subset = torch.utils.data.Subset(full_train_dataset, range(train_subset_size))
    test_subset = torch.utils.data.Subset(full_test_dataset, range(test_subset_size))
    
    train_dataset, val_dataset = random_split(train_subset, [5000, 1000])

    # Data loaders
    train_loader = torch.utils.data.DataLoader(dataset=train_subset,
                                            batch_size=batch_size,
                                            shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_subset,
                                            batch_size=batch_size,
                                            shuffle=False)
    
    validation_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                            batch_size=batch_size,
                                            shuffle=True)
    
    return train_loader, test_loader, validation_loader






# Fully connected neural network
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size) 
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)  

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out
    
    
# eval loader loss according to model
def loader_eval(loader, model, criterion):
    set_loss = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.reshape(-1, input_size).to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            set_loss += loss.item()

    set_loss /= len(loader)
    return set_loss