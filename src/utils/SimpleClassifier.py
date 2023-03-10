import torch
import torch.nn as nn

class SimpleNeuralNetwork(nn.Module):
    def __init__(self, in_size, out_size):
        super(SimpleNeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(in_size, 1024)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(1024, out_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
