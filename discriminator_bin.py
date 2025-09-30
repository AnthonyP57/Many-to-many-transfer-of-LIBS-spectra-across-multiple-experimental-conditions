import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Define the discriminator model
class Discriminator(nn.Module):
    def __init__(self, input_size, num_filters):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(input_size, num_filters)
        self.leaky_relu1 = nn.LeakyReLU(0.2)
        self.fc2 = nn.Linear(num_filters, num_filters * 2)
        self.leaky_relu2 = nn.LeakyReLU(0.2)
        self.fc3 = nn.Linear(num_filters * 2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.leaky_relu1(self.fc1(x))
        x = self.leaky_relu2(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

# Define the dataset and data loader
class Dataset(torch.utils.data.Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

# Define the loss function and optimizer
def train(discriminator, dataset, batch_size, epochs, learning_rate):
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(discriminator.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        for i, (data, labels) in enumerate(data_loader):
            # Forward pass
            outputs = discriminator(data)
            loss = criterion(outputs, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Print loss at each iteration
            print(f'Epoch {epoch+1}, Iteration {i+1}, Loss: {loss.item()}')

        # Print loss at each epoch
        print(f'Epoch {epoch+1}, Loss: {loss.item()}')

# Example usage:
input_size = 100
num_filters = 64
batch_size = 32
epochs = 10
learning_rate = 0.001

# Create a discriminator model
discriminator = Discriminator(input_size, num_filters)

# Create a dataset and data loader
data = torch.randn(1000, input_size)
labels = torch.ones(1000,1)
dataset = Dataset(data, labels)

# Train the discriminator
train(discriminator, dataset, batch_size, epochs, learning_rate)