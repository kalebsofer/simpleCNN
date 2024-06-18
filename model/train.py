import torch
import torch.nn as nn
import torch.optim as optim

from .model import SimpleCNN
from .load_data import load_cifar10


# Load CIFAR-10 dataset
trainloader, testloader = load_cifar10()

# Instantiate the model, define the loss function and the optimizer
net = SimpleCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)


def main():
    # Training the network
    epochs = 10
    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # Get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Print statistics
            running_loss += loss.item()
            if i % 100 == 99:  # Print every 100 mini-batches
                print(f'Epoch {epoch + 1}, Batch {i + 1}: Loss {running_loss / 100:.3f}')
                running_loss = 0.0

    print('Finished Training')

    PATH = './saved_models/cifar_net.pth'
    torch.save(net.state_dict(), PATH)


if __name__ == '__main__':
    main()
