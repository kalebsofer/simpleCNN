import torch
import torch.nn as nn
import torch.optim as optim

from .model import SimpleCNN
from .load_data import load_cifar10


trainloader, testloader = load_cifar10()

# Instantiate the model, define the loss function and the optimizer
net = SimpleCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

# Initialize lists to store metrics
train_acc = []
val_acc = []
train_loss = []
val_loss = []

def main():
    # Training the network
    epochs = 10
    for epoch in range(epochs):  
        net.train()  
        running_loss = 0.0
        correct = 0
        total = 0

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

            # Accumulate loss and accuracy
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        # Calculate and store training loss and accuracy
        epoch_loss = running_loss / len(trainloader)
        epoch_acc = 100 * correct / total
        train_loss.append(epoch_loss)
        train_acc.append(epoch_acc)

        # Validation
        net.eval()  # Set the network to evaluation mode
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for data in testloader:
                inputs, labels = data
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        # Calculate and store validation loss and accuracy
        val_epoch_loss = val_running_loss / len(testloader)
        val_epoch_acc = 100 * val_correct / val_total
        val_loss.append(val_epoch_loss)
        val_acc.append(val_epoch_acc)

        print(f'Epoch {epoch + 1}, Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.2f}%, Val Loss: {val_epoch_loss:.4f}, Val Acc: {val_epoch_acc:.2f}%')

    print('Finished Training')

    # Save the trained model and metrics
    PATH = './saved_models/cifar_net.pth'
    torch.save(net.state_dict(), PATH)
    torch.save({
        'train_loss': train_loss,
        'train_acc': train_acc,
        'val_loss': val_loss,
        'val_acc': val_acc
    }, './saved_models/metrics.pth')


if __name__ == '__main__':
    main()
