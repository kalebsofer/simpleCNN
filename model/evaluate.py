import torch
from .model import SimpleCNN
from .load_data import load_cifar10

# Load CIFAR-10 dataset
trainloader, testloader = load_cifar10()

# Instantiate the model and load the trained weights
net = SimpleCNN()
PATH = '../saved_models/cifar_net.pth'
net.load_state_dict(torch.load(PATH))
net.eval()

# Evaluate the network on the test data
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the 10000 test images: {100 * correct / total:.2f}%')
