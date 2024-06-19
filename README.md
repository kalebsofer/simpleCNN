# SimpleCNN Project

## Contents
- [SimpleCNN Project](#simplecnn-project)
  - [Contents](#contents)
  - [Overview](#overview)
  - [The network architecture](#the-network-architecture)
    - [Forward Pass:](#forward-pass)
    - [Summary:](#summary)
  - [Running locally](#running-locally)
    - [Prerequisites](#prerequisites)
    - [Installation](#installation)
    - [Usage](#usage)

---

## Overview

Welcome to SimpleCNN, a project for training and evaluating a Convolutional Neural Network (CNN) on the CIFAR-10 dataset using PyTorch.

SimpleCNN utilizes a CNN architecture to classify images from the CIFAR-10 dataset into 10 different classes. The project includes modules for loading data, defining the model architecture, training the model, and visualizing results.


---
## The network architecture
[See model.py](model/model.py)
1. **Input Layer:**
   - Input images are RGB with 3 channels.

2. **First Convolutional Layer (`self.conv1`):**
   - Applies a convolutional operation with a 3x3 kernel.
   - Input: 3 channels (RGB)
   - Output: 32 channels
   - Padding: 1 pixel on all sides to maintain the spatial dimensions after convolution.
   - Activation Function: Rectified Linear Unit (ReLU) applied after convolution.
   - Pooling: Average pooling with a 2x2 kernel and stride of 2 reduces the spatial dimensions by half.

3. **Second Convolutional Layer (`self.conv2`):**
   - Applies a convolutional operation with a 3x3 kernel.
   - Input: 32 channels (from the previous layer)
   - Output: 64 channels
   - Padding: 1 pixel on all sides to maintain the spatial dimensions.
   - Activation Function: ReLU applied after convolution.
   - Pooling: Average pooling with a 2x2 kernel and stride of 2 reduces the spatial dimensions by half.

4. **Flattening (`x.view(-1, 64 * 8 * 8)`):**
   - The output from the second convolutional layer is flattened to be fed into fully connected layers.
   - `64 * 8 * 8` comes from the 64 channels output from `conv2` and the spatial dimensions after two pooling layers.

5. **First Fully Connected Layer (`self.fc1`):**
   - Linear transformation with 64 * 8 * 8 inputs (flattened size) and 512 outputs.
   - Activation Function: ReLU applied after the linear transformation.

6. **Second Fully Connected Layer (`self.fc2`):**
   - Linear transformation with 512 inputs and 10 outputs.
   - Output Layer: Produces logits for 10 classes in the CIFAR-10 dataset.

### Forward Pass:
- `forward(self, x)`: Defines the sequence of operations for a forward pass through the network.
  - Applies `self.conv1` followed by ReLU activation and average pooling (`self.pool`).
  - Applies `self.conv2` followed by ReLU activation and average pooling (`self.pool`).
  - Flattens the output and passes through `self.fc1` followed by ReLU activation.
  - Computes the final logits through `self.fc2`.

### Summary:
- **Layers**: Two convolutional layers (`conv1` and `conv2`), each followed by ReLU activation and average pooling.
- **Fully Connected Layers**: Two fully connected layers (`fc1` and `fc2`) for final classification.
- **Activation Function**: ReLU is used after each convolutional and fully connected layer.
- **Pooling**: Average pooling with a 2x2 kernel and stride of 2 after each convolutional layer to reduce spatial dimensions.




---

## Running locally

To run this project locally:

### Prerequisites

- Python 3.12.4 or higher
- PyTorch 2.3.1 or higher
- Jupyter Notebook (for visualizing results)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/simpleCNN.git
   cd simpleCNN
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
### Usage
1. If you want to edit and retrain the model, remove existing files in saved_models/
   ```bash
   rm -rf saved_models/*
2. Execute run_project.py in scripts/:
    ```bash
    python scripts/run_project.py
