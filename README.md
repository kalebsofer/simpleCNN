# SimpleCNN Project

## Contents
- [SimpleCNN Project](#simplecnn-project)
  - [Contents](#contents)
  - [Overview](#overview)
  - [Running locally](#running-locally)
    - [Prerequisites](#prerequisites)
    - [Installation](#installation)
    - [Usage](#usage)

---

## Overview

Welcome to SimpleCNN, a project for training and evaluating a Convolutional Neural Network (CNN) on the CIFAR-10 dataset using PyTorch.

SimpleCNN utilizes a CNN architecture to classify images from the CIFAR-10 dataset into 10 different classes. The project includes modules for loading data, defining the model architecture, training the model, and visualizing results.


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
