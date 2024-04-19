# Python-PyTorch

This repository contains examples and tutorials for using PyTorch, a popular deep learning library developed by Facebook's AI Research lab (FAIR). PyTorch provides an excellent platform for building deep learning models with its easy-to-use API and dynamic computational graph.

## Prerequisites

Before running the examples, ensure you have Python 3.6 or higher installed on your machine.

## Installation

1. It's recommended to create a virtual environment for Python projects to manage dependencies efficiently:

   ```bash
   python -m venv pytorch-env
   source pytorch-env/bin/activate  # On Windows use `pytorch-env\Scripts\activate`
   ```

2. Install PyTorch by running:

   ```bash
   pip install torch torchvision
   ```

   Visit the [official PyTorch website](https://pytorch.org/get-started/locally/) for installation commands specific to your platform.

## Example - Basic Neural Network

Here is a simple example of a neural network that classifies digits from the MNIST dataset.

### `nn_example.py`

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.log_softmax(self.fc3(x), dim=1)

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data.view(-1, 784))
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=64, shuffle=True)

model = SimpleNet().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

# Training loop
for epoch in range(1, 2):
    train(model, device, train_loader, optimizer, epoch)
```

## Contributing

Contributions are welcome! Please fork the repository, add your contributions, and submit a pull request.

## License

This project is released under the MIT License. See the included LICENSE file for more details.
