# !pip install torch torchvision torchaudio matplotlib
# !pip install matplotlib

import torch
import numpy as np

# Check PyTorch and CUDA installation
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

# Create a simple tensor and perform a CUDA operation
a = torch.tensor([1.0, 2.0, 3.0], device='cuda')
print("Tensor on GPU:", a)


import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.distributed as dist
import matplotlib.pyplot as plt

# Function to initialize the process group
def initialize_process_group():
    if not dist.is_initialized():
        # Set environment variables required for torch.distributed
        os.environ['RANK'] = '0'
        os.environ['WORLD_SIZE'] = '1'
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        dist.init_process_group(backend='gloo', init_method='env://')

# Initialize the process group
initialize_process_group()

# Import SpecTrain optimizer from the uploaded files
from spectrain import Spectrain

# Define a simple neural network
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(10, 50)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(50, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Create a dummy dataset
x = torch.randn(100, 10)
y = torch.randn(100, 1)
dataset = TensorDataset(x, y)
dataloader = DataLoader(dataset, batch_size=10, shuffle=True)

# Initialize the model and move it to GPU
model = SimpleModel().cuda()
criterion = nn.MSELoss().cuda()

# Initialize the SpecTrain optimizer
optimizer = Spectrain(model.parameters(), lr=0.01)

# Training loop with CUDA streams and speculative weight prediction
num_epochs = 5
losses = []

for epoch in range(num_epochs):
    epoch_loss = 0
    for inputs, targets in dataloader:
        inputs, targets = inputs.cuda(), targets.cuda()

        # Create CUDA streams for each part of the model
        stream1 = torch.cuda.Stream()
        stream2 = torch.cuda.Stream()

        with torch.cuda.stream(stream1):
            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass first part of the model
            outputs = model.fc1(inputs)
            outputs = model.relu(outputs)

        with torch.cuda.stream(stream2):
            # Forward pass second part of the model
            with torch.cuda.stream(stream1):
                outputs = outputs.cuda(non_blocking=True)
            outputs = model.fc2(outputs)

        # Synchronize streams
        torch.cuda.synchronize(stream1)
        torch.cuda.synchronize(stream2)

        # Calculate loss
        loss = criterion(outputs, targets)

        # Backward pass
        loss.backward()

        # Apply speculative weight prediction and optimizer step
        optimizer.step()

        # Accumulate loss
        epoch_loss += loss.item()

    # Average loss for the epoch
    avg_loss = epoch_loss / len(dataloader)
    losses.append(avg_loss)

    # Print loss for the epoch
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss}")

# Plot the training loss
plt.plot(range(1, num_epochs + 1), losses, marker='o')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.show()

print("Training complete.")
