# train_autoencoder.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import MatrixDataset
from autoencoder import ConvAutoencoder

# Load dataset
inputs_file = 'all_inputs.npy'
outputs_file = 'all_outputs.npy'

dataset = MatrixDataset(inputs_file, outputs_file)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# Initialize model, loss function, and optimizer
model = ConvAutoencoder()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Training loop
num_epochs = 20

for epoch in range(num_epochs):
    for batch_inputs, _ in dataloader:
        # Move data to GPU if available
        batch_inputs = batch_inputs.cuda()
        model = model.cuda()

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(batch_inputs)

        # Compute loss
        loss = criterion(outputs, batch_inputs)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# Save the trained model
torch.save(model.state_dict(), 'autoencoder.pth')
