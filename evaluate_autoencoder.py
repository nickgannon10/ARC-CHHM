# evaluate_autoencoder.py
import torch
import matplotlib.pyplot as plt
from dataset import MatrixDataset
from autoencoder import ConvAutoencoder

# Load dataset
inputs_file = 'all_inputs.npy'
outputs_file = 'all_outputs.npy'

dataset = MatrixDataset(inputs_file, outputs_file)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

# Load the trained model
model = ConvAutoencoder()
model.load_state_dict(torch.load('autoencoder.pth'))
model.eval()

# Get a sample input
with torch.no_grad():
    for batch_inputs, _ in dataloader:
        # Forward pass
        outputs = model(batch_inputs)
        break  # Only need one sample

# Visualize the original and reconstructed images
original = batch_inputs[0].squeeze().numpy()
reconstructed = outputs[0].squeeze().numpy()

plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.imshow(original, cmap='viridis')
plt.title('Original Input')
plt.colorbar()

plt.subplot(1, 2, 2)
plt.imshow(reconstructed, cmap='viridis')
plt.title('Reconstructed Output')
plt.colorbar()

plt.tight_layout()
plt.savefig('autoencoder_reconstruction.png')
print("Reconstruction saved as 'autoencoder_reconstruction.png'")
