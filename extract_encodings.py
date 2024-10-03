# extract_encodings.py
import torch
from torch.utils.data import DataLoader
from dataset import MatrixDataset
from autoencoder import ConvAutoencoder

# Load dataset
inputs_file = 'all_inputs.npy'
outputs_file = 'all_outputs.npy'

dataset = MatrixDataset(inputs_file, outputs_file)
dataloader = DataLoader(dataset, batch_size=64, shuffle=False)

# Load the trained model
model = ConvAutoencoder()
model.load_state_dict(torch.load('autoencoder.pth'))
model.eval()

# Create a list to store encodings
encodings = []

with torch.no_grad():
    for batch_inputs, _ in dataloader:
        # Get encoded representations
        latent_vectors = model.encoder(batch_inputs)
        encodings.append(latent_vectors.numpy())

# Concatenate all encodings
encodings = np.concatenate(encodings, axis=0)

# Save encodings
np.save('encodings.npy', encodings)
print("Encodings saved as 'encodings.npy'")
