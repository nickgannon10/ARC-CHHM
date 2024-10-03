import torch
import json
import os
import numpy as np

def normalize_and_pad(data, max_rows=30, max_cols=30, pad_value=0):
    # Convert to PyTorch tensor if it's not already
    if not isinstance(data, torch.Tensor):
        data = torch.tensor(data, dtype=torch.float32)

    # Normalize data to range [0, 1]
    data = data / 9.0  # Since all values are between 0-9

    # Handle both 2D and 3D tensors
    if data.dim() == 2:
        data = data.unsqueeze(0)  # Add batch dimension
    elif data.dim() != 3:
        raise ValueError("Input must be a 2D or 3D tensor")

    batch_size, rows, cols = data.shape

    # Create padded tensor
    padded_data = torch.full((batch_size, max_rows, max_cols), pad_value, dtype=torch.float32)

    # Copy data into padded tensor
    padded_data[:, :rows, :cols] = data

    return padded_data

def process_file(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)

    train_data = data['train']
    test_data = data['test']

    processed_train = []
    for item in train_data:
        processed_train.append({
            'input': normalize_and_pad(item['input']).numpy(),
            'output': normalize_and_pad(item['output']).numpy()
        })

    processed_test = []
    for item in test_data:
        processed_test.append({
            'input': normalize_and_pad(item['input']).numpy(),
            'output': normalize_and_pad(item['output']).numpy()
        })

    return processed_train, processed_test

def process_directory(directory_path):
    all_inputs = []
    all_outputs = []

    for filename in os.listdir(directory_path):
        if filename.endswith('.json'):
            file_path = os.path.join(directory_path, filename)
            processed_train, processed_test = process_file(file_path)
            
            for item in processed_train:
                all_inputs.append(item['input'])
                all_outputs.append(item['output'])
            for item in processed_test:
                all_inputs.append(item['input'])
                all_outputs.append(item['output'])

    # Convert lists to NumPy arrays
    all_inputs = np.concatenate(all_inputs, axis=0)
    all_outputs = np.concatenate(all_outputs, axis=0)

    # Save the inputs and outputs separately
    np.save('all_inputs.npy', all_inputs)
    np.save('all_outputs.npy', all_outputs)

    print(f"Processed {len(all_inputs)} total samples.")
    print("Data saved to 'all_inputs.npy' and 'all_outputs.npy'")

# Example usage
directory_path = '/root/ARC-AGI/data/training/'

process_directory(directory_path)
