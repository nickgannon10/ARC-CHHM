# dataset.py
import torch
from torch.utils.data import Dataset

class MatrixDataset(Dataset):
    def __init__(self, inputs_file, outputs_file, transform=None):
        self.inputs = np.load(inputs_file)
        self.outputs = np.load(outputs_file)
        self.transform = transform

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_sample = self.inputs[idx]
        output_sample = self.outputs[idx]
        
        if self.transform:
            input_sample = self.transform(input_sample)
            output_sample = self.transform(output_sample)
        
        return torch.tensor(input_sample, dtype=torch.float32), torch.tensor(output_sample, dtype=torch.float32)
