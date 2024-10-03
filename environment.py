import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os

def check_environment():
    print("Checking environment setup...")

    # Check PyTorch version
    print(f"PyTorch version: {torch.__version__}")

    # Check CUDA availability
    if torch.cuda.is_available():
        print("CUDA is available. GPU will be used for training.")
        print(f"GPU device: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA is not available. Training will be done on CPU.")

    # Check for required modules
    required_modules = ['torch', 'torch.nn', 'torch.optim', 'numpy']
    for module in required_modules:
        try:
            __import__(module)
            print(f"{module} is installed.")
        except ImportError:
            print(f"ERROR: {module} is not installed.")

    # Check if dataset files exist
    inputs_file = 'all_inputs.npy'
    outputs_file = 'all_outputs.npy'
    if os.path.exists(inputs_file) and os.path.exists(outputs_file):
        print("Dataset files found.")
        
        # Try loading the dataset
        try:
            inputs = np.load(inputs_file)
            outputs = np.load(outputs_file)
            print(f"Dataset loaded successfully. Input shape: {inputs.shape}, Output shape: {outputs.shape}")
        except Exception as e:
            print(f"ERROR: Failed to load dataset. Exception: {e}")
    else:
        print("ERROR: Dataset files not found.")

    # Try importing custom modules
    try:
        from dataset import MatrixDataset
        from autoencoder import ConvAutoencoder
        print("Custom modules (dataset and autoencoder) imported successfully.")
    except ImportError as e:
        print(f"ERROR: Failed to import custom modules. Exception: {e}")

    # Test model instantiation
    try:
        model = ConvAutoencoder()
        print("ConvAutoencoder model instantiated successfully.")
    except Exception as e:
        print(f"ERROR: Failed to instantiate ConvAutoencoder model. Exception: {e}")

    print("\nEnvironment check completed.")

if __name__ == "__main__":
    check_environment()