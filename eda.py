import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

def load_data(file_path):
    return np.load(file_path, allow_pickle=True)

def analyze_data(data):
    print(f"Total number of samples: {len(data)}")
    
    # Count train and test samples
    type_counts = Counter(item['type'] for item in data)
    print(f"Number of training samples: {type_counts['train']}")
    print(f"Number of test samples: {type_counts['test']}")
    
    # Analyze shapes
    input_shapes = [item['data']['input'].shape for item in data]
    output_shapes = [item['data']['output'].shape for item in data]
    
    print(f"\nInput shape: {input_shapes[0]}")
    print(f"Output shape: {output_shapes[0]}")
    
    # Check if all shapes are consistent
    if len(set(input_shapes)) == 1 and len(set(output_shapes)) == 1:
        print("All input and output shapes are consistent.")
    else:
        print("Warning: Inconsistent shapes detected.")
    
    # Analyze value ranges
    all_input_values = np.concatenate([item['data']['input'].flatten() for item in data])
    all_output_values = np.concatenate([item['data']['output'].flatten() for item in data])
    
    print(f"\nInput value range: [{all_input_values.min()}, {all_input_values.max()}]")
    print(f"Output value range: [{all_output_values.min()}, {all_output_values.max()}]")
    
    # Visualize data distribution
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.hist(all_input_values, bins=50, alpha=0.7)
    plt.title("Distribution of Input Values")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    
    plt.subplot(1, 2, 2)
    plt.hist(all_output_values, bins=50, alpha=0.7)
    plt.title("Distribution of Output Values")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    
    plt.tight_layout()
    plt.savefig('value_distribution.png')
    print("Value distribution plot saved as 'value_distribution.png'")
    
    # Analyze sparsity
    input_sparsity = (all_input_values == 0).mean()
    output_sparsity = (all_output_values == 0).mean()
    
    print(f"\nInput sparsity (fraction of zero values): {input_sparsity:.2%}")
    print(f"Output sparsity (fraction of zero values): {output_sparsity:.2%}")
    
    # Sample visualization
    sample_input = data[0]['data']['input'][0]
    sample_output = data[0]['data']['output'][0]
    
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.imshow(sample_input, cmap='viridis')
    plt.title("Sample Input")
    plt.colorbar()
    
    plt.subplot(1, 2, 2)
    plt.imshow(sample_output, cmap='viridis')
    plt.title("Sample Output")
    plt.colorbar()
    
    plt.tight_layout()
    plt.savefig('sample_visualization.png')
    print("Sample visualization saved as 'sample_visualization.png'")

if __name__ == "__main__":
    file_path = '/root/ARC-AGI/processed_data.npy'
    data = load_data(file_path)
    analyze_data(data)