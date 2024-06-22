import torch
import numpy as np

print("NumPy version:", np.__version__)
print("PyTorch version:", torch.__version__)

# Create a PyTorch tensor and convert to NumPy array
tensor = torch.tensor([1, 2, 3])
numpy_array = tensor.numpy()
print("Tensor:", tensor)
print("NumPy array:", numpy_array)
