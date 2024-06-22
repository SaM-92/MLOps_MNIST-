import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from PIL import Image
import numpy as np
import pickle
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))#

from models.model import Network  # Import your model class

def load_model(model_path):
    """Load a pre-trained model from a file."""
    input_size = 784  # Define according to your model's input size
    output_size = 10  # Define according to your model's output size
    hidden_layers = [512, 256, 128]  # Define according to your model's architecture
    model = Network(input_size, output_size, hidden_layers)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()  # Set the model to evaluation mode
    return model

def load_images_from_folder(folder):
    """Load images from a folder and transform them to tensors."""
    transform = transforms.Compose([transforms.Grayscale(), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    images = []
    for filename in os.listdir(folder):
        if filename.endswith('.png') or filename.endswith('.jpg'):
            img_path = os.path.join(folder, filename)
            image = Image.open(img_path)
            image = transform(image)
            image = image.view(1, 784)  # Flatten the image
            images.append(image)
    return torch.cat(images)

def load_images_from_file(file_path):
    """Load images from a numpy, pickle, or pt file."""
    if file_path.endswith('.npy'):
        images = np.load(file_path)
    elif file_path.endswith('.pkl'):
        with open(file_path, 'rb') as f:
            images = pickle.load(f)
    elif file_path.endswith('.pt'):
        images = torch.load(file_path)
    else:
        raise ValueError("Unsupported file format: {}".format(file_path))
    
    if not isinstance(images, torch.Tensor):
        images = torch.tensor(images, dtype=torch.float32)
    
    images = (images - 0.5) / 0.5  # Normalize
    return images.view(images.size(0), -1)  # Flatten the images

def predict(model, dataloader):
    """Run prediction for a given model and dataloader."""
    all_preds = []
    with torch.no_grad():
        for data in dataloader:
            data = data[0]  # Extract data from the TensorDataset
            outputs = model(data)
            _, preds = torch.max(outputs, 1)
            all_preds.append(preds)
    return torch.cat(all_preds)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python predict_model.py <model_path> <data_path>")
        sys.exit(1)

    model_path = sys.argv[1]
    data_path = sys.argv[2]

    # Load the pre-trained model
    model = load_model(model_path)

    # Load the data
    if os.path.isdir(data_path):
        images = load_images_from_folder(data_path)
    else:
        images = load_images_from_file(data_path)

    # Create a dataloader
    dataset = TensorDataset(images)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False)

    # Run prediction
    predictions = predict(model, dataloader)

    print("Predictions:", predictions.numpy())
