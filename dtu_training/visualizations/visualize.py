import click
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import os
import sys
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))#

from models.model import Network  # Import your model class

@click.command()
@click.option("--model_checkpoint", default="models/trained_model.pth", help="Path to model checkpoint")
@click.option("--processed_dir", default="data/processed", help="Path to processed data directory")
@click.option("--figure_dir", default="reports/figures", help="Path to save figures")
@click.option("--figure_name", default="embeddings.png", help="Name of the figure")
def visualize(model_checkpoint: str, processed_dir: str, figure_dir: str, figure_name: str) -> None:
    """Visualize model predictions using t-SNE."""
    # Load the model and set it to evaluation mode
    input_size = 784  # Define according to your model's input size
    output_size = 10  # Define according to your model's output size
    hidden_layers = [512, 256, 128]  # Define according to your model's architecture
    model = Network(input_size, output_size, hidden_layers)
    model.load_state_dict(torch.load(model_checkpoint))
    model.eval()
    model.fc = torch.nn.Identity()  # Remove the final classification layer

    # Load the test images and targets
    test_images = torch.load(f"{processed_dir}/test_images.pt")
    test_target = torch.load(f"{processed_dir}/test_target.pt")
    test_dataset = torch.utils.data.TensorDataset(test_images, test_target)

    # Collect embeddings and targets
    embeddings, targets = [], []
    with torch.inference_mode():
        for batch in torch.utils.data.DataLoader(test_dataset, batch_size=32):
            images, target = batch

            # Reshape images to match input size
            images = images.view(images.size(0), -1)

            predictions = model(images)
            embeddings.append(predictions)
            targets.append(target)
        embeddings = torch.cat(embeddings).numpy()
        targets = torch.cat(targets).numpy()

    # Standardize the embeddings
    embeddings = (embeddings - np.mean(embeddings, axis=0)) / np.std(embeddings, axis=0)

    # Reduce dimensionality for large embeddings using PCA
    if embeddings.shape[1] > 50:
        pca = PCA(n_components=50)
        embeddings = pca.fit_transform(embeddings)

    # Apply t-SNE to reduce embeddings to 2D
    tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, n_iter=1000)
    embeddings = tsne.fit_transform(embeddings)

    # Create a scatter plot of the t-SNE embeddings
    plt.figure(figsize=(12, 8))
    for i in range(10):
        mask = targets == i
        plt.scatter(embeddings[mask, 0], embeddings[mask, 1], label=f'Class {i}', alpha=0.6, edgecolors='w', s=100)
    plt.legend(title="Classes")
    plt.title('t-SNE visualization of model embeddings')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.grid(True)

    # Save the figure
    plt.savefig(f"{figure_dir}/{figure_name}")
    plt.close()

if __name__ == "__main__":
    visualize()