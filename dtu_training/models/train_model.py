
import torch
from torch import nn
import os 
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))#

from models.model import Network


def load_data(train_data_path, test_data_path):
    """Load training and testing data from specified paths."""
    print(os.path.join(train_data_path, 'train_images.pt'))
    train_images = torch.load(os.path.join(train_data_path, 'train_images.pt'))
    train_targets = torch.load(os.path.join(train_data_path, 'train_target.pt'))
    test_images = torch.load(os.path.join(test_data_path, 'test_images.pt'))
    test_targets = torch.load(os.path.join(test_data_path, 'test_target.pt'))

    train_dataset = TensorDataset(train_images, train_targets)
    test_dataset = TensorDataset(test_images, test_targets)

    trainloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    testloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    return trainloader, testloader


def validation(model, testloader, criterion):
    """Validate the model on the testdata by calculating the sum of mean loss and mean accuracy for each test batch.

    Arguments:
        model: torch network
        testloader: torch.utils.data.DataLoader, dataloader of test set
        criterion: loss function
    """
    accuracy = 0
    test_loss = 0
    for images, labels in testloader:
        images = images.resize_(images.size()[0], 784)

        output = model.forward(images)
        test_loss += criterion(output, labels).item()

        ## Calculating the accuracy
        # Model's output is log-softmax, take exponential to get the probabilities
        ps = torch.exp(output)
        # Class with highest probability is our predicted class, compare with true label
        equality = labels.data == ps.max(1)[1]
        # Accuracy is number of correct predictions divided by all predictions, just take the mean
        accuracy += equality.type_as(torch.FloatTensor()).mean()

    return test_loss, accuracy

def train(model, trainloader, testloader, criterion, optimizer=None, epochs=5, print_every=40):
    """Train model."""
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    steps = 0
    running_loss = 0
    train_losses, test_losses, test_accuracies = [], [], []
    for e in range(epochs):
        # Model in training mode, dropout is on
        model.train()
        for images, labels in trainloader:
            steps += 1

            # Flatten images into a 784 long vector
            images.resize_(images.size()[0], 784)

            optimizer.zero_grad()

            output = model.forward(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                # Model in inference mode, dropout is off
                model.eval()

                # Turn off gradients for validation, will speed up inference
                with torch.no_grad():
                    test_loss, accuracy = validation(model, testloader, criterion)

                train_losses.append(running_loss / print_every)
                test_losses.append(test_loss / len(testloader))
                test_accuracies.append(accuracy / len(testloader))

                print(
                    f"Epoch: {e+1}/{epochs}.. ",
                    f"Training Loss: {running_loss/print_every:.3f}.. ",
                    f"Test Loss: {test_loss/len(testloader):.3f}.. ",
                    f"Test Accuracy: {accuracy/len(testloader):.3f}"
                )

                running_loss = 0

                # Make sure dropout and grads are on for training
                model.train()


    # Save the trained model
    # model_path = os.path.join('models', 'trained_model.pth')
    model_path = os.path.join(os.path.dirname(__file__),'..', '..','models','trained_model.pth')
    torch.save(model.state_dict(), model_path)

    # Save the training statistics/visualizations
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.plot(test_accuracies, label='Test Accuracy')
    plt.legend(frameon=False)
    plt.xlabel('Epochs')
    plt.ylabel('Loss/Accuracy')
    plt.title('Training and Validation Metrics')
    # figure_path = os.path.join('reports', 'figures', 'training_curve.png')
    figure_path = os.path.join(os.path.dirname(__file__),'..', '..', 'reports', 'figures', 'training_curve.png')

    plt.savefig(figure_path)                


# Example usage:
# Example usage:
if __name__ == "__main__":
    train_data_path = os.path.abspath(os.path.join(os.path.dirname(__file__),'..', '..', 'data', 'processed'))
    test_data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..', 'data', 'processed'))

    print(os.path.join(train_data_path, 'train_images.pt'))  # Print the path to verify it

    trainloader, testloader = load_data(train_data_path, test_data_path)


    input_size = 784
    output_size = 10
    hidden_layers = [512, 256, 128]

    model = Network(input_size, output_size, hidden_layers)

    criterion = nn.NLLLoss()  # Replace with your loss function

    train(model, trainloader, testloader, criterion)