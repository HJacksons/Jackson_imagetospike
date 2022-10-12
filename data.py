from pkgutil import get_loader
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from snntorch import spikegen


class Encoder:

    def __init__(self, batch_size=128):
        # Init batch size
        self.batch_size = batch_size

        # Define a transform
        self.transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0,), (1,))])

        # Download training data
        self.mnist_train = datasets.MNIST(root="../data", train=True, download=True, transform=self.transform)

        # Download test data
        self.mnist_test = datasets.MNIST(root="../data", train=False, download=True, transform=self.transform)

    def get_loaders(self):
        train_loader = DataLoader(self.mnist_train, batch_size=self.batch_size, shuffle=True, drop_last=True)

        test_loader = DataLoader(self.mnist_test, batch_size=self.batch_size, shuffle=True, drop_last=True)

        return train_loader, test_loader

        # Rate coding 
    def rate_encoder(self, num_steps):
        data = iter(get_loader.train_loader)
        data_it = next(data)
        spike_data = spikegen.rate(data_it, num_steps=num_steps)
        
        return spike_data

        # Temporal coding
    def latency_encoder(self, num_steps):
        data = iter(get_loader.train_loader)
        data_it = next(data)
        spike_data = spikegen.latency(data_it, num_steps=num_steps, tau=5, threshold=0.01)
        
        return spike_data

    



       