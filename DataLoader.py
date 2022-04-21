import torch
import torchvision
import torchvision.transforms as transforms

class DataLoader():
    def __init__(self, **kwargs):

        # transforms
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

        val_fraction = kwargs.get('val_fraction', 0.1)
        batch_size = kwargs.get('batch_size', 64)

        train_set = torchvision.datasets.MNIST('../data', download=True, train=True, transform=transform)
        self.test_set = torchvision.datasets.MNIST('../data', download=True, train=False, transform=transform)

        # split original train set into train and validation
        val_size = int(val_fraction*len(train_set))
        train_size = len(train_set) - val_size
        self.train_set, self.val_set = torch.utils.data.random_split(train_set, [train_size, val_size])

        # dataloaders
        self.train_loader = torch.utils.data.DataLoader(self.train_set, batch_size=batch_size, shuffle=True, num_workers=0)
        self.val_loader = torch.utils.data.DataLoader(self.val_set, batch_size=val_size, num_workers=0)
        self.test_loader = torch.utils.data.DataLoader(self.test_set, batch_size=len(self.test_set), shuffle=True, num_workers=0)