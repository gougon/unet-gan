import config

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler


class Dataset:
    def __init__(self):
        const = config.Constant
        hp = config.Hyperparameter

        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(0.5, 0.5)
                                        ])
        train_dataset = datasets.MNIST(root=const.DATA_FOLDER, download=True,
                                       train=True, transform=transform)
        test_dataset = datasets.MNIST(root=const.DATA_FOLDER, download=True,
                                      train=False, transform=transform)

        split = int(0.8 * len(train_dataset))
        idx_list = list(range(len(train_dataset)))
        train_idx, val_idx = idx_list[:split], idx_list[split:]

        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)

        self.__train_loader = DataLoader(train_dataset, batch_size=hp.BATCH_SIZE, sampler=train_sampler)
        self.__val_loader = DataLoader(train_dataset, batch_size=hp.BATCH_SIZE, sampler=val_sampler)

    @property
    def train_loader(self):
        return self.__train_loader

    @property
    def val_loader(self):
        return self.__val_loader
