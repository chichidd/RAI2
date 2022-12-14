"""
Custom Dataloaders for each of the considered datasets
"""

import os

from torchvision import datasets
import numpy as np
from mixmo.augmentations.standard_augmentations import get_default_composed_augmentations
from mixmo.loaders import cifar_dataset, abstract_loader
from mixmo.utils.logger import get_logger
from torch.utils.data import Dataset
from PIL import Image

LOGGER = get_logger(__name__, level="DEBUG")


class SubTrainDataset(Dataset):
    def __init__(self, data, targets, transform=None, target_transform=None):
        self.data = data
        self.targets = targets
        self.transform = transform
        self.target_transform = target_transform
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target
    def __len__(self):
        return len(self.data)


class CIFAR10Loader(abstract_loader.AbstractDataLoader):
    """
    Loader for the CIFAR10 dataset that inherits the abstract_loader.AbstractDataLoader dataloading API
    and defines the proper augmentations and datasets
    """

    def _init_dataaugmentations(self):
        (self.augmentations_train, self.augmentations_test) = get_default_composed_augmentations(
            dataset_name="cifar",
        )

    def _init_dataset(self, corruptions=False):
        self.train_dataset = cifar_dataset.CustomCIFAR10(
            root=self.data_dir, train=True, download=True, transform=self.augmentations_train
        )
        if not corruptions:
            self.test_dataset = cifar_dataset.CustomCIFAR10(
                root=self.data_dir, train=False, download=True, transform=self.augmentations_test
            )
        else:
            self.test_dataset = cifar_dataset.CIFARCorruptions(
                root=self.corruptions_data_dir, train=False, transform=self.augmentations_test
            )

    @property
    def data_dir(self):
        return os.path.join(self.dataplace, "cifar10-data")

    @property
    def corruptions_data_dir(self):
        return os.path.join(self.dataplace, "CIFAR-10-C")


    @staticmethod
    def properties(key):
        dict_key_to_values = {
            "conv1_input_size": (16, 32, 32),
            "conv1_is_half_size": False,
            "pixels_size": 32,
        }
        return dict_key_to_values[key]


class CIFAR100Loader(CIFAR10Loader):
    """
    Loader for the CIFAR100 dataset that inherits the abstract_loader.AbstractDataLoader dataloading API
    and defines the proper augmentations and datasets
    """

    def _init_dataset(self, corruptions=False):
        self.train_dataset = cifar_dataset.CustomCIFAR100(
            root=self.data_dir, train=True, download=True, transform=self.augmentations_train
        )
        if not corruptions:
            self.test_dataset = cifar_dataset.CustomCIFAR100(
                root=self.data_dir, train=False, download=True, transform=self.augmentations_test
            )
        else:
            self.test_dataset = cifar_dataset.CIFARCorruptions(
                root=self.corruptions_data_dir, train=False, transform=self.augmentations_test
            )

    @property
    def data_dir(self):
        return os.path.join(self.dataplace, "cifar100-data")

    @property
    def corruptions_data_dir(self):
        return os.path.join(self.dataplace, "CIFAR-100-C")


class TinyImagenet200Loader(abstract_loader.AbstractDataLoader):
    """
    Loader for the TinyImageNet dataset that inherits the abstract_loader.AbstractDataLoader dataloading API
    and defines the proper augmentations and datasets
    """

    def _init_dataaugmentations(self):
        (self.augmentations_train, self.augmentations_test) = get_default_composed_augmentations(
            dataset_name="tinyimagenet",
        )

    @property
    def data_dir(self):
        return os.path.join(self.dataplace, "tinyimagenet200-data")

    def _init_dataset(self, corruptions=False):
        traindir = os.path.join(self.data_dir, 'train')
        self.train_dataset = datasets.ImageFolder(traindir)

        X_list, y_list = [], []
        # recover from np.array
        # from PIL import Image
        # im = Image.fromarray(X_train_list[0])
        for x, y in self.train_dataset:
            X_list.append(np.array(x))
            y_list.append(y)
        X_np, y_np = np.array(X_list), np.array(y_list)

        vic_num = len(self.train_dataset) // 2
        np.random.seed(0)
        train_idx_array = np.arange(len(self.train_dataset))
        np.random.shuffle(train_idx_array)
        vic_idx = train_idx_array[:vic_num]

        shift = int(self.config_args['inter_propor'] * vic_num)
        start_att_idx = vic_num - shift
        att_idx = train_idx_array[start_att_idx: start_att_idx + vic_num]
        X_set = X_np[att_idx]
        y_set = y_np[att_idx]

        self.train_dataset = SubTrainDataset(X_set, list(y_set), self.augmentations_train)
        
        valdir = os.path.join(self.data_dir, 'val/images')
        self.test_dataset = datasets.ImageFolder(valdir, self.augmentations_test)

    @staticmethod
    def properties(key):
        dict_key_to_values = {
            "conv1_input_size": (64, 32, 32),
            "conv1_is_half_size": True,
            "pixels_size": 64,
        }
        return dict_key_to_values[key]
