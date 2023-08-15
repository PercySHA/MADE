from torchvision.datasets import SVHN
import numpy as np
import torchvision.transforms as transforms
import torch

class SVHNWithIdx(SVHN):
    """
    Extends SVHN dataset to yield index of element in addition to image and target label.
    """

    def __init__(self,
                 root,
                 train=True,
                 transform=None,
                 target_transform=None,
                 download=False,
                 rand_fraction=0.0):
        super(SVHNWithIdx, self).__init__(root=root,
                                          split='train' if train else 'test',
                                          transform=transform,
                                          target_transform=target_transform,
                                          download=download)

        assert (rand_fraction <= 1.0) and (rand_fraction >= 0.0)
        self.rand_fraction = rand_fraction
        self.train_bool = train

        if self.rand_fraction > 0.0:
            self.data = self.corrupt_fraction_of_data()

    def corrupt_fraction_of_data(self):
        """Corrupts fraction of train data by permuting image-label pairs."""

        # Check if we are not corrupting test data
        assert self.train_bool is True, 'We should not corrupt test data.'

        nr_points = len(self.data)
        nr_corrupt_instances = int(np.floor(nr_points * self.rand_fraction))
        print('Randomizing {} fraction of data == {} / {}'.format(self.rand_fraction,
                                                                  nr_corrupt_instances,
                                                                  nr_points))
        # We will corrupt the top fraction data points
        corrupt_data = self.data[:nr_corrupt_instances, :, :, :]
        clean_data = self.data[nr_corrupt_instances:, :, :, :]

        # Corrupting data
        rand_idx = np.random.permutation(np.arange(len(corrupt_data)))
        corrupt_data = corrupt_data[rand_idx, :, :, :]

        # Adding corrupt and clean data back together
        return np.vstack((corrupt_data, clean_data))

    def __getitem__(self, index):
        """
        Args:
            index (int):  index of element to be fetched

        Returns:
            tuple: (index， sample, target) where index is the index of this sample in dataset.
        """
        img, target = super().__getitem__(index)
        return img, target

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

def get_SVHN_train_loader(batch_size, rand_fraction=0.0, num_workers=1, shuffle=True):
    svhn_train_dataset = SVHNWithIdx(root='./data',
                                           train=True,
                                           download=True,
                                           transform=train_transform,
                                           rand_fraction=rand_fraction)

    svhn_train_loader = torch.utils.data.DataLoader(dataset=svhn_train_dataset,
                                                       batch_size=batch_size,
                                                       shuffle=shuffle,
                                                       num_workers=num_workers)
    return svhn_train_loader

def get_SVHN_test_loader(batch_size, num_workers=1, shuffle=False):
    svhn_test_dataset = SVHNWithIdx(root='./data',
                                          train=False,
                                          download=True,
                                          transform=test_transform)

    svhn_test_loader = torch.utils.data.DataLoader(dataset=svhn_test_dataset,
                                                      batch_size=batch_size,
                                                      shuffle=shuffle,
                                                      num_workers=num_workers)
    return svhn_test_loader

def get_SVHN_train_valid_split_loader(batch_size, rand_fraction=0.0, num_workers=1, shuffle=True, valid_size=0.1):
    """ SVHN training set loader """
    svhn_train_dataset = SVHNWithIdx(root='./data',
                                           train=True,
                                           download=True,
                                           transform=train_transform,
                                           rand_fraction=rand_fraction)

    num_train = len(svhn_train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    if shuffle:
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_idx)
    valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(valid_idx)

    svhn_train_loader = torch.utils.data.DataLoader(dataset=svhn_train_dataset,
                                                       batch_size=batch_size,
                                                       shuffle=False,
                                                       num_workers=num_workers,
                                                       sampler=train_sampler)
    svhn_valid_loader = torch.utils.data.DataLoader(dataset=svhn_train_dataset,
                                                       batch_size=batch_size,
                                                       shuffle=False,
                                                       num_workers=num_workers,
                                                       sampler=valid_sampler)
    return svhn_train_loader, svhn_valid_loader