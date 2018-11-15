from pathlib import Path

import torch
from torch.utils import data


class ColorDataset(data.Dataset):
    """Characterizes a dataset for PyTorch"""
    DATA_PATH = Path("../../data/processed")

    def __init__(self, split: str):
        """Initialization"""
        self.split = split
        length = self._count_files()
        self.list_IDs = [f"{split}{i}" for i in range(length)]

    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.list_IDs)

    def __getitem__(self, index):
        """Generates one sample of data"""
        # Select sample
        ID = self.list_IDs[index]

        # Load data
        instance = torch.load(self.DATA_PATH / (ID + '.pt'))

        return instance

    def _count_files(self):
        glob_expression = f"{self.split}*"
        matching_files = self.DATA_PATH.glob(glob_expression)
        # glob is a generator; no len() operation.
        return sum(1 for _ in matching_files)

if __name__ == '__main__':
    train = ColorDataset("train")
    print(train._count_files())
    print(train.list_IDs)
    print(train[20])
