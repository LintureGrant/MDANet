import torch
import numpy as np
from torch.utils.data import Dataset
from utils_data import h5_virtual_file, window, weights_init
class TrafficDataset(Dataset):
    def __init__(self, X, Y):
        super(TrafficDataset, self).__init__()
        # self.X = (X + 1) / 2
        # self.Y = (Y + 1) / 2
        self.X = X
        self.Y = Y
        self.mean = 0
        self.std = 1

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        data = torch.tensor(self.X[index, ::]).float()
        labels = torch.tensor(self.Y[index, ::]).float()
        return data, labels

def load_data(
        batch_size, val_batch_size,
        data_root, num_workers):
    taxibj_dir = "./data/taxibj/TaxiBJ/"

    f = h5_virtual_file(
        [
            f"{taxibj_dir}BJ13_M32x32_T30_InOut.h5",
            f"{taxibj_dir}BJ14_M32x32_T30_InOut.h5",
            f"{taxibj_dir}BJ15_M32x32_T30_InOut.h5",
            f"{taxibj_dir}BJ16_M32x32_T30_InOut.h5",
        ]
    )

    data = f.get("data")

    ## generate data

    batch_size = 16

    train_dataset = SlidingWindowDataset.SlidingWindowDataset(data, lambda t: t / 255)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=True, )
    test_dataset = SlidingWindowDataset.SlidingWindowDataset(data, lambda t: t / 255, train=False)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=True, )
    train_dataset.mean=0
    train_dataset.std=1

    test_dataset.mean=0
    test_dataset.std=1
    return train_dataloader, None, test_dataloader, 0, 1