import torch
import numpy as np
from torch.utils.data import Dataset


class HumanDataset(Dataset):
    def __init__(self, X, Y):
        super(HumanDataset, self).__init__()
        self.X = (X + 1) / 2
        self.Y = (Y + 1) / 2
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

    dataset = np.load(data_root+'human/dataset.npz')
    X_train, Y_train, X_test, Y_test = dataset['X_train'], dataset['Y_train'], dataset['X_test'], dataset['Y_test']
    # X_test, Y_test =  dataset['X_test'], dataset['Y_test']
    train_set = HumanDataset(X=X_train, Y=Y_train)
    test_set = HumanDataset(X=X_test, Y=Y_test)

    dataloader_train = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)

    dataloader_test = torch.utils.data.DataLoader(
        test_set, batch_size=val_batch_size, shuffle=False, pin_memory=True, num_workers=num_workers)
    # dataloader_train = dataloader_test
    return dataloader_train, None, dataloader_test, 0, 1