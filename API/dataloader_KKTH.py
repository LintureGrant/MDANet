import numpy as np
import os
import pickle
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from PIL import Image

class TimeSeriesDataset(Dataset):
    def __init__(self, root_dir, n_frames_input=10, n_frames_output=20):
        self.mean, self.std = 0, 1
        self.n_frames_in = n_frames_input
        self.n_frames_out = n_frames_output
        n_frames = n_frames_input + n_frames_output
        subject_dirs = sorted([os.path.join(root_dir, d) for d in os.listdir(root_dir)])  # [01, 02, 03]

        self.specific_view_files = []
        for d in subject_dirs:
            self.specific_view_files.append(sorted([os.path.join(d, f) for f in os.listdir(d)]))
        ##选中了所有文件夹
        self.nframes_list = []
        for f in self.specific_view_files:
            #for i in range(len(f) - n_frames):
            for i in range(28):
                self.nframes_list.append(f[i*5:i*5 + n_frames])
        self.transforms = A.Compose([
            A.Resize(width=128, height=128),
            A.Normalize(mean=0, std=1, max_pixel_value=255.0),
            ToTensorV2()
        ])

    def __len__(self):
        return len(self.nframes_list)

    def __getitem__(self, index):
        input = []
        output = []
        for f in self.nframes_list[index][:self.n_frames_in]:
            input_image = np.array(Image.open(f))
            input_aug = self.transforms(image=input_image)['image']
            input.append(input_aug)
        input = torch.stack((input))

        for f in self.nframes_list[index][self.n_frames_out:]:
            output_image = np.array(Image.open(f))
            output_aug = self.transforms(image=output_image)['image']
            output.append(output_aug)
        output = torch.stack((output))

        return index, input, output

def load_data(
        batch_size, val_batch_size,
        data_root, num_workers):
    data_root = os.path.join(data_root, 'training_lib_KTH')
    train_path=os.path.join(data_root,'train')
    vaild_path=os.path.join(data_root,'valid')
    train_dataset = TimeSeriesDataset(root_dir=train_path, n_frames_input=10, n_frames_output=20)
    vaild_dataset = TimeSeriesDataset(root_dir=vaild_path, n_frames_input=10, n_frames_output=20)
    validation_split = .3
    shuffle_dataset = True
    random_seed = 420

    dataset_size = len(train_dataset)
    print(dataset_size)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers,pin_memory=True)
    print('len1',len(train_loader))
    valid_loader = torch.utils.data.DataLoader(vaild_dataset, batch_size=val_batch_size,num_workers=num_workers)
    print('len2', len(valid_loader))
    return train_loader, None, valid_loader, 0, 1