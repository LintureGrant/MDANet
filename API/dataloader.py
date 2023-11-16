from .dataloader_taxibj import load_data as load_taxibj
from .dataloader_moving_mnist import load_data as load_mmnist
from .dataloader_human import load_data as load_human

def load_data(dataname,batch_size, val_batch_size, data_root, num_workers, **kwargs):
    if dataname == 'taxibj':
        return load_taxibj(batch_size, val_batch_size, data_root, num_workers)
    elif dataname == 'mmnist':
        return load_mmnist(batch_size, val_batch_size, data_root, num_workers)
    elif dataname == 'human':
        return load_human(batch_size, val_batch_size, data_root, num_workers)
