import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import time

class cifar100(Dataset):

    def __init__(self, config, mode):
        assert mode in ['train', 'val', 'test']
        if mode=='train':
            idx = np.load(config.train_idx_path)
            data_dir = config.data_dir+'/train'
        if mode=='val':
            idx = np.load(config.val_idx_path)
            data_dir = config.data_dir+'/train'
        if mode=='test':
            idx = range(config.nr_test)
            data_dir = config.data_dir+'/test'
        self.imgs = np.load(data_dir+'/imgs.npy')[idx]
        self.labels = np.load(data_dir+'/labels.npy')[idx]
        if mode=='train':
            self.tfs = transforms.Compose([
                       transforms.Pad(4, padding_mode='reflect'),
                       transforms.RandomHorizontalFlip(),
                       transforms.RandomCrop(32),
                       transforms.ToTensor(),
                       transforms.Normalize(config.data_mean, config.data_std)])
        else:
            self.tfs = transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize(config.data_mean, config.data_std)])

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img, label = self.process(self.imgs[idx]), self.labels[idx]
        return img, label

    def process(self, img):
        return self.tfs(Image.fromarray(img))

if __name__=='__main__':
    from common import Config
    config = Config()
    data  = cifar100(config, 'train')
    data_loader = DataLoader(data, config.batch_size['train'], True, num_workers=2)
    t = time.perf_counter()
    a, b = next(iter(data_loader))
    print(a.size())
    t = time.perf_counter()-t
    print('Time for load a batch data is: ', '%.3f'%t)
