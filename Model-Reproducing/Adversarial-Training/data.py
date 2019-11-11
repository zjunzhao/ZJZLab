import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import time

class cifar10(object):

    def __init__(self, config, mode):
        assert mode in ['train', 'val', 'test']
        idx = range(config.nr_test)
        if mode=='train':
            idx = np.load(config.train_idx_path)
        if mode=='val':
            idx = np.load(config.val_idx_path)
            mode = 'train'
        self.imgs = np.load(config.data_dir+'/'+mode+'/imgs.npy')[idx]
        self.labels = np.load(config.data_dir+'/'+mode+'/labels.npy')[idx]
        if mode=='train':
            self.tfs = transforms.Compose([
                       transforms.Pad(4, padding_mode='reflect'),
                       transforms.RandomHorizontalFlip(),
                       transforms.RandomCrop(32),
                       transforms.ToTensor()])
        else:
            self.tfs = transforms.ToTensor()

    def __len__(self):
        return self.imgs.shape[0]

    def __getitem__(self, idx):
        img, label = self.process(self.imgs[idx]), self.labels[idx]
        idx2 = np.random.choice(self.imgs.shape[0], 1)[0]
        while self.labels[idx2]==label:
            idx2 = np.random.choice(self.imgs.shape[0], 1)[0]
        img2, label2 = self.process(self.imgs[idx2]), self.labels[idx2]
        return img, label, img2, label2

    def process(self, img):
        return self.tfs(Image.fromarray(img))

if __name__=='__main__':
    from config import Config_ClassifierGen
    config = Config_ClassifierGen()
    data  = cifar10(config, 'train')
    data_loader = DataLoader(data, config.batch_size['train'], True, num_workers=2)
    t = time.perf_counter()
    a, b, c, d = next(iter(data_loader))
    print(a.size())
    t = time.perf_counter()-t
    print(t)
