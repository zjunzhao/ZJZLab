import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader 

class cifar10(Dataset):

    def __init__(self, config, mode='train'):
        assert mode in ['train', 'val', 'test']
        idx = range(config.nr_test)
        if mode=='train':
            idx = np.load(config.train_idx_path)
        if mode=='val':
            idx = np.load(config.val_idx_path)
        if mode=='val':
            mode = 'train'
        self.imgs = np.load('../../datasets/cifar10/'+mode+'/imgs.npy')[idx].astype(np.uint8)
        self.labels = np.load('../../datasets/cifar10/'+mode+'/labels.npy')[idx]
        self.idx_map = range(self.imgs.shape[0])
        if mode=='train':
            self.tfs = transforms.Compose([
                       transforms.RandomResizedCrop(size=32, scale=(0.2, 1)),
                       transforms.RandomHorizontalFlip(),
                       transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
                       transforms.RandomGrayscale(p=0.2),
                       transforms.ToTensor(),
                       transforms.Normalize(config.data_mean, config.data_std)])
        else:
            self.tfs = transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize(config.data_mean, config.data_std)])

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        return self.tfs(Image.fromarray(self.imgs[idx])), self.labels[idx], self.idx_map[idx]

if __name__=='__main__':
    from config import Config
    config = Config()
    data1, data2, data3 = cifar10(config, 'train'), cifar10(config, 'val'), cifar10(config, 'test')
    print(data1.__len__(), data2.__len__(), data3.__len__())
