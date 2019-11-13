import numpy as np
import os

eps = 0.03
mean = (0, 0, 0)#(0.49146724, 0.48226175, 0.44677013)
std = (1, 1, 1)#(0.24703224, 0.24348514, 0.26158786)
w_f = 10

class Config_Classifier(object):

    def __init__(self, name='Classifier', nr_train=50000, nr_val=5000, mode='random'):
        # make work dir and save dir
        self.work_dir = './tmp/'+name
        self.save_dir = self.work_dir+'/model'
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        # train validation split
        self.val_idx_path = self.work_dir+'/val_idx.npy'
        if mode=='random' or not os.path.exists(self.val_idx_path):
            np.save(self.val_idx_path, np.random.choice(nr_train, nr_val, False))
        val_idx = np.load(self.val_idx_path)
        train_idx = list(set(range(nr_train))-set(val_idx))
        self.train_idx_path = self.work_dir+'/train_idx.npy'
        np.save(self.train_idx_path, train_idx)

    # CIFAR10 (pytorch version) parameters
    data_dir = '../../datasets/cifar10_pytorch'
    data_mean = mean
    data_std = std
    nr_label = 10
    # parameters for Classifier
    nr_block = 4
    widen_factor = w_f
    dropout = 0
    # parameters for Attacker
    epsilon = eps
    # parameters for optimizer
    init_lr = 0.1
    boundary = [37, 73, 109]
    lr_decay_rate = 0.2
    momentum = 0.9
    weight_decay = 1e-4
    # parameters for training process
    nr_epoch = 120
    batch_size = {'train':128, 'val':100, 'test':100}
    # other parameters
    random_seed = 19951015

if __name__=='__main__':
    config = Config_Classifier()
