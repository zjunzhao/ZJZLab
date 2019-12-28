import os
import numpy as np

class Config():

    def __init__(self, nr=50000, nr_train=45000, nr_val=5000, nr_test=10000, work_dir='./work_dir', mode='random'):
        if not os.path.exists(work_dir):
            os.makedirs(work_dir)
        self.model_dir = work_dir+'/model'
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        self.nr_train = nr_train
        self.nr_val = nr_val
        self.nr_test = nr_test
        self.train_idx_path = work_dir+'/train_idx.npy'
        if mode=='random' or not os.path.exists(self.train_idx_path):
            train_idx = np.random.choice(nr, nr_train, False)
            np.save(self.train_idx_path, train_idx)
            rest_idx = np.array(list(set(range(nr))-set(train_idx)))
            val_idx = rest_idx[np.random.choice(len(rest_idx), nr_val, False)]
            self.val_idx_path = work_dir+'/val_idx.npy'
            np.save(self.val_idx_path, val_idx)

    # model parameters
    nr_block = 4
    widen_factor = 10
    dropout = 0.0
    # data parameters
    data_dir = '../../datasets/cifar100_custom'
    nr_label = 100
    data_mean = (0.50707516, 0.48654887, 0.44091784)
    data_std = (0.26733429, 0.25643846, 0.27615047)
    # training parameters
    random_seed = 19951015
    nr_epoch = 200
    batch_size = {'train':128, 'val':100, 'test':100}
    init_lr = 1e-1
    lr_boundaries = [61, 121, 181]
    lr_decay_rate = 0.2
    momentum = 0.9
    weight_decay = 5e-4

if __name__=='__main__':
    config = Config()
