import numpy as np
import os

class Config():

    def __init__(self):
        self.save_dir = './tmp/model'
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        self.val_idx_path = './tmp/val_idx.npy'
        if not os.path.exists(self.val_idx_path):
            np.save(self.val_idx_path, np.random.choice(50000, 5000, False))
        val_idx = np.load(self.val_idx_path)
        train_idx = list(set(range(50000))-set(val_idx))
        self.train_idx_path = './tmp/train_idx.npy'
        np.save(self.train_idx_path, train_idx)

    # CIFAR10 parameters
    data_dir = '../../datasets/cifar10_pytorch'
    data_mean = (0.49146724, 0.48226175, 0.44677013)
    data_std = (0.24703224, 0.24348514, 0.26158786)
    nr_label = 10
    nr_train = 45000
    nr_val = 5000
    nr_test = 10000
    # MemoryBank parameters
    dim_embed = 128
    mb_momentum = 0.5
    mb_init_mode = 'unit'
    # KNN parameters
    K = 200
    # training parameters for FeatureExtractor
    nr_sample = 300
    tau = 0.1
    alpha = 1
    adv_mode = 'unfixed'
    final_l = 0
    final_r = 0.2
    # training parameters for optimizer
    init_lr = 0.1
    boundary = [121, 181]
    lr_decay_rate = 0.2
    momentum = 0.9
    weight_decay = 1e-4
    # other training parameters
    nr_epoch = 200
    train_batch_size = 128
    eval_batch_size = 200
    show_interval = 150
    val_interval = 1
    # other parameters
    random_seed = 19951015

if __name__=='__main__':
    config = Config()
