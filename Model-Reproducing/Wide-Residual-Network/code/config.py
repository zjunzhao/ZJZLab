import os

class Config():

    def __init__(self):
        self.save_dir = './model'
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    # model parameters
    nr_block = 4
    widen_factor = 10
    dropout = 0.0
    # data parameters
    data_dir = '../../datasets/cifar10'
    data_mean = (0.49146724, 0.48226175, 0.44677013)
    data_std = (0.24703224, 0.24348514, 0.26158786)
    # training parameters
    random_seed = 19951015
    nr_epoch = 200
    train_batch_size = 128
    eval_batch_size = 100
    lr = 1e-1
    momentum = 0.9
    weight_decay = 5e-4
    show_interval = 100
    val_interval = 1

if __name__=='__main__':
    config = Config()
