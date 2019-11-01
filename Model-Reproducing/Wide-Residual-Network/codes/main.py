import numpy as np
import time
from config import Config
from train import train_process

if __name__=='__main__':
    '''
    t = time.perf_counter()
    config = Config()
    history = train_process(config)
    for name1 in ['train', 'val']:
        for name2 in ['loss', 'acc']:
            name = name1+'_'+name2+'_list'
            np.save(name+'.npy', np.array(history[name]))
    print('time: ', '%.3f'%(time.perf_counter()-t))
    '''
    config = Config()
    loss_list, acc_list = [], []
    for i in range(1, 5):
        history = train_process(config)
        loss_list.append(history['test_loss'])
        acc_list.append(history['test_acc'])
    print('mean and std of loss are: ', '%.3f'%np.array(loss_list).mean(), '%.3f'%np.array(loss_list).std())
    print('mean and std of acc are: ', '%.3f'%np.array(acc_list).mean(), '%.3f'%np.array(acc_list).std())
