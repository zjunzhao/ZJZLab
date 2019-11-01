import numpy as np
import time
from config import Config
from train import train_process_representation

if __name__=='__main__':
    
    config = Config()
    test_by_mb_list, test_by_train_list, time_list = [], [], []
    for _ in range(5):
        t = time.perf_counter()
        res = train_process_representation(config)
        t = time.perf_counter()-t
        test_by_mb_list.append(res['test_by_mb'])
        test_by_train_list.append(res['test_by_train'])
        print('total time: ', '%.3f'%t)
        time_list.append(t)
    L1, L2, L3 = np.array(test_by_mb_list), np.array(test_by_train_list), np.array(time_list)
    print('test by mb acc mean and std are: ', '%.3f'%L1.mean(), '%.3f'%L1.std())
    print('test by train acc mean and std are: ', '%.3f'%L2.mean(), '%.3f'%L2.std())
    print('time used mean and std are', '%.3f'%L3.mean(), '%.3f'%L3.std())
    '''
    for adv_mode in ['fixed', 'unfixed']:
        for (final_l, final_r) in [(0, 0.1), (0, 0.15), (0, 0.2)]:
            config.adv_mode = adv_mode
            config.final_l = final_l
            config.final_r = final_r
            t = time.perf_counter()
            res = train_process_representation(config)
            t = time.perf_counter()-t
            print('_______res_______')
            #print('weight_decay: ', weight_decay, 'init_lr: ', init_lr, 'lr_decay_rate: ', lr_decay_rate)
            print('adv_mode: ', adv_mode, 'final_l: ', final_l, 'final_r: ', final_r)
            print('total time: ', '%.3f'%t)
            print(res)
    '''
