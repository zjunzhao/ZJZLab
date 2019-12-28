from common import Config
from train import Trainer
import numpy as np
import time

nr_run, test_loss_list, test_acc_list, time_list = 5, [], [], []
for _ in range(nr_run):
    config = Config(nr_train=45000, nr_val=5000)
    trainer = Trainer(config)
    t = time.perf_counter()
    history = trainer.train()
    t = time.perf_counter()-t
    test_loss_list.append(history['test_loss'])
    test_acc_list.append(history['test_acc'])
    time_list.append(t)
test_loss_array, test_acc_array = np.array(test_loss_list), np.array(test_acc_list)
test_loss_mean, test_loss_std = test_loss_array.mean(), test_loss_array.std()
test_acc_mean, test_acc_std = test_acc_array.mean(), test_acc_array.std()
time_array = np.array(time_list)
time_mean, time_std = time_array.mean(), time_array.std()

print('mean and std of test loss are: ', '%.3f'%test_loss_mean, '%.3f'%test_loss_std)
print('mean and std of test acc are: ', '%.3f'%test_acc_mean, '%.3f'%test_acc_std)
print('mean and std of time used are: ', '%.3f'%time_mean, '%.3f'%time_std)
