import numpy as np
import time
from config import Config_Classifier
from train import ClassifierTrainer

if __name__=='__main__':
    config = Config_Classifier()
    train_attack_mode_list = ['FGSM']
    eval_attack_mode_list = ['vanilla', 'FGSM', 'PGD']
    for train_attack_mode in train_attack_mode_list:
        print('Attack '+train_attack_mode)
        t = time.perf_counter()
        ClassifierTrainer(config, train_attack_mode, eval_attack_mode_list).train()
        t = time.perf_counter()-t
        print('Total time: ', '%.3f'%t)
