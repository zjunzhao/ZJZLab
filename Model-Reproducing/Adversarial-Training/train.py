import numpy as np
import torch
from torchvision import transforms
from data import cifar10
from torchvision.datasets import CIFAR10
from torch.utils.data import Subset, DataLoader
import os
from model import Classifier, Gen

attack_mode_list = ['vanilla', 'FGSM', 'PGD']
tot_step_pgd = 10
lr_pgd = 0.01#2/255

def adjust_lr(optimizer, rate):
    for param_group in optimizer.param_groups:
        param_group['lr'] *= rate

class ClassifierTrainer(object):

    def __init__(self, config, train_attack_mode='vanilla', eval_attack_mode_list=['vanilla']):
        assert train_attack_mode in attack_mode_list
        for eval_attack_mode in eval_attack_mode_list:
            assert eval_attack_mode in attack_mode_list
        # set up random seed and device
        torch.manual_seed(config.random_seed)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # set up data and loader
        train_tfs = transforms.Compose([
                    transforms.Pad(4, padding_mode='reflect'),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(32),
                    transforms.ToTensor()])
        eval_tfs = transforms.ToTensor()
        train_idx = np.load(config.train_idx_path)
        train_data = Subset(CIFAR10(config.data_dir, True, train_tfs, download=True), train_idx)
        self.train_loader = DataLoader(train_data, config.batch_size['train'], True, num_workers=2)
        val_idx = np.load(config.val_idx_path)
        val_data = Subset(CIFAR10(config.data_dir, True, eval_tfs, download=True), val_idx)
        self.val_loader = DataLoader(val_data, config.batch_size['val'], False, num_workers=2)
        test_data = CIFAR10(config.data_dir, False, eval_tfs, download=True)
        self.test_loader = DataLoader(test_data, config.batch_size['test'], False, num_workers=2)
        # set up Classifier
        self.cla = Classifier(config.nr_block, config.widen_factor, config.dropout, config.nr_label)
        if torch.cuda.device_count()>1:
            self.cla = torch.nn.DataParallel(self.cla)
        self.cla.to(self.device)
        # set up Attacker
        self.train_attack_mode = train_attack_mode
        self.eval_attack_mode_list = eval_attack_mode_list
        self.epsilon = config.epsilon
        # set up criterion and optimizer
        self.criterion = torch.nn.CrossEntropyLoss()
        init_lr, momentum, weight_decay = config.init_lr, config.momentum, config.weight_decay
        self.optimizer = torch.optim.SGD(self.cla.parameters(), init_lr, momentum, weight_decay=weight_decay)
        # set up other parameters
        self.data_mean = torch.tensor(config.data_mean).view(1, 3, 1, 1).to(self.device)
        self.data_std = torch.tensor(config.data_std).view(1, 3, 1, 1).to(self.device)
        self.nr_epoch = config.nr_epoch
        self.boundary, self.lr_decay_rate = config.boundary, config.lr_decay_rate
        self.save_dir = config.save_dir

    def train(self):
        print('training start!')
        # training process
        opt_val_acc = 0
        for epoch in range(1, self.nr_epoch+1):
            if epoch in self.boundary:
                adjust_lr(self.optimizer, self.lr_decay_rate)
            train_loss, train_acc = 0, 0
            for i, data in enumerate(self.train_loader, 1):
                imgs, labels = data[0].to(self.device), data[1].to(self.device)
                # generate adversarial example
                #if i%2==0:
                imgs = self.attacker(imgs, labels, self.train_attack_mode).detach()
                # calc loss
                logits = self.cla(self.normalize(imgs))
                loss = self.criterion(logits, labels)
                # update parameters of Classifer
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                # update output metrics
                train_loss = 0.5*train_loss+0.5*loss.item()
                _, labels_pre = logits.max(dim=1)
                acc = (labels_pre==labels).float().mean()
                train_acc = 0.5*train_acc+0.5*acc.item()
            print('_______training_______')
            print('epoch: ', epoch, 'loss: ', '%.3f'%train_loss, 'acc: ', '%.3f'%train_acc)
            self.cla.eval()
            val_acc = self.evaluation('val', self.train_attack_mode)
            print('_______validation_______')
            print('epoch: ', epoch, 'acc: ', '%.3f'%val_acc)
            if opt_val_acc<=val_acc:
                opt_val_acc = val_acc
                torch.save(self.cla.state_dict(), self.save_dir+'/opt'+self.train_attack_mode)
                print('Current opt val acc is: ', '%.3f'%opt_val_acc)
            self.cla.train()
        print('Training finish! Optimal val acc is: ', '%.3f'%opt_val_acc)
        self.cla.load_state_dict(torch.load(self.save_dir+'/opt'+self.train_attack_mode))
        self.cla.eval()
        print('_______testing_______')
        for eval_attack_mode in self.eval_attack_mode_list:
            test_acc = self.evaluation('test', eval_attack_mode)
            print('acc for '+eval_attack_mode+' is: ', '%.3f'%test_acc)
        self.cla.train()

    def evaluation(self, mode, attack_mode):
        assert mode in ['val', 'test']
        assert attack_mode in attack_mode_list
        if mode=='val':
            loader = self.val_loader
        else:
            loader = self.test_loader
        ret_acc = 0
        for i, data in enumerate(loader, 1):
            imgs, labels = data[0].to(self.device), data[1].to(self.device)
            # generate adversarial example
            imgs = self.attacker(imgs, labels, attack_mode).detach()
            logits = self.cla(self.normalize(imgs))
            _, labels_pre = logits.max(dim=1)
            acc = (labels_pre==labels).float().mean()
            ret_acc = (i-1)/i*ret_acc+acc.item()/i
        return ret_acc

    def normalize(self, imgs):
        return (imgs-self.data_mean)/self.data_std

    def attacker(self, imgs, labels, attack_mode):
        assert attack_mode in attack_mode_list
        # gradient based attacker
        if attack_mode=='vanilla':
            return imgs
        if attack_mode=='FGSM':
            imgs.requires_grad = True
            logits = self.cla(self.normalize(imgs))
            loss = self.criterion(logits, labels)
            loss.backward()
            grads = imgs.grad.data
            return torch.clamp(imgs+self.epsilon*grads.sign(), 0, 1).detach()
        if attack_mode=='PGD':
            tot_step, lr = tot_step_pgd, lr_pgd
            delta = torch.rand_like(imgs, requires_grad=True)
            delta.data = (delta*2-1)*self.epsilon
            delta.data = torch.max(torch.min(delta, 1-imgs), -imgs)
            for _ in range(tot_step):
                logits = self.cla(self.normalize(imgs+delta))
                loss = self.criterion(logits, labels)
                loss.backward()
                grads = delta.grad.data
                delta.data = torch.clamp(delta+lr*grads.sign().detach(), -self.epsilon, self.epsilon)
                delta.data = torch.max(torch.min(delta, 1-imgs), -imgs)
                delta.grad.zero_()
            return imgs+delta.detach()