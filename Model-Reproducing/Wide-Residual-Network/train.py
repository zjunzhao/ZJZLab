import torch
from data import cifar100
from torch.utils.data import DataLoader
from model import WideResnet

class Trainer(object):

    def __init__(self, config):
        self.config = config
        # set up random seed
        torch.manual_seed(config.random_seed)
        # set up device
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # set up data
        train_data = cifar100(config, 'train')
        self.train_loader = DataLoader(train_data, config.batch_size['train'], True, num_workers=2)
        val_data = cifar100(config, 'val')
        self.val_loader = DataLoader(val_data, config.batch_size['val'], False, num_workers=2)
        test_data = cifar100(config, 'test')
        self.test_loader = DataLoader(test_data, config.batch_size['test'], False, num_workers=2)
        # set up nn, criterion and optimizer
        self.net = WideResnet(config.nr_block, config.widen_factor)
        self.net.to(self.device)
        self.criterion = torch.nn.CrossEntropyLoss()
        lr, momentum, weight_decay = config.init_lr, config.momentum, config.weight_decay
        parameters = self.net.parameters()
        self.optimizer = torch.optim.SGD(parameters, lr, momentum, weight_decay=weight_decay)

    def train(self):
        # training process
        opt_val_acc = 0
        train_loss_list, train_acc_list, val_loss_list, val_acc_list = [], [], [], []
        for epoch in range(1, self.config.nr_epoch+1):
            if epoch in self.config.lr_boundaries:
                self.adjust_lr(self.config.lr_decay_rate)
            train_loss, train_acc = 0, 0
            for i, data in enumerate(self.train_loader, 1):
                imgs, labels = data[0].to(self.device), data[1].to(self.device)
                logits = self.net(imgs)
                loss = self.criterion(logits, labels)
                # update nn's paremeters
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                # update metrics
                train_loss = (i-1)/i*train_loss+loss.item()/i
                _, labels_pre = logits.max(dim=1)
                acc = (labels_pre==labels).float().mean()
                train_acc = (i-1)/i*train_acc+acc.item()/i
            train_loss_list.append(train_loss)
            train_acc_list.append(train_acc)
            print('_______training_______')
            print('epoch: ', epoch, 'loss: ', '%.3f'%train_loss, 'acc: ', '%.3f'%train_acc)
            val_loss, val_acc = self.evaluate('val')
            if val_acc>=opt_val_acc:
                opt_val_acc = val_acc
                torch.save(self.net.state_dict(), self.config.model_dir+'/opt')
            val_loss_list.append(val_loss)
            val_acc_list.append(val_acc)
            print('_______validation_______')
            print('epoch: ', epoch, 'loss: ', '%.3f'%val_loss, 'acc: ', '%.3f'%val_acc)
        print('Training finish. The optimal validation accuracy is: ', '%.3f'%opt_val_acc)
        self.net.load_state_dict(torch.load(self.config.model_dir+'/opt'))
        test_loss, test_acc = self.evaluate('test')
        print('test_loss: ', '%.3f'%test_loss, 'test_acc: ', '%.3f'%test_acc)
        history = {'train_loss_list':train_loss_list, 'train_acc_list':train_acc_list,
                   'val_loss_list':val_loss_list, 'val_acc_list':val_acc_list,
                   'test_loss':test_loss, 'test_acc':test_acc}
        return history

    def evaluate(self, mode):
        assert mode in ['val', 'test']
        if mode=='val':
            loader = self.val_loader
        else:
            loader = self.test_loader
        self.net.eval()
        ret_loss, ret_acc = 0, 0
        for i, data in enumerate(loader, 1):
            imgs, labels = data[0].to(self.device), data[1].to(self.device)
            logits = self.net(imgs)
            loss = self.criterion(logits, labels)
            ret_loss = (i-1)/i*ret_loss+loss.item()/i
            _, labels_pre = logits.max(dim=1)
            acc = (labels_pre==labels).float().mean()
            ret_acc = (i-1)/i*ret_acc+acc.item()/i
        self.net.train()
        return ret_loss, ret_acc

    def adjust_lr(self, rate):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] *= rate
