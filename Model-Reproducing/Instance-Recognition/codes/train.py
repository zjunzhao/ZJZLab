import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torch.utils.data import Subset, DataLoader
import os
import math
from data import cifar10
from resnet_cifar import ResNet18

class MemoryBank(object):

    def __init__(self, n, d, momentum=0.5, init_mode='unit'):
        assert init_mode in ['unit', 'nonunit']
        if init_mode=='unit':
            # init content with zero mean unit vector
            self.content = torch.rand(n, d)-0.5
            self.content = self.content/self.content.norm(p=2, dim=1, keepdim=True)
        else:
            # init content with zero mean 1/sqrt(d) std vector
            std = 1/np.sqrt(d/3)
            self.content = torch.rand(n, d)*2*std-std
        self.momentum = momentum

    def update(self, inp, idx):
        m = self.momentum
        # moving average update content
        self.content[idx] = m*self.content[idx]+(1-m)*inp.cpu().detach()
        self.content[idx] = self.content[idx]/self.content[idx].norm(p=2, dim=1, keepdim=True)

    def sample(self, idx):
        # sample by idx
        return self.content[idx]

    def random_sample(self, size):
        # random sample
        idx = np.random.choice(self.content.size(0), size)
        return self.content[idx]

class KNN(object):

    def __init__(self, X, Y, nr_label, K=200, tau=0.1):
        self.X, self.Y, self.nr_label, self.K, self.tau = X, Y, nr_label, K, tau

    def predict(self, inp):
        X, inp = self.X, inp
        # s[i, j] = exp(sum(inp[i]*X[j])/tau)
        s = torch.matmul(inp, X.permute(1, 0))
        s = (s/self.tau).exp()
        _, idx = torch.topk(s, self.K, 1)
        y = self.Y[idx.view(-1)].view(-1, self.K)
        s = s[torch.tensor(range(inp.size(0))).repeat_interleave(self.K), idx.view(-1)].view(-1, self.K)
        w_list = []
        for i in range(self.nr_label):
            w = (s*(y==i).float()).sum(dim=1).view(-1, 1)
            w_list.append(w)
        w = torch.cat(w_list, 1)
        _, preds = torch.max(w, 1)
        return preds

def adjust_lr(optimizer, rate=0.1):
    for param_group in optimizer.param_groups:
        param_group['lr'] *= rate

def train_process_representation(config):
    # set up random seed
    torch.manual_seed(config.random_seed)
    # set up device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # set up data
    train_data = cifar10(config, 'train')
    train_loader = DataLoader(train_data, config.train_batch_size, True, num_workers=2)
    val_data = cifar10(config, 'val')
    val_loader = DataLoader(val_data, config.eval_batch_size, False, num_workers=2)
    test_data = cifar10(config, 'test')
    test_loader = DataLoader(test_data, config.eval_batch_size, False, num_workers=2)
    # set up nn, loss and optimizer
    net = ResNet18()
    if torch.cuda.device_count()>1:
        net = torch.nn.DataParallel(net)
    net.to(device)
    lr, momentum, weight_decay = config.init_lr, config.momentum, config.weight_decay
    optimizer = torch.optim.SGD(net.parameters(), lr, momentum, weight_decay=weight_decay)
    # training process
    mb = MemoryBank(config.nr_train, config.dim_embed, config.mb_momentum, config.mb_init_mode)
    opt_val_acc = 0
    for epoch in range(1, config.nr_epoch+1):
        if epoch in config.boundary:
            adjust_lr(optimizer, config.lr_decay_rate)
        train_prob, train_sep = 0, 0
        for i, data in enumerate(train_loader, 1):
            # calc embedding of imgs, sample positive embedding and negative embedding from mb
            imgs, idx = data[0].to(device), data[2]
            embed = net(imgs)
            sz = embed.size(0)
            embed = embed.view(sz, 1, -1)
            embed_p = mb.sample(idx).to(device).view(sz, 1, -1)
            embed_n = mb.random_sample(sz*config.nr_sample).to(device).view(sz, config.nr_sample, -1)
            # calc logits and extract logits of the most hard examples
            logits_p, logits_n = (embed*embed_p).sum(dim=2)/config.tau, (embed*embed_n).sum(dim=2)/config.tau
            logits_n, _ = logits_n.sort(dim=1, descending=True)
            if epoch<=10:
                idx_l, idx_r = 0, 1
            else:
                beta = (epoch-10)/(config.nr_epoch-10)
                idx_l, idx_r = config.final_l*beta, 1-(1-config.final_r)*beta
            if config.adv_mode=='fixed':
                idx_l, idx_r = config.final_l, config.final_r
            idx_l, idx_r = int(config.nr_sample*idx_l), int(config.nr_sample*idx_r)
            logits = torch.cat((logits_p, logits_n[:, idx_l:idx_r]), dim=1)
            embed = embed.view(sz, -1)
            # calc loss
            probs = F.softmax(logits, dim=1)[:, 0]
            alpha = (config.nr_train-1)/(idx_r-idx_l)*config.alpha
            probs = probs/((1-alpha)*probs+alpha)
            loss = -probs.log().mean()
            # calc output metrics
            prob = probs.mean()
            train_prob = 0.5*train_prob+0.5*prob.item()
            sep = torch.mm(embed, embed.permute(1, 0)).mean()
            train_sep = 0.5*train_sep+0.5*sep.item()
            # update parameters and MemoryBank
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            mb.update(embed, idx)
            # output metrics
            if i%config.show_interval==0:
                print('_______training_______')
                print('epoch: ', epoch, 'step: ', i, 'prob: ', '%.3f'%train_prob, 'sep: ', '%.3f'%train_sep)
        if epoch%config.val_interval==0:
            # validate by KNN
            net.eval()
            with torch.no_grad():
                x_train, y_train = mb.content.to(device), torch.tensor(train_data.labels).to(device)
                knn = KNN(x_train, y_train, config.nr_label, config.K, config.tau)
                val_acc, val_sep = 0, 0
                for i, data in enumerate(val_loader, 1):
                    # calc embedding
                    imgs, labels = data[0].to(device), data[1].to(device)
                    embed = net(imgs)
                    # calc output metrics
                    labels_pre = knn.predict(embed)
                    acc = (labels_pre==labels.long()).float().mean()
                    val_acc = (i-1)/i*val_acc+acc.item()/i
                    sep = torch.mm(embed, embed.permute(1, 0)).mean()
                    val_sep = (i-1)/i*val_sep+sep.item()/i
            print('_______validation_______')
            print('epoch: ', epoch, 'acc: ', '%.3f'%val_acc, 'sep: ', '%.3f'%val_sep)
            if opt_val_acc<=val_acc:
                opt_val_acc = val_acc
                torch.save(net.state_dict(), config.save_dir+'/opt')
            net.train()
    print('finish training. optimal val acc: ', '%.3f'%opt_val_acc)
    ret = {}
    np.save('./tmp/mb.npy', mb.content.numpy())
    net.eval()
    net.load_state_dict(torch.load(config.save_dir+'/opt'))
    # testing embedding in MemoryBank
    x_train, y_train = mb.content.to(device), torch.tensor(train_data.labels).to(device)
    knn = KNN(x_train, y_train, config.nr_label, config.K, config.tau)
    test_acc = 0
    for i, data in enumerate(test_loader, 1):
        # calc embedding
        imgs, labels = data[0].to(device), data[1].to(device)
        embed = net(imgs)
        # calc output metric
        labels_pre = knn.predict(embed)
        acc = (labels_pre==labels.long()).float().mean()
        test_acc = (i-1)/i*test_acc+acc.item()/i
    print('_______testing_______')
    print('acc: ', '%.3f'%test_acc)
    ret['test_by_mb'] = test_acc
    # testing embeding generated by net and train set
    x_train, y_train = [], []
    with torch.no_grad():
        for data in train_loader:
            imgs, labels = data[0].to(device), data[1]
            embed = net(imgs)
            x_train.extend(embed.cpu().tolist())
            y_train.extend(labels.tolist())
    knn = KNN(torch.tensor(x_train).to(device), torch.tensor(y_train).to(device), config.nr_label, config.K, config.tau)
    test_acc = 0
    for i, data in enumerate(test_loader, 1):
        # calc embedding
        imgs, labels = data[0].to(device), data[1].to(device)
        embed = net(imgs)
        # calc output metric
        labels_pre = knn.predict(embed)
        acc = (labels_pre==labels.long()).float().mean()
        test_acc = (i-1)/i*test_acc+acc.item()/i
    print('_______testing_______')
    print('acc: ', '%.3f'%test_acc)
    ret['test_by_train'] = test_acc
    # testing embeding generated by net and val set
    x_train, y_train = [], []
    with torch.no_grad():
        for data in val_loader:
            imgs, labels = data[0].to(device), data[1]
            embed = net(imgs)
            x_train.extend(embed.cpu().tolist())
            y_train.extend(labels.tolist())
    knn = KNN(torch.tensor(x_train).to(device), torch.tensor(y_train).to(device), config.nr_label, config.K, config.tau)
    test_acc = 0
    for i, data in enumerate(test_loader, 1):
        # calc embed
        imgs, labels = data[0].to(device), data[1].to(device)
        embed = net(imgs)
        # calc output metric
        labels_pre = knn.predict(embed)
        acc = (labels_pre==labels.long()).float().mean()
        test_acc = (i-1)/i*test_acc+acc.item()/i
    print('_______testing_______')
    print('acc: ', '%.3f'%test_acc)
    ret['test_by_val'] = test_acc
    return ret
