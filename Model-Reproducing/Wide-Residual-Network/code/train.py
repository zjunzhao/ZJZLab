import numpy as np
import torch
import torchvision
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import Subset, DataLoader
from model import WideResnet

def adjust_lr(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr'] *= 0.2

def train_process(config):
    # set up random seed
    torch.manual_seed(config.random_seed)
    # set up device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # set up data
    num_workers = 2
    train_transform = transforms.Compose(
                      [transforms.Pad(4, padding_mode='reflect'),
                       transforms.RandomHorizontalFlip(),
                       transforms.RandomCrop(32),
                       transforms.ToTensor(),
                       transforms.Normalize(config.data_mean, config.data_std)])
    eval_transform = transforms.Compose(
                     [transforms.ToTensor(),
                      transforms.Normalize(config.data_mean, config.data_std)])
    train_indices = list(np.random.choice(50000, 45000, False))
    val_indices = list(set(range(50000))-set(train_indices))
    train_data = CIFAR10(config.data_dir, True, train_transform, download=True)
    train_data = Subset(train_data, train_indices)
    train_loader = DataLoader(train_data, config.train_batch_size, True, num_workers=num_workers)
    val_data = CIFAR10(config.data_dir, True, eval_transform, download=True)
    val_data = Subset(val_data, val_indices)
    val_loader = DataLoader(val_data, config.eval_batch_size, False, num_workers=num_workers)
    test_data = CIFAR10(config.data_dir, False, eval_transform, download=True)
    test_loader = DataLoader(test_data, config.eval_batch_size, False, num_workers=num_workers)
    # set up nn, loss and optimizer
    net = WideResnet(config)
    if torch.cuda.device_count()>1:
        net = torch.nn.DataParallel(net)
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    lr, momentum, weight_decay = config.lr, config.momentum, config.weight_decay
    optimizer = torch.optim.SGD(net.parameters(), lr, momentum, weight_decay=weight_decay)
    # training process
    opt_acc = 0.0
    train_loss_list = []
    train_acc_list = []
    val_loss_list = []
    val_acc_list = []
    for epoch in range(1, config.nr_epoch+1):
        if epoch in [61, 121, 181]:
            adjust_lr(optimizer)
        train_loss, train_acc = 0.0, 0.0
        for i, data in enumerate(train_loader, 1):
            imgs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            logits = net(imgs)
            loss = criterion(logits, labels)
            train_loss = 0.5*train_loss+0.5*loss.item()
            _, labels_pre = torch.max(logits, 1)
            acc = (labels_pre==labels).float().mean()
            train_acc = 0.5*train_acc+0.5*acc.item()
            loss.backward()
            optimizer.step()
            if i%config.show_interval==0:
                print('_______training_______')
                print('epoch: ', epoch, 'step: ', i, 'loss: ', '%.3f'%train_loss, 'acc: ', '%.3f'%train_acc)
        if epoch%config.val_interval==0:
            net.eval()
            with torch.no_grad():
                val_loss, val_acc = 0.0, 0.0
                for i, data in enumerate(val_loader, 1):
                    imgs, labels = data[0].to(device), data[1].to(device)
                    logits = net(imgs)
                    loss = criterion(logits, labels)
                    val_loss = 1.0*(i-1)/i*val_loss+loss.item()/i
                    _, labels_pre = torch.max(logits, 1)
                    acc = (labels_pre==labels).float().mean()
                    val_acc = 1.0*(i-1)/i*val_acc+acc.item()/i
            train_loss_list.append(train_loss)
            train_acc_list.append(train_acc)
            val_loss_list.append(val_loss)
            val_acc_list.append(val_acc)
            print('_______validation_______')
            print('epoch: ', epoch, 'loss: ', '%.3f'%val_loss, 'acc: ', '%.3f'%val_acc)
            if opt_acc<val_acc:
                opt_acc=val_acc
                torch.save(net.state_dict(), config.save_dir+'/opt')
            net.train()
    print('finish training. optimal val acc: ', '%.3f'%opt_acc)
    # testing process
    net.load_state_dict(torch.load(config.save_dir+'/opt'))
    net.eval()
    with torch.no_grad():
        test_loss, test_acc = 0.0, 0.0
        for i, data in enumerate(test_loader, 1):
            imgs, labels = data[0].to(device), data[1].to(device)
            logits = net(imgs)
            loss = criterion(logits, labels)
            test_loss = 1.0*(i-1)/i*test_loss+loss.item()/i
            _, labels_pre = torch.max(logits, 1)
            acc = (labels_pre==labels).float().mean()
            test_acc = 1.0*(i-1)/i*test_acc+acc.item()/i
    print('_______testing_______')
    print('loss: ', '%.3f'%test_loss, 'acc: ', '%.3f'%test_acc)
    history = {'train_loss_list':train_loss_list, 'train_acc_list':train_acc_list,
               'val_loss_list':val_loss_list, 'val_acc_list':val_acc_list,
               'test_loss':test_loss, 'test_acc':test_acc}
    return history
