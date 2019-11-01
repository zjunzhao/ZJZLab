import torch
from config import Config
from model import WideResnet
import hiddenlayer as hl

net = WideResnet(Config())
net.eval()
im = hl.build_graph(net, torch.zeros([1, 3, 32, 32]))
im.save(path='WideResnet_28_10', format='png')
