from torch import nn
from torch.nn.init import kaiming_normal_

class ResBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels, stride=1, dropout=0.0):
        super(ResBlock, self).__init__()
        self.has_extra_conv = (in_channels!=out_channels)
        modules = []
        modules.append(nn.BatchNorm2d(in_channels))
        modules.append(nn.ReLU(True))
        modules.append(nn.Conv2d(in_channels, out_channels, 3, stride, 1))
        kaiming_normal_(modules[-1].weight)
        modules.append(nn.BatchNorm2d(out_channels))
        modules.append(nn.ReLU(True))
        if dropout>0:
            modules.append(nn.Dropout(dropout))
        modules.append(nn.Conv2d(out_channels, out_channels, 3, 1, 1))
        kaiming_normal_(modules[-1].weight)
        self.model = nn.Sequential(*modules)
        if self.has_extra_conv:
            self.extra_conv = nn.Conv2d(in_channels, out_channels, 1, stride, 0)
            kaiming_normal_(self.extra_conv.weight)
    
    def forward(self, inp):
        x = self.model(inp)
        if self.has_extra_conv:
            inp = self.extra_conv(inp)
        return x+inp

class WideResnet(nn.Module):
    
    def __init__(self, n, k, dropout=0, nr_label=100):
        super(WideResnet, self).__init__()
        modules = []
        # init conv
        modules.append(nn.Conv2d(3, 16, 3, 1, 1))
        kaiming_normal_(modules[-1].weight)
        # group1
        modules.append(ResBlock(16, 16*k, 1, dropout))
        for _ in range(n-1):
            modules.append(ResBlock(16*k, 16*k, 1, dropout))
        # group2
        modules.append(ResBlock(16*k, 32*k, 2, dropout))
        for _ in range(n-1):
            modules.append(ResBlock(32*k, 32*k, 1, dropout))
        # group3
        modules.append(ResBlock(32*k, 64*k, 2, dropout))
        for _ in range(n-1):
            modules.append(ResBlock(64*k, 64*k, 1, dropout))
        modules.append(nn.BatchNorm2d(64*k))
        modules.append(nn.ReLU(True))
        # avg pool
        modules.append(nn.AvgPool2d(8, 8))
        if dropout>0:
            modules.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*modules)
        self.fc = nn.Linear(64*k, nr_label)
    
    def forward(self, inp):
        x = self.model(inp)
        sz = x.size(0)
        x = self.fc(x.view(sz, -1))
        return x
