import torch.nn as nn
from torchvision import transforms
from torch.nn import functional as F

theta_deg = 180

transform1 = transforms.Compose([
    transforms.RandomRotation(theta_deg,interpolation=transforms.InterpolationMode.NEAREST),
    transforms.RandomHorizontalFlip(0.95)
])

def nin_block(out_channels, kernel_size, strides, padding):
    return nn.Sequential(
        nn.LazyConv2d(out_channels, kernel_size, strides, padding), nn.ReLU(),
        nn.LazyConv2d(out_channels, kernel_size=1), nn.ReLU(),
        nn.LazyConv2d(out_channels, kernel_size=1), 
        #batch normalization implementation
        nn.LazyBatchNorm2d())#,

class NiN(nn.Module):
    def __init__(self, lr=0.01, num_classes=2):
        super().__init__()
        #self.save_hyperparameters()
        self.net = nn.Sequential(
            nin_block(96, kernel_size=5, strides=3, padding=0),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.MaxPool2d(3, stride=2),
            nin_block(256, kernel_size=3, strides=1, padding=2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.MaxPool2d(3, stride=2),
            nin_block(384, kernel_size=3, strides=1, padding=1),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.MaxPool2d(3, stride=2),
            nin_block(num_classes, kernel_size=3, strides=1, padding=1),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten())

    def forward(self, x):
        return self.net(x)

class Residual(nn.Module):  #@save
    def __init__(self, num_channels, use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.LazyConv2d(num_channels, kernel_size=3, padding=1,
                                   stride=strides)
        self.conv2 = nn.LazyConv2d(num_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.LazyConv2d(num_channels, kernel_size=1,
                                       stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.LazyBatchNorm2d()
        self.bn2 = nn.LazyBatchNorm2d()

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)

class resnet(nn.Module):
    def b1(self):
        return nn.Sequential(
            nn.LazyConv2d(64, kernel_size=7, stride=2, padding=3),
            nn.Dropout(0.25),
            nn.LazyBatchNorm2d(), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

    def block(self, num_residuals, num_channels, first_block=False):
        blk = []
        for i in range(num_residuals):
            if i == 0 and not first_block:
                blk.append(Residual(num_channels, use_1x1conv=True, strides=2))
            else:
                blk.append(Residual(num_channels))
        return nn.Sequential(*blk)    

    def __init__(self, arch=((2, 64), (2, 128), (2, 256), (2, 512)), lr=0.1, num_classes=10):
        super(resnet, self).__init__()
        self.net = nn.Sequential(self.b1())
        for i, b in enumerate(arch):
            self.net.add_module(f'b{i+2}', self.block(*b, first_block=(i==0)))
        self.net.add_module('last', nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
            nn.LazyLinear(num_classes)))
        
    def forward(self, x):
        return self.net(x)



def init_weights(module):
    if type(module) == nn.Linear or type(module) == nn.Conv2d:
        nn.init.xavier_uniform_(module.weight)