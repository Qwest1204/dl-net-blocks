import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = downsample
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)    
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        
        return out
    
 
class BottleneckResidualBlock(nn.Module):
    expansion = 4
    
    def __init__(self, in_channels, internal_channels, stride=1, downsample=None):
        super(BottleneckResidualBlock, self).__init__()
        
        out_channels = internal_channels*self.expansion
        
        self.conv1 = nn.Conv2d(in_channels, internal_channels, kernel_size=1,
                               bias=False)
        self.bn1   = nn.BatchNorm2d(internal_channels)

        self.conv2 = nn.Conv2d(internal_channels, internal_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(internal_channels)

        self.conv3 = nn.Conv2d(internal_channels, out_channels, kernel_size=1,
                               bias=False)
        self.bn3   = nn.BatchNorm2d(out_channels)

        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class InceptionModule(nn.Module):
    def __init__(self, in_channels, ch1x1, ch3x3_reduce, ch3x3, ch5x5_reduce, ch5x5, pool_proj):
        super(InceptionModule, self).__init__()
        
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, ch1x1, kernel_size=1),
            nn.BatchNorm2d(ch1x1),
            nn.ReLU(inplace=True)
        )
        
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, ch3x3_reduce, kernel_size=1),
            nn.BatchNorm2d(ch3x3_reduce),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch3x3_reduce, ch3x3, kernel_size=3, padding=1),
            nn.BatchNorm2d(ch3x3),
            nn.ReLU(inplace=True)
        )
                
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, ch5x5_reduce, kernel_size=1),
            nn.BatchNorm2d(ch5x5_reduce),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch5x5_reduce, ch5x5, kernel_size=5, padding=2),
            nn.BatchNorm2d(ch5x5),
            nn.ReLU(inplace=True)
        )
                        
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, pool_proj, kernel_size=1),
            nn.BatchNorm2d(pool_proj),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        
        out1 = self.branch1(x)
        out2 = self.branch2(x) 
        out3 = self.branch3(x) 
        out4 = self.branch4(x) 
        
        out = torch.cat([out1, out2, out3, out4], dim=1)
        return out
    
    
class DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate, bn_size=4, drop_rate=0.0):
        super(DenseLayer, self).__init__()

        inter_channels = bn_size * growth_rate

        self.norm1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, inter_channels, kernel_size=1, bias=False)

        self.norm2 = nn.BatchNorm2d(inter_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(inter_channels, growth_rate, kernel_size=3, padding=1, bias=False)

        self.drop_rate = drop_rate

    def forward(self, x):

        out = self.norm1(x)
        out = self.relu1(out)
        out = self.conv1(out)

        out = self.norm2(out)
        out = self.relu2(out)
        out = self.conv2(out)

        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)

        return torch.cat([x, out], dim=1)


class DenseBlock(nn.Module):
    def __init__(self, num_layers, in_channels, growth_rate, bn_size=4, drop_rate=0.0):
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList()
        current_channels = in_channels
        for i in range(num_layers):
            layer = DenseLayer(current_channels, growth_rate, bn_size, drop_rate)
            self.layers.append(layer)
            current_channels += growth_rate

        self.out_channels = current_channels

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    
class ResNeXtBlock(nn.Module):
    expansion = 2 #4

    def __init__(self, in_channels, internal_channels, stride=1, cardinality=32,
                 base_width=4, downsample=None):
        super(ResNeXtBlock, self).__init__()
        
        out_channels = internal_channels * self.expansion
        D = cardinality * base_width

        self.conv1 = nn.Conv2d(in_channels, D, kernel_size=1, stride=1, bias=False)
        self.bn1   = nn.BatchNorm2d(D)

        self.conv2 = nn.Conv2d(D, D, kernel_size=3, stride=stride, padding=1,
                               groups=cardinality, bias=False)
        self.bn2   = nn.BatchNorm2d(D)

        self.conv3 = nn.Conv2d(D, out_channels, kernel_size=1, stride=1, bias=False)
        self.bn3   = nn.BatchNorm2d(out_channels)

        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
    
    
class FireModule(nn.Module):
    def __init__(self, in_channels, squeeze_channels, expand1x1_channels, expand3x3_channels):
        super(FireModule, self).__init__()
        
        self.squeeze = nn.Sequential(
            nn.Conv2d(in_channels, squeeze_channels, kernel_size=1),
            nn.BatchNorm2d(squeeze_channels),
            nn.ReLU(inplace=True)
        )

        self.expand1x1 = nn.Sequential(
            nn.Conv2d(squeeze_channels, expand1x1_channels, kernel_size=1),
            nn.BatchNorm2d(expand1x1_channels),
            nn.ReLU(inplace=True)
        )

        self.expand3x3 = nn.Sequential(
            nn.Conv2d(squeeze_channels, expand3x3_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(expand3x3_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        
        squeezed = self.squeeze(x)
        out1 = self.expand1x1(squeezed)
        out3 = self.expand3x3(squeezed)
        
        return torch.cat([out1, out3], dim=1)


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=None, dilation=1, bias=False):
        super(DepthwiseSeparableConv, self).__init__()
        
        if padding is None:
            padding = kernel_size // 2 * dilation

        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=kernel_size, stride=stride,
            padding=padding, dilation=dilation, groups=in_channels, bias=bias
        )
        self.bn_dw = nn.BatchNorm2d(in_channels)

        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.bn_pw = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        x = self.depthwise(x)
        x = self.bn_dw(x)
        x = self.relu(x)

        x = self.pointwise(x)
        x = self.bn_pw(x)
        x = self.relu(x)
        return x

    
class AtrousConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=2, stride=1, bias=False):
        super(AtrousConv, self).__init__()
        
        padding = (kernel_size - 1) * dilation // 2

        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, dilation=dilation, bias=bias
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x