#########################################################################
## Project: Explain-ability and Interpret-ability for segmentation models
## Purpose: Python file to deep copy FCN_resnet101 Model by replacing
##          inplace true ReLU activations
## Author: Arnab Das
#########################################################################

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import Sequential

class R2Plus1dStem4MRI(nn.Sequential):
    """R(2+1)D stem is different than the default one as it uses separated 3D convolution
    """

    def __init__(self):
        super(R2Plus1dStem4MRI, self).__init__(
            nn.Conv3d(1, 45, kernel_size=(1, 7, 7),
                      stride=(1, 2, 2), padding=(0, 3, 3),
                      bias=False),
            nn.BatchNorm3d(45),
            nn.ReLU(inplace=True),

            nn.Conv3d(45, 64, kernel_size=(3, 1, 1),
                      stride=(1, 1, 1), padding=(1, 0, 0),
                      bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True))

class modifybasicstem(nn.Sequential):
    """The default conv-batchnorm-relu stem
    """
    def __init__(self):
        super(modifybasicstem, self).__init__(
            nn.Conv3d(1, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2),
                      padding=(1, 3, 3), bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True))

class Bottleneck(torch.nn.Module):
    def __init__(self, conv1, bn1, conv2, bn2, conv3, bn3, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1
        self.bn1 = bn1
        self.conv2 = conv2
        self.bn2 = bn2
        self.conv3 = conv3
        self.bn3 = bn3
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = nn.ReLU()(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = nn.ReLU()(out)
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = nn.ReLU()(out)

        return out


class Net(torch.nn.Module):
    def __init__(self, resModel):
        super(Net, self).__init__()
        layer1 = resModel.backbone.layer1
        layer2 = resModel.backbone.layer2
        layer3 = resModel.backbone.layer3
        layer4 = resModel.backbone.layer4
        self.conv1 = resModel.backbone.conv1
        self.bn1 = resModel.backbone.bn1
        self.relu1 = nn.ReLU(inplace=False)
        self.maxpool1 = resModel.backbone.maxpool
        self.layer1 = Sequential(
            Bottleneck(layer1[0].conv1, layer1[0].bn1, layer1[0].conv2, layer1[0].bn2, layer1[0].conv3, layer1[0].bn3,
                       layer1[0].downsample),
            Bottleneck(layer1[1].conv1, layer1[1].bn1, layer1[1].conv2, layer1[1].bn2, layer1[1].conv3, layer1[1].bn3),
            Bottleneck(layer1[2].conv1, layer1[2].bn1, layer1[2].conv2, layer1[2].bn2, layer1[2].conv3, layer1[2].bn3))

        self.layer2 = Sequential(
            Bottleneck(layer2[0].conv1, layer2[0].bn1, layer2[0].conv2, layer2[0].bn2, layer2[0].conv3, layer2[0].bn3,
                       layer2[0].downsample),
            Bottleneck(layer2[1].conv1, layer2[1].bn1, layer2[1].conv2, layer2[1].bn2, layer2[1].conv3, layer2[1].bn3),
            Bottleneck(layer2[2].conv1, layer2[2].bn1, layer2[2].conv2, layer2[2].bn2, layer2[2].conv3, layer2[2].bn3),
            Bottleneck(layer2[3].conv1, layer2[3].bn1, layer2[3].conv2, layer2[3].bn2, layer2[3].conv3, layer2[3].bn3))

        self.layer3 = Sequential(
            Bottleneck(layer3[0].conv1, layer3[0].bn1, layer3[0].conv2, layer3[0].bn2, layer3[0].conv3, layer3[0].bn3,
                       layer3[0].downsample),
            Bottleneck(layer3[1].conv1, layer3[1].bn1, layer3[1].conv2, layer3[1].bn2, layer3[1].conv3, layer3[1].bn3),
            Bottleneck(layer3[2].conv1, layer3[2].bn1, layer3[2].conv2, layer3[2].bn2, layer3[2].conv3, layer3[2].bn3),
            Bottleneck(layer3[3].conv1, layer3[3].bn1, layer3[3].conv2, layer3[3].bn2, layer3[3].conv3, layer3[3].bn3),
            Bottleneck(layer3[4].conv1, layer3[4].bn1, layer3[4].conv2, layer3[4].bn2, layer3[4].conv3, layer3[4].bn3),
            Bottleneck(layer3[5].conv1, layer3[5].bn1, layer3[5].conv2, layer3[5].bn2, layer3[5].conv3, layer3[5].bn3),
            Bottleneck(layer3[6].conv1, layer3[6].bn1, layer3[6].conv2, layer3[6].bn2, layer3[6].conv3, layer3[6].bn3),
            Bottleneck(layer3[7].conv1, layer3[7].bn1, layer3[7].conv2, layer3[7].bn2, layer3[7].conv3, layer3[7].bn3),
            Bottleneck(layer3[8].conv1, layer3[8].bn1, layer3[8].conv2, layer3[8].bn2, layer3[8].conv3, layer3[8].bn3),
            Bottleneck(layer3[9].conv1, layer3[9].bn1, layer3[9].conv2, layer3[9].bn2, layer3[9].conv3, layer3[9].bn3),
            Bottleneck(layer3[10].conv1, layer3[10].bn1, layer3[10].conv2, layer3[10].bn2, layer3[10].conv3,
                       layer3[10].bn3),
            Bottleneck(layer3[11].conv1, layer3[11].bn1, layer3[11].conv2, layer3[11].bn2, layer3[11].conv3,
                       layer3[11].bn3),
            Bottleneck(layer3[12].conv1, layer3[12].bn1, layer3[12].conv2, layer3[12].bn2, layer3[12].conv3,
                       layer3[12].bn3),
            Bottleneck(layer3[13].conv1, layer3[13].bn1, layer3[13].conv2, layer3[13].bn2, layer3[13].conv3,
                       layer3[13].bn3),
            Bottleneck(layer3[14].conv1, layer3[14].bn1, layer3[14].conv2, layer3[14].bn2, layer3[14].conv3,
                       layer3[14].bn3),
            Bottleneck(layer3[15].conv1, layer3[15].bn1, layer3[15].conv2, layer3[15].bn2, layer3[15].conv3,
                       layer3[15].bn3),
            Bottleneck(layer3[16].conv1, layer3[16].bn1, layer3[16].conv2, layer3[16].bn2, layer3[16].conv3,
                       layer3[16].bn3),
            Bottleneck(layer3[17].conv1, layer3[17].bn1, layer3[17].conv2, layer3[17].bn2, layer3[17].conv3,
                       layer3[17].bn3),
            Bottleneck(layer3[18].conv1, layer3[18].bn1, layer3[18].conv2, layer3[18].bn2, layer3[18].conv3,
                       layer3[18].bn3),
            Bottleneck(layer3[19].conv1, layer3[19].bn1, layer3[19].conv2, layer3[19].bn2, layer3[19].conv3,
                       layer3[19].bn3),
            Bottleneck(layer3[20].conv1, layer3[20].bn1, layer3[20].conv2, layer3[20].bn2, layer3[20].conv3,
                       layer3[20].bn3),
            Bottleneck(layer3[21].conv1, layer3[21].bn1, layer3[21].conv2, layer3[21].bn2, layer3[21].conv3,
                       layer3[21].bn3),
            Bottleneck(layer3[22].conv1, layer3[22].bn1, layer3[22].conv2, layer3[22].bn2, layer3[22].conv3,
                       layer3[22].bn3))

        self.layer4 = Sequential(
            Bottleneck(layer4[0].conv1, layer4[0].bn1, layer4[0].conv2, layer4[0].bn2, layer4[0].conv3, layer4[0].bn3,
                       layer4[0].downsample),
            Bottleneck(layer4[1].conv1, layer4[1].bn1, layer4[1].conv2, layer4[1].bn2, layer4[1].conv3, layer4[1].bn3),
            Bottleneck(layer4[2].conv1, layer4[2].bn1, layer4[2].conv2, layer4[2].bn2, layer4[2].conv3, layer4[2].bn3))

        self.layer5 = resModel.classifier
        # self.layer6 = resModel.aux_classifier

    def forward(self, x):
        input_shape = x.shape[-2:]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        #op_max = torch.argmax(x, dim=1, keepdim=True)
        #selected_inds = torch.zeros_like(x[0:]).scatter_(1, op_max, 1)
        #return (x * selected_inds).sum(dim=(2, 3))
        return  x
