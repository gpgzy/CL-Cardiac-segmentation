import math
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import torch
from models.modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
import os
from torchvision import models
from models.backbones.module_helper import ModuleHelper
from collections import OrderedDict


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        # if dilation > 1:
        #     raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride,dilation=dilation)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes,dilation=dilation)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

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


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, BatchNorm=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               dilation=dilation, padding=dilation, bias=False)
        self.bn2 = BatchNorm(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = BatchNorm(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, output_stride, BatchNorm, pretrained=True, resnet_layers=101):

        self.resnet_layers = resnet_layers
        self.inplanes = 64
        super(ResNet, self).__init__()
        blocks = [1, 2, 4]
        if output_stride == 16:
            strides = [1, 2, 2, 1]
            dilations = [1, 1, 1, 2]
        elif output_stride == 8:
            strides = [1, 2, 1, 1]
            dilations = [1, 1, 2, 4]
        else:
            raise NotImplementedError

        # Modules
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = BatchNorm(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], stride=strides[0], dilation=dilations[0],
                                       BatchNorm=BatchNorm)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=strides[1], dilation=dilations[1],
                                       BatchNorm=BatchNorm)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=strides[2], dilation=dilations[2],
                                       BatchNorm=BatchNorm)
        self.layer4 = self._make_MG_unit(block, 512, blocks=blocks, stride=strides[3], dilation=dilations[3],
                                         BatchNorm=BatchNorm)
        # self.layer4 = self._make_layer(block, 512, layers[3], stride=strides[3], dilation=dilations[3], BatchNorm=BatchNorm)
        self._init_weight()

        if pretrained:
            self._load_pretrained_model()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, BatchNorm=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation, downsample, BatchNorm))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation, BatchNorm=BatchNorm))

        return nn.Sequential(*layers)

    def _make_MG_unit(self, block, planes, blocks, stride=1, dilation=1, BatchNorm=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation=blocks[0] * dilation,
                            downsample=downsample, BatchNorm=BatchNorm))
        self.inplanes = planes * block.expansion
        for i in range(1, len(blocks)):
            layers.append(block(self.inplanes, planes, stride=1,
                                dilation=blocks[i] * dilation, BatchNorm=BatchNorm))

        return nn.Sequential(*layers)

    def forward(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        low_level_feat = x
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x, low_level_feat

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.SyncBatchNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _load_pretrained_model(self):
        # pretrain_dict = model_zoo.load_url('https://download.pytorch.org/models/resnet101-5d3b4d8f.pth')
        if self.resnet_layers == 101:
            path = 'pretrained/resnet101-5d3b4d8f.pth'
        elif self.resnet_layers == 50:
            path = 'pretrained/resnet50-19c8e357.pth'
        else:
            raise ValueError("{} layers not supported".format(self.resnet_layers))

        if os.path.exists(path):
            pretrain_dict = path
        else:
            raise ValueError("The path {} not exists".format(path))

        print("load pretrained weight from {}".format(pretrain_dict))
        pretrain_dict = torch.load(pretrain_dict)
        model_dict = {}
        state_dict = self.state_dict()
        for k, v in pretrain_dict.items():
            if k in state_dict:
                model_dict[k] = v
        state_dict.update(model_dict)
        self.load_state_dict(state_dict)

# class Bottleneck(nn.Module):
#     expansion = 4
#
#     def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, BatchNorm=None):
#         super(Bottleneck, self).__init__()
#         self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
#         self.bn1 = BatchNorm(planes)
#         self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
#                                dilation=dilation, padding=dilation, bias=False)
#         self.bn2 = BatchNorm(planes)
#         self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
#         self.bn3 = BatchNorm(planes * 4)
#         self.relu = nn.ReLU(inplace=True)
#         self.downsample = downsample
#         self.stride = stride
#         self.dilation = dilation
#
#     def forward(self, x):
#         residual = x
#
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#
#         out = self.conv2(out)
#         out = self.bn2(out)
#         out = self.relu(out)
#
#         out = self.conv3(out)
#         out = self.bn3(out)
#
#         if self.downsample is not None:
#             residual = self.downsample(x)
#
#         out += residual
#         out = self.relu(out)
#
#         return out
#
# class ResNet(nn.Module):
#
#     def __init__(self, block, layers, output_stride, BatchNorm, pretrained=True, resnet_layers=101):
#
#         self.resnet_layers = resnet_layers
#         self.inplanes = 64
#         super(ResNet, self).__init__()
#         blocks = [1, 2, 4]
#         if output_stride == 16:
#             strides = [1, 2, 2, 1]
#             dilations = [1, 1, 1, 2]
#         elif output_stride == 8:
#             strides = [1, 2, 1, 1]
#             dilations = [1, 1, 2, 4]
#         else:
#             raise NotImplementedError
#
#         # Modules
#         self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
#                                 bias=False)
#         self.bn1 = BatchNorm(64)
#         self.relu = nn.ReLU(inplace=True)
#         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#
#         self.layer1 = self._make_layer(block, 64, layers[0], stride=strides[0], dilation=dilations[0], BatchNorm=BatchNorm)
#         self.layer2 = self._make_layer(block, 128, layers[1], stride=strides[1], dilation=dilations[1], BatchNorm=BatchNorm)
#         self.layer3 = self._make_layer(block, 256, layers[2], stride=strides[2], dilation=dilations[2], BatchNorm=BatchNorm)
#         self.layer4 = self._make_MG_unit(block, 512, blocks=blocks, stride=strides[3], dilation=dilations[3], BatchNorm=BatchNorm)
#         # self.layer4 = self._make_layer(block, 512, layers[3], stride=strides[3], dilation=dilations[3], BatchNorm=BatchNorm)
#         self._init_weight()
#
#         # if pretrained:
#         #     self._load_pretrained_model()
#
#     def _make_layer(self, block, planes, blocks, stride=1, dilation=1, BatchNorm=None):
#         downsample = None
#         if stride != 1 or self.inplanes != planes * block.expansion:
#             # print(planes* block.expansion)
#             downsample = nn.Sequential(
#                 nn.Conv2d(self.inplanes, planes * block.expansion,
#                           kernel_size=1, stride=stride, bias=False),
#                 BatchNorm(planes * block.expansion),
#             )
#
#         layers = []
#         layers.append(block(self.inplanes, planes, stride, dilation, downsample, BatchNorm))
#         # layers.append(block(self.inplanes, planes, stride, downsample, BatchNorm))
#         self.inplanes = planes * block.expansion
#         for i in range(1, blocks):
#             layers.append(block(self.inplanes, planes, stride,dilation, downsample, BatchNorm))
#             # layers.append(block(self.inplanes, planes, stride, downsample, BatchNorm))
#         return nn.Sequential(*layers)
#
#     def _make_MG_unit(self, block, planes, blocks, stride=1, dilation=1, BatchNorm=None):
#         downsample = None
#         if stride != 1 or self.inplanes != planes * block.expansion:
#             downsample = nn.Sequential(
#                 nn.Conv2d(self.inplanes, planes * block.expansion,
#                           kernel_size=1, stride=stride, bias=False),
#                 BatchNorm(planes * block.expansion),
#             )
#
#         layers = []
#         # layers.append(block(self.inplanes, planes, stride, dilation=blocks[0]*dilation,
#         #                     downsample=downsample, BatchNorm=BatchNorm))
#         layers.append(block(self.inplanes, planes, stride, blocks[0]*dilation, downsample, BatchNorm))
#
#         self.inplanes = planes * block.expansion
#         for i in range(1, len(blocks)):
#             # layers.append(block(self.inplanes, planes, stride=1,
#             #                     dilation=blocks[i]*dilation, BatchNorm=BatchNorm))
#             layers.append(block(self.inplanes, planes, 1, blocks[i] * dilation, downsample, BatchNorm))
#         return nn.Sequential(*layers)
#
#     def forward(self, input):
#         x = self.conv1(input)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.maxpool(x)
#         x = self.layer1(x)
#         low_level_feat = x
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)
#         return x, low_level_feat
#
#     def _init_weight(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#                 m.weight.data.normal_(0, math.sqrt(2. / n))
#             elif isinstance(m, SynchronizedBatchNorm2d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()
#             elif isinstance(m, nn.BatchNorm2d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()
#             elif isinstance(m, nn.SyncBatchNorm):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()
#
#     def _load_pretrained_model(self):
#         # pretrain_dict = model_zoo.load_url('https://download.pytorch.org/models/resnet101-5d3b4d8f.pth')
#         if self.resnet_layers == 101:
#             path = 'pretrained/resnet101-5d3b4d8f.pth'
#         elif self.resnet_layers == 50:
#             path = 'pretrained/resnet50-19c8e357.pth'
#         elif self.resnet_layers == 18:
#             path = 'pretrained/resnet18-5c106cde.pth'
#             # path = 'pretrained/resnet50-19c8e357.pth'
#         else:
#             raise ValueError("{} layers not supported".format(self.resnet_layers))
#
#         if os.path.exists(path):
#             pretrain_dict = path
#         else:
#             raise ValueError("The path {} not exists".format(path))
#
#         print("load pretrained weight from {}".format(pretrain_dict))
#         pretrain_dict = torch.load(pretrain_dict)
#         model_dict = {}
#         state_dict = self.state_dict()
#         for k, v in pretrain_dict.items():
#             if k in state_dict:
#                 model_dict[k] = v
#         state_dict.update(model_dict)
#         self.load_state_dict(state_dict)
# class ResNet(nn.Module):
#     def __init__(self, block, layers, pretrained=True):  # layers=参数列表 block选择不同的类
#         self.inplanes = 64
#         super(ResNet, self).__init__()
#         self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,bias=False)
#         self.bn1 = nn.BatchNorm2d(64)
#         self.relu = nn.ReLU(inplace=True)
#         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#         self.layer1 = self._make_layer(block, 64, layers[0])
#         self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
#         self.layer3 = self._make_layer(block, 256, layers[2], stride=2,dilation=2)
#         # self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
#         self.layer4 = self._make_MG_unit(block, 512, blocks=[1,2,4], stride=1, dilation=4,
#                                          BatchNorm=nn.BatchNorm2d)
#         # self.avgpool = nn.AvgPool2d(7)
#         # self.fc = nn.Linear(512 * block.expansion, num_classes)
#         # self.layer5 = nn.Sequential(
#         #         nn.Conv2d(512, 2048,
#         #                   kernel_size=1, stride=1, bias=False),
#         #         nn.BatchNorm2d(2048),
#         #     )
#
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#                 m.weight.data.normal_(0, math.sqrt(2. / n))
#             elif isinstance(m, nn.BatchNorm2d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()
#         if pretrained:
#             self._load_pretrained_model()
#
#     def _make_layer(self, block, planes, blocks, stride=1,dilation=1):
#         downsample = None
#         if stride != 1 or self.inplanes != planes * block.expansion:
#             downsample = nn.Sequential(
#                 nn.Conv2d(self.inplanes, planes * block.expansion,
#                           kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(planes * block.expansion),
#             )
#
#         layers = []
#         layers.append(block(self.inplanes, planes, stride, downsample,dilation=dilation)) # 每个blocks的第一个residual结构保存在layers列表中。
#         self.inplanes = planes * block.expansion
#         for i in range(1, blocks):
#             layers.append(block(self.inplanes, planes))   #该部分是将每个blocks的剩下residual 结构保存在layers列表中，这样就完成了一个blocks的构造。
#
#         return nn.Sequential(*layers)
#
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.maxpool(x)
#         low_level_feat = x
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)
#         # x = self.layer5(x)
#         # x = self.avgpool(x)
#         # x = x.view(x.size(0), -1)   # 将输出结果展成一行
#         # x = self.fc(x)
#
#         return x,low_level_feat
#
#     def _make_MG_unit(self, block, planes, blocks, stride=1, dilation=1, BatchNorm=None):
#         downsample = None
#         if stride != 1 or self.inplanes != planes * block.expansion:
#             downsample = nn.Sequential(
#                 nn.Conv2d(self.inplanes, planes * block.expansion,
#                           kernel_size=1, stride=stride, bias=False),
#                 BatchNorm(planes * block.expansion),
#             )
#
#         layers = []
#         layers.append(block(self.inplanes, planes, stride, dilation=blocks[0] * dilation,
#                             downsample=downsample))
#         self.inplanes = planes * block.expansion
#         # self.inplanes = planes * 4
#         for i in range(1, len(blocks)):
#             # print(self.inplanes)
#             # print(planes)
#             layers.append(block(self.inplanes, planes, stride=1,
#                                 dilation=blocks[i] * dilation))
#         # layers.append(block(self.inplanes, planes*4, stride=1,
#         #                         dilation=1))
#         return nn.Sequential(*layers)
#     def _load_pretrained_model(self):
#         # pretrain_dict = model_zoo.load_url('https://download.pytorch.org/models/resnet101-5d3b4d8f.pth')
#         # path = '/home/gzy/CAC_Resnet18/pretrained/resnet18-5c106cde.pth'
#         # path = 'pretrained/resnet18-5c106cde.pth'
#         path = 'pretrained/resnet50-19c8e357.pth'
#         if os.path.exists(path):
#             pretrain_dict = path
#         else:
#             raise ValueError("The path {} not exists".format(path))
#
#         print("load pretrained weight from {}".format(pretrain_dict))
#         pretrain_dict = torch.load(pretrain_dict)
#         model_dict = {}
#         state_dict = self.state_dict()
#         for k, v in pretrain_dict.items():
#             if k in state_dict:
#                 model_dict[k] = v
#         state_dict.update(model_dict)
#         self.load_state_dict(state_dict)

def ResNet101(output_stride, BatchNorm, pretrained=True):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], output_stride, BatchNorm, pretrained=pretrained)
    return model

def ResNet50(output_stride, BatchNorm, pretrained=True):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], output_stride, BatchNorm, pretrained=pretrained, resnet_layers=50)
    return model
# 下面model还需要改成对应18的
def ResNet18(output_stride, BatchNorm, pretrained=True):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    # model = ResNet(BasicBlock, [2, 2, 2, 2], output_stride, BatchNorm=nn.BatchNorm2d, pretrained=pretrained, resnet_layers=18)
    # model = models.resnet18(pretrained=False)
    # model = ModuleHelper.load_model(model, pretrained=pretrained)
    print('Constructs a ResNet-18 model')
    # model = ResNet(BasicBlock, [2, 2, 2, 2], output_stride, BatchNorm, pretrained=pretrained, resnet_layers=18)
    model = ResNet(BasicBlock, [2, 2, 2, 2])
    return model

if __name__ == "__main__":
    import torch
    # model = ResNet101(BatchNorm=nn.BatchNorm2d, pretrained=True, output_stride=8)
    model = ResNet18(BatchNorm=nn.BatchNorm2d, pretrained=False, output_stride=8)
    input = torch.rand(1,3,512, 512)
    output, low_level_feat = model(input)
    print(output.size())
    print(low_level_feat.size())
