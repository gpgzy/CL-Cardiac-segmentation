""" Full assembly of the parts to form the complete network """

import torch.nn.functional as F

from models.modeling.unet_parts import *
from models.modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.freeze_bn = False
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256,dilation=1)
        self.down3 = Down(256, 512,dilation=2)# our is 2
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor,dilation=4)# our is 4
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        # self.layer5 = nn.Sequential(
        #         nn.Conv2d(64, 256,
        #                   kernel_size=1, stride=1, bias=False),
        #         nn.BatchNorm2d(256),
        #     )
        # self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        # logits = self.outc(x)
        # return logits
        # x = self.layer5(x)
        return x

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, SynchronizedBatchNorm2d):
                m.eval()
            elif isinstance(m, nn.BatchNorm2d):
                m.eval()
    def get_module_params(self):
        modules = [self.up1,self.up2,self.up3,self.up4]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if self.freeze_bn:
                    if isinstance(m[1], nn.Conv2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p
                else:
                    if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) or isinstance(m[1],nn.ConvTranspose2d) \
                            or isinstance(m[1], nn.BatchNorm2d) or isinstance(m[1], nn.SyncBatchNorm):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p

    def get_backbone_params(self):
        modules = [self.inc,self.down1, self.down2,self.down3,self.down4]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if self.freeze_bn:
                    if isinstance(m[1], nn.Conv2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p
                else:
                    if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                            or isinstance(m[1], nn.BatchNorm2d) or isinstance(m[1], nn.SyncBatchNorm):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p
