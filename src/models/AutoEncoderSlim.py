"""
See: https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
"""
import torch.nn as nn
import torch
import torchvision
from models.autoencoder import AutoEncoder
from collections import OrderedDict
#############################
# Static params: Config
#############################
conv_kernel_size = 3
img_input_channels = 1
cfg = {
    '19normal': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    '16normal': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    '11normal': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],

    # models TinyImgnet
    'small_UNet': [32, 'M', 64, 'M', 128, 'M', 256, 'M', 512, 'U', 256, 'U', 128, 'U', 64, 'U', 32 ],  # 334,016 feat params,
    'base_VGG9': [64, 'M', 64, 'M', 128, 128, 'M', 256, 256, 'M'],  # 1.145.408 feat params
    'wide_VGG9': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M'],  # 4.500.864 feat params
    'deep_VGG22': [64, 'M', 64, 64, 64, 64, 64, 64, 'M', 128, 128, 128, 128, 128, 128, 'M',
                   256, 256, 256, 256, 256, 256, 'M'],  # 4.280.704 feat params
}

def block(in_channels, features):
    return [
            nn.Conv2d(in_channels=in_channels, out_channels=features, kernel_size=3, padding=1, bias=False,),
            nn.BatchNorm2d(num_features=features),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=1, bias=False,),
            nn.BatchNorm2d(num_features=features),
            nn.ReLU(inplace=True),
            ]

class catenate(nn.Module):
    def __init__(self):
        super(catenate, self).__init__()
    def forward(self, x1, x2):
        return torch.cat((x1, x2), 1)

def make_layers(nfilters=32, n_levels=5):
    layers = []
    # blocks = []
    in_channels = img_input_channels

    for i in range(n_levels): # encoder part
        block_ = block(in_channels, nfilters * (2 ** i))
        # blocks += [block_]
        layers += block_
        layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        in_channels = nfilters * (2 ** i)
    # bottleneck
    layers += block(in_channels, nfilters * (2 ** n_levels))
    in_channels = nfilters * (2 ** n_levels)

    for i in range(n_levels): # decoder part
        up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        layers += [up]
        block_ = block(in_channels//2, nfilters * (2 ** (n_levels - 1 - i)))
        # blocks += [block_]
        layers += block_
        in_channels = nfilters * (2 ** (n_levels - 1 - i))

    last_conv = nn.Conv2d(nfilters, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True)
    layers += [last_conv]
    return nn.Sequential(*layers)


class AutoEncoderSlim(AutoEncoder):
    """
    Creates VGG feature extractor from config and custom classifier.
    """

    def __init__(self, config='11Slim', num_classes=1, init_weights=True,
                 classifier_inputdim=256 * 256, classifier_dim1=512, classifier_dim2=512, batch_norm=False,
                 dropout=False):
        features = make_layers(nfilters=32, n_levels=3)
        super(AutoEncoderSlim, self).__init__(features)

        if hasattr(self, 'avgpool'):  # Compat Pytorch>1.0.0
            self.avgpool = torch.nn.Identity()

        self.classifier = nn.Sequential(
            nn.Sigmoid()
        )
        # if init_weights:
        #     self._initialize_weights()
