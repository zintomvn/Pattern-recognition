import os
import sys
import torch
from torch import nn
import torch.nn.functional as F

from .base_block import Conv_block, Basic_block
from .backbones import build_backbone

sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', '..', '..', '..'))
from common.utils.model_init import init_weights


class FeatExtractor(nn.Module):
    """Thin wrapper delegating to a BackboneBase instance.

    Backward-compatible: old configs without 'backbone' key default to 'custom_cnn'.
    """

    def __init__(self, backbone_name='custom_cnn', **backbone_kwargs):
        """
        Args:
            backbone_name (str): registry key ('custom_cnn', 'resnet18', 'resnet50', 'vgg16')
            **backbone_kwargs: forwarded to the backbone constructor.
                               For 'custom_cnn': in_ch, AdapNorm, AdapNorm_attention_flag,
                               model_initial. Ignored for pretrained backbones.
        """
        super(FeatExtractor, self).__init__()
        self.backbone_name = backbone_name
        self.backbone = build_backbone(backbone_name, **backbone_kwargs)

    def forward(self, x, params=None):
        return self.backbone.forward(x, params)

    def get_meta_params(self):
        return self.backbone.get_meta_params()


class FeatEmbedder(nn.Module):
    '''
        Args:
            in_ch (int): the channel numbers of input features
            AdapNorm (bool): 
                'True' allow the Conv_block to combine BN and IN
                'False' allow the Conv_block to use BN
            AdapNorm_attention_flag:
                '1layer' allow the Conv_block to use 1layer FC to generate the balance factor
                '2layer' allow the Conv_block to use 2layer FC to generate the balance factor
            model_initial:
                'kaiming' allow the Conv_block to use 'kaiming' methods to initialize the networks
    '''

    def __init__(self, in_ch=384, AdapNorm=True, AdapNorm_attention_flag='1layer', model_initial='kaiming'):
        super(FeatEmbedder, self).__init__()
        self.AdapNorm = AdapNorm
        self.AdapNorm_attention_flag = AdapNorm_attention_flag
        self.model_initial = model_initial

        self.conv_block1 = Conv_block(in_ch, 128, self.AdapNorm, self.AdapNorm_attention_flag, self.model_initial)
        self.conv_block2 = Conv_block(128, 256, self.AdapNorm, self.AdapNorm_attention_flag, self.model_initial)
        self.conv_block3 = Conv_block(256, 512, self.AdapNorm, self.AdapNorm_attention_flag, self.model_initial)
        self.max_pool = nn.MaxPool2d(2)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, 2)

        # model initial
        init_weights(self.fc, init_type=self.model_initial)

    def forward(self, x, params=None):
        if params is not None:
            params_conv_block1 = {}
            params_conv_block2 = {}
            params_conv_block3 = {}
            for k in params:
                if 'conv_block1' in k:
                    params_conv_block1[k] = params[k]
                elif 'conv_block2' in k:
                    params_conv_block2[k] = params[k]
                elif 'conv_block3' in k:
                    params_conv_block3[k] = params[k]
                else:
                    pass
            x = self.conv_block1(x, params_conv_block1)
            x = self.max_pool(x)
            x = self.conv_block2(x, params_conv_block2)
            x = self.max_pool(x)
            x = self.conv_block3(x, params_conv_block3)
        else:
            x = self.conv_block1(x)
            x = self.max_pool(x)
            x = self.conv_block2(x)
            x = self.max_pool(x)
            x = self.conv_block3(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)

        x = self.fc(x)
        return x


class DepthEstmator(nn.Module):
    '''
        Args:
            in_ch (int): the channel numbers of input features
            AdapNorm (bool): 
                'True' allow the Conv_block to combine BN and IN
                'False' allow the Conv_block to use BN
            AdapNorm_attention_flag:
                '1layer' allow the Conv_block to use 1layer FC to generate the balance factor
                '2layer' allow the Conv_block to use 2layer FC to generate the balance factor
            model_initial:
                'kaiming' allow the Conv_block to use 'kaiming' methods to initialize the networks
    '''

    def __init__(self, in_ch=384, AdapNorm=True, AdapNorm_attention_flag='1layer', model_initial='kaiming'):
        super(DepthEstmator, self).__init__()
        self.AdapNorm = AdapNorm
        self.AdapNorm_attention_flag = AdapNorm_attention_flag
        self.model_initial = model_initial

        self.conv_block1 = Conv_block(in_ch, 128, self.AdapNorm, self.AdapNorm_attention_flag, self.model_initial)
        self.conv_block2 = Conv_block(128, 64, self.AdapNorm, self.AdapNorm_attention_flag, self.model_initial)
        self.conv_block3 = Conv_block(64, 1, self.AdapNorm, self.AdapNorm_attention_flag, self.model_initial)

    def forward(self, x, params=None):
        if params is not None:
            params_conv_block1 = {}
            params_conv_block2 = {}
            params_conv_block3 = {}
            for k in params:
                if 'conv_block1' in k:
                    params_conv_block1[k] = params[k]
                elif 'conv_block2' in k:
                    params_conv_block2[k] = params[k]
                elif 'conv_block3' in k:
                    params_conv_block3[k] = params[k]
                else:
                    pass
            x = self.conv_block1(x, params_conv_block1)
            x = self.conv_block2(x, params_conv_block2)
            x = self.conv_block3(x, params_conv_block3)
        else:
            x = self.conv_block1(x)
            x = self.conv_block2(x)
            x = self.conv_block3(x)
        return x


class Framework(nn.Module):
    """Full ANRL model with modular FeatExtractor.

    Backward-compatible: old configs without 'backbone' key default to 'custom_cnn'.
    """

    def __init__(
        self,
        in_ch=6,
        mid_ch=384,
        AdapNorm=True,
        AdapNorm_attention_flag='1layer',
        model_initial='kaiming',
        backbone='custom_cnn',
    ):
        super(Framework, self).__init__()
        self.in_ch = in_ch
        self.mid_ch = mid_ch
        self.AdapNorm = AdapNorm
        self.AdapNorm_attention_flag = AdapNorm_attention_flag
        self.model_initial = model_initial
        self.backbone_name = backbone

        self.FeatExtractor = FeatExtractor(
            backbone_name=backbone,
            in_ch=in_ch,
            AdapNorm=AdapNorm,
            AdapNorm_attention_flag=AdapNorm_attention_flag,
            model_initial=model_initial,
        )
        self.Classifier = FeatEmbedder(in_ch=self.mid_ch, AdapNorm=False)
        self.DepthEstmator = DepthEstmator(in_ch=self.mid_ch, AdapNorm=False)

    def forward(self, x, param=None):
        x, _ = self.FeatExtractor(x, param)
        y = self.Classifier(x)
        depth = self.DepthEstmator(x)
        return y, depth, x
