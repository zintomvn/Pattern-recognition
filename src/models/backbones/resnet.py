"""
ResNet18Backbone / ResNet50Backbone — torchvision pretrained, adapted for 6ch input.

Key design:
  1. First conv: average pretrained RGB weights across 3 channels -> broadcast
     to all 6 channels (RGB+HSV).
  2. No AdapNorm — standard BatchNorm2d (full fine-tune).
  3. params accepted but ignored — no MAML for pretrained backbones.
  4. Output: adaptive_avg_pool2d(layer4, 32) -> 1x1 conv -> 384 channels
     last_feat: 1x1 conv -> 128 channels (backward compat).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models.resnet as _resnet
from torchvision.models import resnet18, resnet50

from .base import BackboneBase


def _adapt_first_conv(model, in_ch=6):
    """Replace first conv to accept in_ch input channels.

    Strategy: average pretrained RGB weights [64, 3, 7, 7] -> [64, 1, 7, 7],
    then expand to [64, in_ch, 7, 7] so all channels get the same pattern.
    """
    old = model.conv1   # Conv2d(3, 64, 7, stride=2, padding=3)
    old_w = old.weight.data

    avg_w = old_w.mean(dim=1, keepdim=True)                # (64, 1, 7, 7)
    new_w = avg_w.expand(-1, in_ch, -1, -1).clone()        # (64, 6, 7, 7)

    new_conv = nn.Conv2d(in_ch, 64, kernel_size=7, stride=2, padding=3,
                         bias=(old.bias is not None))
    new_conv.weight.data = new_w
    if old.bias is not None:
        new_conv.bias.data = old_bias.data.clone()
    return new_conv


def _get_weights(cls_name, pretrained):
    """Load torchvision pretrained weights for a given resnet function name."""
    weights_cls = getattr(_resnet, f'{cls_name}_Weights', None)
    if pretrained and weights_cls:
        return weights_cls.DEFAULT
    return None


class _ResNetBackbone(BackboneBase):
    """Shared base for ResNet18 and ResNet50."""

    # subclass overrides
    RESNET_FN = None   # e.g. resnet18
    FEAT_CH    = None   # e.g. 512 for ResNet18

    def __init__(self, in_ch=6, mid_ch=384, pretrained=True, **kwargs):
        super().__init__(in_ch=in_ch, mid_ch=mid_ch)
        self.pretrained = pretrained
        self._build_body()

    @property
    def backbone_type(self):
        raise NotImplementedError

    def _build_body(self):
        fn_name = self.RESNET_FN.__name__              # 'resnet18'
        weights = _get_weights(fn_name, self.pretrained)
        body = self.RESNET_FN(weights=weights)

        body.conv1 = _adapt_first_conv(body, self.in_ch)
        del body.avgpool
        del body.fc

        self.body = body
        self.conv_final     = nn.Conv2d(self.FEAT_CH, self.mid_ch, 1, 1, 0)
        self.last_feat_conv = nn.Conv2d(self.FEAT_CH, 128, 1, 1, 0)

    def forward(self, x, params=None):
        bx = self.body.conv1(x)
        bx = self.body.bn1(bx)
        bx = self.body.relu(bx)
        bx = self.body.maxpool(bx)
        bx = self.body.layer1(bx)
        bx = self.body.layer2(bx)
        bx = self.body.layer3(bx)
        bx = self.body.layer4(bx)                 # (B, FEAT_CH, H, W)

        bx = F.adaptive_avg_pool2d(bx, 32)        # (B, FEAT_CH, 32, 32)
        catfeat   = self.conv_final(bx)           # (B, 384, 32, 32)
        last_feat = self.last_feat_conv(bx)       # (B, 128, 32, 32)
        return catfeat, last_feat

    def get_meta_params(self):
        return []   # full fine-tune, no MAML


class ResNet18Backbone(_ResNetBackbone):
    RESNET_FN = resnet18
    FEAT_CH   = 512   # layer4 out channels (ResNet18 layer4: 256 -> 512 via downsample)

    @property
    def backbone_type(self):
        return 'resnet18'


class ResNet50Backbone(_ResNetBackbone):
    RESNET_FN = resnet50
    FEAT_CH   = 2048  # layer4 out channels

    @property
    def backbone_type(self):
        return 'resnet50'
