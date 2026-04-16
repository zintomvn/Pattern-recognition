"""
VGG16Backbone — torchvision pretrained VGG16, adapted for 6ch input.

Key design:
  1. First conv: average pretrained RGB weights → broadcast to all 6 channels.
  2. Remove classifier; keep features body (output 512ch).
  3. No AdapNorm — standard BatchNorm2d (full fine-tune).
  4. params ignored — no MAML.
  5. Output: adaptive_avg_pool2d(features, 32) → 1x1 conv → 384 channels
     last_feat: 1x1 conv → 128 channels (backward compat).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models.vgg as _vgg

from .base import BackboneBase


def _adapt_first_conv(model, in_ch=6):
    """Replace first conv to accept in_ch input channels.

    Strategy: average pretrained RGB weights [64, 3, 3, 3] → [64, 1, 3, 3],
    then expand to [64, in_ch, 3, 3].
    """
    old = model.features[0]   # Conv2d(3, 64, 3, padding=1)
    old_w = old.weight.data

    avg_w = old_w.mean(dim=1, keepdim=True)               # (64, 1, 3, 3)
    new_w = avg_w.expand(-1, in_ch, -1, -1).clone()        # (64, 6, 3, 3)

    new_conv = nn.Conv2d(in_ch, 64, kernel_size=3, stride=1, padding=1,
                         bias=(old.bias is not None))
    new_conv.weight.data = new_w
    if old.bias is not None:
        new_conv.bias.data = old.bias.data.clone()
    return new_conv


class VGG16Backbone(BackboneBase):
    """VGG16-based backbone.

    VGG features output = 512 channels @ 32x32.
    Pooled to 32x32 → 1x1 conv → 384 (catfeat) + 128 (last_feat).

    Args:
        in_ch      (int): input channels (default 6)
        mid_ch     (int): output channel count (default 384)
        pretrained (bool): load torchvision pretrained weights
    """

    def __init__(self, in_ch=6, mid_ch=384, pretrained=True, **kwargs):
        super().__init__(in_ch=in_ch, mid_ch=mid_ch)
        self.pretrained = pretrained
        self._build_body()

    def _build_body(self):
        weights = (_vgg.VGG16_Weights.DEFAULT if self.pretrained else None)
        vgg = _vgg.vgg16(weights=weights)

        vgg.features[0] = _adapt_first_conv(vgg, self.in_ch)
        self.body = vgg.features   # Sequential conv layers

        self.conv_final     = nn.Conv2d(512, self.mid_ch, 1, 1, 0)
        self.last_feat_conv = nn.Conv2d(512, 128, 1, 1, 0)

    def forward(self, x, params=None):
        bx = self.body(x)                           # (B, 512, 32, 32)
        bx = F.adaptive_avg_pool2d(bx, 32)           # explicit safety pool
        catfeat   = self.conv_final(bx)              # (B, 384, 32, 32)
        last_feat = self.last_feat_conv(bx)          # (B, 128, 32, 32)
        return catfeat, last_feat

    def get_meta_params(self):
        return []   # full fine-tune, no MAML

    @property
    def backbone_type(self):
        return 'vgg16'
