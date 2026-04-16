"""
CustomCNNBackbone — refactored original FeatExtractor (AdapNorm + MAML-aware).
MAML: AttentionNet parameters are meta-learned in inner loop.
"""
import sys
import os as _os
sys.path.append(_os.path.join(_os.path.abspath(_os.path.dirname(__file__)), '..', '..', '..'))

import torch
from torch import nn
import torch.nn.functional as F

from common.utils.model_init import init_weights
from .base import BackboneBase
from ..base_block import Conv_block, Basic_block


class CustomCNNBackbone(BackboneBase):
    """Custom lightweight CNN — identical to original FeatExtractor.

    Architecture:
        inc:   Conv_block(in_ch, 64)
        down1: Basic_block(64,  128) → maxpool → 128ch @ 128x128
        down2: Basic_block(128, 128) → maxpool → 128ch @ 64x64
        down3: Basic_block(128, 128) → maxpool → 128ch @ 32x32

    Output: concat[re_pool(down1), re_pool(down2), down3]
            → 128+128+128 = 384 channels @ 32x32

    MAML: get_meta_params() returns AttentionNet params inside Conv_block.
          forward() routes params dict to each Conv_block for fast-weight updates.

    Args:
        in_ch                  (int): input channels (default 6)
        AdapNorm               (bool): enable BN+IN blend via attention
        AdapNorm_attention_flag (str): '1layer' or '2layer' for AttentionNet topology
        model_initial           (str): weight init method
    """

    def __init__(
        self,
        in_ch=6,
        AdapNorm=True,
        AdapNorm_attention_flag='1layer',
        model_initial='kaiming',
    ):
        super().__init__(in_ch=in_ch, mid_ch=384)
        self.AdapNorm = AdapNorm
        self.AdapNorm_attention_flag = AdapNorm_attention_flag
        self.model_initial = model_initial

        self.inc   = Conv_block(in_ch,  64,  self.AdapNorm, self.AdapNorm_attention_flag, self.model_initial)
        self.down1 = Basic_block(64,   128, self.AdapNorm, self.AdapNorm_attention_flag, self.model_initial)
        self.down2 = Basic_block(128,  128, self.AdapNorm, self.AdapNorm_attention_flag, self.model_initial)
        self.down3 = Basic_block(128,  128, self.AdapNorm, self.AdapNorm_attention_flag, self.model_initial)

    # ------------------------------------------------------------------ #
    def _slice_params(self, params, prefix):
        if params is None:
            return None
        return {k: v for k, v in params.items() if prefix in k}

    # ------------------------------------------------------------------ #
    def forward(self, x, params=None):
        dx1 = self.inc(x,   self._slice_params(params, 'inc'))
        dx2 = self.down1(dx1, self._slice_params(params, 'down1'))
        dx3 = self.down2(dx2, self._slice_params(params, 'down2'))
        dx4 = self.down3(dx3, self._slice_params(params, 'down3'))

        re_dx2   = F.adaptive_avg_pool2d(dx2, 32)
        re_dx3   = F.adaptive_avg_pool2d(dx3, 32)
        catfeat  = torch.cat([re_dx2, re_dx3, dx4], 1)   # (B, 384, 32, 32)

        return catfeat, dx4                               # (B, 384, 32, 32), (B, 128, 32, 32)

    # ------------------------------------------------------------------ #
    def get_meta_params(self):
        """MAML: return only AttentionNet parameters (used in inner loop)."""
        meta = []
        for name, module in self.named_modules():
            if 'AttentionNet' in name:
                for _, param in module.named_parameters(recurse=False):
                    meta.append(param)
        return meta

    # ------------------------------------------------------------------ #
    @property
    def backbone_type(self):
        return 'custom_cnn'
