"""Abstract Backbone base class — defines the interface for all backbones."""
from abc import ABC, abstractmethod
from torch import nn


class BackboneBase(ABC, nn.Module):
    """Abstract base for all FeatExtractor backbones.

    Subclasses MUST implement:
        - forward(x, params=None) -> (catfeat_384_32x32, last_feat_128_32x32)
        - get_meta_params()       -> List[nn.Parameter]
        - backbone_type           -> str property

    Shared invariants:
        Input:  (B, 6, 256, 256) — RGB + HSV
        catfeat: (B, 384, 32, 32) — for FeatEmbedder & DepthEstmator
        last_feat: (B, 128, 32, 32) — auxiliary (backward compat)
    """

    def __init__(self, in_ch=6, mid_ch=384):
        super().__init__()
        self.in_ch = in_ch
        self.mid_ch = mid_ch

    @abstractmethod
    def forward(self, x, params=None):
        """Extract features.

        Args:
            x:      (B, 6, 256, 256) input
            params (dict|None): fast-weight overrides for MAML inner loop;
                                 None = use stored weights

        Returns:
            tuple:
                catfeat   (B, 384, 32, 32) — for heads
                last_feat (B, 128, 32, 32) — auxiliary
        """
        ...

    @abstractmethod
    def get_meta_params(self):
        """Return parameters updated in MAML inner loop.

        CustomCNNBackbone: AttentionNet params only.
        Pretrained backbones: [] (full fine-tune, no MAML).
        """
        ...

    @property
    @abstractmethod
    def backbone_type(self) -> str:
        """Registry key, e.g. 'resnet18', 'custom_cnn'."""
        ...
