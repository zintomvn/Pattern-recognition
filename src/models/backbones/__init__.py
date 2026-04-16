"""
Backbone registry — add new backbones by adding a class here.

Usage:
    from models.backbones import build_backbone
    backbone = build_backbone('resnet18', in_ch=6, mid_ch=384)

To add a new backbone:
    1. Create src/models/backbones/<name>.py with a class inheriting BackboneBase
    2. Add one line to _BACKBONE_REGISTRY below
    3. Or use @register_backbone('name') decorator
"""
from .base import BackboneBase
from .custom_cnn import CustomCNNBackbone
from .resnet import ResNet18Backbone, ResNet50Backbone
from .vgg import VGG16Backbone

_BACKBONE_REGISTRY = {
    'custom_cnn': CustomCNNBackbone,
    'resnet18':   ResNet18Backbone,
    'resnet50':   ResNet50Backbone,
    'vgg16':      VGG16Backbone,
}


def build_backbone(backbone_name, **kwargs):
    """Build a backbone by name from the registry.

    Args:
        backbone_name (str): key in _BACKBONE_REGISTRY
        **kwargs: passed to the backbone constructor

    Returns:
        BackboneBase instance

    Raises:
        ValueError: if backbone_name is not registered
    """
    if backbone_name not in _BACKBONE_REGISTRY:
        raise ValueError(
            f"Unknown backbone '{backbone_name}'. "
            f"Available: {list(_BACKBONE_REGISTRY.keys())}"
        )
    return _BACKBONE_REGISTRY[backbone_name](**kwargs)


def register_backbone(name):
    """Decorator to register a new backbone class.

    Usage:
        @register_backbone('my_backbone')
        class MyBackbone(BackboneBase):
            ...
    """
    def decorator(cls):
        if not issubclass(cls, BackboneBase):
            raise TypeError(f"{cls.__name__} must inherit from BackboneBase")
        _BACKBONE_REGISTRY[name] = cls
        return cls
    return decorator


__all__ = [
    'BackboneBase',
    'CustomCNNBackbone',
    'ResNet18Backbone',
    'ResNet50Backbone',
    'VGG16Backbone',
    'build_backbone',
    'register_backbone',
]
