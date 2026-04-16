'''
    Module:
        FeatEmbedder: Classify the input features
            Shape:
                - Input: (B, 384, 32, 32)
                - Output: (B, 2)
        FeatExtractor: Extract the features from the input images
            Shape:
                - Input: (B, 6, 256, 256)
                - Ouput: (B, 384, 32, 32)
        DepthEstmator:  generate the depth map for the input features
            Shape:
                - Input: (B, 382, 32, 32)
                - Ouput: (B, 1, 32, 32)
        Framork: Contain the above three modules
            Shape:
                - Input: (B, 6, 256, 256)
                - Output: 
                    (B, 2)
                    (B, 1, 32, 32)
                    (B, 384, 32, 32)
'''
from .framework import Framework, FeatEmbedder, FeatExtractor, DepthEstmator
from .backbones import (
    BackboneBase,
    CustomCNNBackbone,
    ResNet18Backbone,
    ResNet50Backbone,
    VGG16Backbone,
    build_backbone,
    register_backbone,
)

__all__ = [
    # Framework
    'Framework',
    'FeatExtractor',
    'FeatEmbedder',
    'DepthEstmator',
    # Backbones
    'BackboneBase',
    'CustomCNNBackbone',
    'ResNet18Backbone',
    'ResNet50Backbone',
    'VGG16Backbone',
    'build_backbone',
    'register_backbone',
]
