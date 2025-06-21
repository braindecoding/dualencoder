#!/usr/bin/env python3
"""
Miyawaki2 - Advanced Modular Architecture
"""

# Core components
from .fmriencoder import fMRI_Encoder
from .stimencoder import Shape_Encoder
from .clipcorrtrain import CLIP_Correlation
from .diffusion import Diffusion_Decoder
from .gan import GAN_Decoder

# Training components
from .latentlearn import *
from .corrlearning import *
from .genbetrain import *

# Evaluation components
from .metriks import evaluate_decoding_performance
from .analisys import *

# Utilities
from .miyawakidataset import load_miyawaki_dataset_corrected, create_dataloaders_corrected

__version__ = "2.0.0"
__author__ = "Dual Encoder Team"
__description__ = "Advanced modular architecture for fMRI-visual cross-modal learning"

__all__ = [
    'fMRI_Encoder',
    'Shape_Encoder', 
    'CLIP_Correlation',
    'Diffusion_Decoder',
    'GAN_Decoder',
    'evaluate_decoding_performance',
    'load_miyawaki_dataset_corrected',
    'create_dataloaders_corrected'
]
