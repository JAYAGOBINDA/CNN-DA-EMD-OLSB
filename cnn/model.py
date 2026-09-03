"""
CNN Model Architectures for CNN-DA-EMD-OLSB Steganography Framework.

Models
------
DistortionCNN:
    Multi-scale 3-branch CNN for per-channel (R/G/B) distortion sensitivity estimation.
    Core component of the proposed CNN-DA-EMD-OLSB system.
    See: cnn/distortion_cnn.py for the full implementation.
"""

# Re-export DistortionCNN from its dedicated module for convenient import
try:
    from cnn.distortion_cnn import DistortionCNN
except ImportError:
    DistortionCNN = None
