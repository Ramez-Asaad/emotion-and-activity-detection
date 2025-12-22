"""
Models Package
=============

Contains all model architectures for the project.
"""

from .base_cnn import CustomCNN, count_parameters

__all__ = ['CustomCNN', 'count_parameters']
