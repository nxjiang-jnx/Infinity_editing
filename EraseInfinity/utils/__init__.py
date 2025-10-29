# coding: UTF-8
"""
EraseInfinity Utils Package
"""

from .calc_loss import calculate_esd_loss, calculate_esd_loss_simple
from .esd_utils import (
    autoregressive_sample,
    predict_next_scale,
    sample_categorical,
    bits_to_indices,
    indices_to_bits,
    get_default_scale_schedule,
)

__all__ = [
    'calculate_esd_loss',
    'calculate_esd_loss_simple',
    'autoregressive_sample',
    'predict_next_scale',
    'sample_categorical',
    'bits_to_indices',
    'indices_to_bits',
    'get_default_scale_schedule',
]

