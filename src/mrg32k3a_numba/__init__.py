"""
MRG32k3a-numba: An accelerated version of the MRG32k3a generator.

This package provides the MRG32k3a_numba class, which leverages Numba for
high-performance batch generation of random variates from various distributions.
"""
from .mrg32k3a_numba import MRG32k3a_numba

__version__ = "0.1.0"
__author__ = "Song Huang" "Guangxin Jiang" "Ying Zhong"