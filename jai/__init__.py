"""
--- ___init___.py ---
"""

__version__ = "v0.19.1"

from jai.core.jai import Jai
from jai.utilities.image import read_image
from jai.utilities.processing import (filter_resolution, filter_similar,
                                      find_threshold, predict2df, treat_unix)
from jai.utilities.splits import split, split_recommendation

__all__ = [
    "__version__", "Jai", "read_image", "split", "split_recommendation",
    "find_threshold", "filter_similar", "predict2df", "filter_resolution",
    "treat_unix"
]
