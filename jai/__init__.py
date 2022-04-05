"""
--- ___init___.py ---
"""

__version__ = "v0.19.0"

from jai.core.jai import Jai
from jai.utilities._image import read_image_folder
from jai.utilities._splits import split, split_recommendation
from jai.utilities._processing import find_threshold, filter_similar, predict2df, filter_resolution, treat_unix

__all__ = [
    "__version__", "Jai", "read_image_folder", "split", "split_recommendation",
    "find_threshold", "filter_similar", "predict2df", "filter_resolution",
    "treat_unix"
]
