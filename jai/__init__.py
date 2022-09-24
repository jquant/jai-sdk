"""
--- ___init___.py ---
"""

__version__ = "v0.22.1"

from jai.core.authentication import get_auth_key, get_authentication, set_authentication
from jai.core.jai import Jai
from jai.task.explorer import Explorer
from jai.task.linear import LinearModel
from jai.task.query import Query
from jai.task.trainer import Trainer
from jai.task.vectors import Vectors

__all__ = [
    "__version__",
    "Jai",
    "Explorer",
    "Trainer",
    "Query",
    "LinearModel",
    "Vectors",
    "get_auth_key",
    "get_authentication",
    "set_authentication",
]
