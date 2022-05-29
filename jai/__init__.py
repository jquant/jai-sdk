"""
--- ___init___.py ---
"""

__version__ = "v0.19.1"

from jai.core.jai import Jai
from jai.core.authentication import get_auth_key, get_authentication, set_authentication

from jai.task.trainer import Trainer
from jai.task.query import Query
from jai.task.linear import LinearModel

__all__ = [
    "__version__", "Jai", "Trainer", "Query", "LinearModel", "get_auth_key",
    "get_authentication", "set_authentication"
]
