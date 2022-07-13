import functools

from .exceptions import DeprecatedError


def deprecated(func):
    @functools.wraps(func)
    def inner(*args, **kwargs):
        error_str = "Function {} has been deprecated".format(func.__name__)
        raise DeprecatedError(error_str)

    return inner
