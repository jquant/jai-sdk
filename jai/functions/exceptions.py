__all__ = ["MaxRetriesExceeded", "DeprecatedError"]


class MaxRetriesExceeded(RuntimeError):
    """
    Number of tries exceeded
    """


class DeprecatedError(AttributeError):
    """
    Deprecated
    """
