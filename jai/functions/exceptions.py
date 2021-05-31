__all__ = ["DeprecatedError", "ValidationError"]


class DeprecatedError(AttributeError):
    """
    Deprecated
    """


class ValidationError(ValueError):
    """
    Error validating the given inputs
    """