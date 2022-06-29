import functools

from .exceptions import DeprecatedError, ParamError, ValidationError


def deprecated(func):
    @functools.wraps(func)
    def inner(*args, **kwargs):
        error_str = "Function {} has been deprecated".format(func.__name__)
        raise DeprecatedError(error_str)

    return inner


def raise_status_error(code):
    """
    Decorator to process responses with unexpected response codes.

    Args
    ----
    code: int
        Expected Code.

    """

    def decorator(function):
        @functools.wraps(function)
        def new_function(*args, **kwargs):
            response = function(*args, **kwargs)
            if response.status_code == code:
                return response.json()
            # find a way to process this
            # what errors to raise, etc.
            message = f"Something went wrong.\n\nSTATUS: {response.status_code}\n"
            try:
                res_json = response.json()
                if isinstance(res_json, dict):
                    detail = res_json.get(
                        "message", res_json.get("detail", response.text)
                    )
                else:
                    detail = response.text
            except:
                detail = response.text

            detail = str(detail)

            if "Error: " in detail:
                error, msg = detail.split(": ", 1)
                try:
                    raise eval(error)(message + msg)
                except:
                    if error == "DeprecatedError":
                        raise DeprecatedError(message + msg)
                    elif error == "ValidationError":
                        raise ValidationError(message + msg)
                    elif error == "ParamError":
                        raise ParamError(message + msg)
                    raise BaseException(message + response.text)
            else:
                raise ValueError(message + detail)

        return new_function

    return decorator
