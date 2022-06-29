import pytest

from jai.core.decorators import deprecated, raise_status_error
from jai.core.exceptions import DeprecatedError, ParamError, ValidationError


class MockResponse:
    def __init__(self, err):
        self.err = err

    @staticmethod
    def status_code():
        return 500

    def json(self):
        return {"message": f"{self.err}: Something went wrong."}

    @property
    def text(self):
        return f"{self.err}: Something went wrong."


def test_deprecated():
    @deprecated
    def printer(string):
        return string

    with pytest.raises(DeprecatedError):
        printer("test")


def test_status_error_deprecated():
    @raise_status_error(200)
    def response(err):
        return MockResponse(err)

    with pytest.raises(DeprecatedError):
        response("DeprecatedError")

    with pytest.raises(ValidationError):
        response("ValidationError")

    with pytest.raises(ParamError):
        response("ParamError")
