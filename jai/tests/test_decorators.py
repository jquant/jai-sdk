import pytest

from jai.core.decorators import deprecated
from jai.core.exceptions import DeprecatedError


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
