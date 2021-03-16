from jai import Jai
import pandas as pd
import pytest

URL = 'http://google.com'
AUTH_KEY = "sdk_test"


def test_names_exception():
    with pytest.raises(ValueError):
        j = Jai(url=URL, auth_key=AUTH_KEY)
        j.names