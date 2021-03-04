from jai import Jai
import pandas as pd

URL = 'http://localhost:8001'
AUTH_KEY = "sdk_test"


def test_names():
    j = Jai(url=URL, auth_key=AUTH_KEY)
    assert isinstance(j.names, list)


def test_info():
    j = Jai(url=URL, auth_key=AUTH_KEY)
    assert isinstance(j.info, pd.DataFrame)


def test_status():
    j = Jai(url=URL, auth_key=AUTH_KEY)
    assert isinstance(j.status, dict)
