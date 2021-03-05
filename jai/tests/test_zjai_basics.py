from jai import Jai
import pandas as pd
import pytest

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


@pytest.mark.parametrize("length", [10, 15])
@pytest.mark.parametrize("prefix", ["", "pre_"])
@pytest.mark.parametrize("suffix", ["", "_fix"])
def test_generate_name(length, prefix, suffix):
    j = Jai(url=URL, auth_key=AUTH_KEY)
    name = j.generate_name(length, prefix, suffix)
    assert len(name) == length, "generated name wrong."

    if prefix != "":
        assert name.startswith(prefix), "prefix not in generated name."

    if suffix != "":
        assert name.endswith(suffix), "suffix not in generated name."


def test_generate_error():
    j = Jai(url=URL, auth_key=AUTH_KEY)
    with pytest.raises(ValueError):
        j.generate_name(8, "prefix", "suffix")