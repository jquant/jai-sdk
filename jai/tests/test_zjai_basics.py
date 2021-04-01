from jai import Jai
from pandas._testing import assert_frame_equal
import pandas as pd
import pytest
import numpy as np

URL = 'http://localhost:8001'
AUTH_KEY = "sdk_test"


def test_url():
    j = Jai(AUTH_KEY)
    assert j.url == "https://mycelia.azure-api.net"


def test_custom_url():
    j = Jai(url=URL + "/", auth_key=AUTH_KEY)
    assert j.url == URL


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


def test_check_dtype_and_clean():
    j = Jai(url=URL, auth_key=AUTH_KEY)

    # mock data
    r = 1100
    data = pd.DataFrame({
        "category": [str(i) for i in range(r)],
        "number": [i for i in range(r)]
    })

    # make a few lines on 'category' column NaN
    data.loc[1050:, "category"] = np.nan
    assert_frame_equal(j._check_dtype_and_clean(data, "Supervised"),
                       data.dropna(subset=["category"]))


@pytest.mark.parametrize("name, batch_size, db_type",
                         [("test", 1024, "SelfSupervised")])
def test_check_ids_consistency(name, batch_size, db_type):
    j = Jai(url=URL, auth_key=AUTH_KEY)

    # mock data
    r = 1100
    data = pd.DataFrame({
        "category": [str(i) for i in range(r)],
        "number": [i for i in range(r)]
    })

    # insert it
    j._insert_data(data=data,
                   name=name,
                   batch_size=batch_size,
                   db_type=db_type)

    # intentionally break it
    with pytest.raises(Exception):
        j._check_ids_consistency(name=name, data=data.iloc[:r - 5])
