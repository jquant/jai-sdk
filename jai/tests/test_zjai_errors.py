from jai import Jai
import numpy as np
import pandas as pd
import pytest

URL = 'http://google.com'
AUTH_KEY = "sdk_test"


def test_names_exception():
    with pytest.raises(ValueError):
        j = Jai(url=URL, auth_key=AUTH_KEY)
        j.names


def test_info_exception():
    with pytest.raises(ValueError):
        j = Jai(url=URL, auth_key=AUTH_KEY)
        j.info


def test_similar_exception_id():
    with pytest.raises(ValueError):
        j = Jai(url=URL, auth_key=AUTH_KEY)
        j.similar(name="name", data=[0])


def test_similar_exception_data():
    with pytest.raises(ValueError):
        j = Jai(url=URL, auth_key=AUTH_KEY)
        j.similar(name="name", data=["a"])


def test_predict_exception():
    with pytest.raises(ValueError):
        j = Jai(url=URL, auth_key=AUTH_KEY)
        j.predict(name="name", data=["a"])


def test_ids_exception():
    with pytest.raises(ValueError):
        j = Jai(url=URL, auth_key=AUTH_KEY)
        j.ids(name="name")


def test_is_valid_exception():
    with pytest.raises(ValueError):
        j = Jai(url=URL, auth_key=AUTH_KEY)
        j.is_valid(name="name")


def test_fields_exception():
    with pytest.raises(ValueError):
        j = Jai(url=URL, auth_key=AUTH_KEY)
        j.fields(name="name")


def test_delete_raw_data_exception():
    with pytest.raises(ValueError):
        j = Jai(url=URL, auth_key=AUTH_KEY)
        j.delete_raw_data(name="name")


def test_delete_database_exception():
    with pytest.raises(ValueError):
        j = Jai(url=URL, auth_key=AUTH_KEY)
        j.delete_database(name="name")
