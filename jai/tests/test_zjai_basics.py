from jai import Jai
from pathlib import Path
from pyspark.sql import SparkSession
from pyspark.sql import dataframe as psdf
from pandas._testing import assert_frame_equal, assert_series_equal

import pandas as pd
import pytest
import numpy as np
import json
import os

URL = 'http://localhost:8001'
AUTH_KEY = ""
HEADER_TEST = json.loads(os.environ['HEADER_TEST'])


@pytest.fixture(scope="session")
def setup_npy_file():
    NPY_FILE = Path("jai/test_data/sdk_test_titanic_ssupervised.npy")
    img_file = np.load(NPY_FILE)
    return img_file


def test_url():
    j = Jai(AUTH_KEY)
    j.header = HEADER_TEST
    assert j.url == "https://mycelia.azure-api.net"
    # assert j.


def test_custom_url():
    j = Jai(url=URL + "/", auth_key=AUTH_KEY)
    j.header = HEADER_TEST
    assert j.url == URL


def test_names():
    j = Jai(url=URL, auth_key=AUTH_KEY)
    j.header = HEADER_TEST
    assert j.names == ['test_match', 'test_resolution', 'titanic_ssupervised']


def test_info():
    j = Jai(url=URL, auth_key=AUTH_KEY)
    j.header = HEADER_TEST
    assert isinstance(j.info, pd.DataFrame)


def test_status():
    j = Jai(url=URL, auth_key=AUTH_KEY)
    j.header = HEADER_TEST
    assert isinstance(j.status, dict)


@pytest.mark.parametrize("length", [10, 15])
@pytest.mark.parametrize("prefix", ["", "pre_"])
@pytest.mark.parametrize("suffix", ["", "_fix"])
def test_generate_name(length, prefix, suffix):
    j = Jai(url=URL, auth_key=AUTH_KEY)
    j.header = HEADER_TEST
    name = j.generate_name(length, prefix, suffix)
    assert len(name) == length, "generated name wrong."

    if prefix != "":
        assert name.startswith(prefix), "prefix not in generated name."

    if suffix != "":
        assert name.endswith(suffix), "suffix not in generated name."


def test_generate_error():
    j = Jai(url=URL, auth_key=AUTH_KEY)
    j.header = HEADER_TEST
    with pytest.raises(ValueError):
        j.generate_name(8, "prefix", "suffix")


def test_check_dtype_and_clean():
    j = Jai(url=URL, auth_key=AUTH_KEY)
    j.header = HEADER_TEST

    # mock data
    r = 1100
    data = pd.DataFrame({
        "category": [str(i) for i in range(r)],
        "number": [i for i in range(r)]
    })

    # Send mock data to Pyspark
    spark = SparkSession.builder.getOrCreate()
    psdata = spark.createDataFrame(data)
    assert_frame_equal(j._check_dtype_and_clean(psdata, "Supervised"), data)

    # Send np.ndarray
    nparray = np.array([10, 20, 30, 40, 50])
    assert_series_equal(j._check_dtype_and_clean(nparray, "Supervised"),
                        pd.Series(nparray))

    # make a few lines on 'category' column NaN
    data.loc[1050:, "category"] = np.nan
    assert_frame_equal(j._check_dtype_and_clean(data, "Supervised"), data)

    # Try text data
    text = pd.Series(['a', 'b', 'c', np.nan, 'd', 'e', np.nan])
    assert_series_equal(j._check_dtype_and_clean(text, "Text"), text.dropna())


@pytest.mark.parametrize("db_type, col, ans", [({
    "col1": "FastText"
}, "col1", "FastText"), ({
    "col1": "FastText"
}, "col2", "TextEdit")])
def test_resolve_db_type(db_type, col, ans):
    j = Jai(url=URL, auth_key=AUTH_KEY)
    j.header = HEADER_TEST
    assert j._resolve_db_type(db_type, col) == ans


@pytest.mark.parametrize('name', ['titanic_ssupervised'])
def test_download_vectors(setup_npy_file, name):
    npy_file = setup_npy_file
    j = Jai(url=URL, auth_key=AUTH_KEY)
    j.header = HEADER_TEST
    np.testing.assert_array_equal(npy_file, j.download_vectors(name=name))


def test_user():
    j = Jai(url=URL, auth_key=AUTH_KEY)
    j.header = HEADER_TEST
    assert j.user() == {
        'email': 'test_sdk@email.com',
        'firstName': 'User SDK',
        'lastName': 'Test',
        'memberRole': 'dev',
        'namespace': 'sdk',
        'userId': 'random_string'
    }


def test_environments():
    j = Jai(url=URL, auth_key=AUTH_KEY)
    j.header = HEADER_TEST
    assert j.environments() == ['sdk_test', 'sdk_prod']


@pytest.mark.parametrize('name', ['test_resolution'])
def test_describe(name):
    j = Jai(url=URL, auth_key=AUTH_KEY)
    j.header = HEADER_TEST
    description = j.describe(name)
    description.pop("version")
    assert description == {
        'dtype': 'TextEdit',
        'features': [{
            'dtype': 'text',
            'name': '0'
        }],
        'has_filter': False,
        'model_hyperparams': {
            'batch_size': 128,
            'channel': 8,
            'embed_dim': 128,
            'epochs': 20,
            'k': 100,
            'maxl': 0,
            'mtc': False,
            'nb': 1385451,
            'nr': 1000,
            'nt': 1000,
            'random_append_train': False,
            'random_train': False,
            'shuffle_seed': 808,
            'test_batch_size': 1024
        },
        'name': 'test_resolution',
        'state': 'active'
    }


def test_rename():
    j = Jai(url=URL, auth_key=AUTH_KEY)
    j.header = HEADER_TEST
    assert j.names == ['test_match', 'test_resolution', 'titanic_ssupervised']
    j.rename(original_name='test_match', new_name='test_match_new')
    assert j.names == [
        'test_match_new', 'test_resolution', 'titanic_ssupervised'
    ]
    j.rename(original_name='test_match_new', new_name='test_match')
    assert j.names == ['test_match', 'test_resolution', 'titanic_ssupervised']


@pytest.mark.parametrize('db_name', ['test_match'])
def test_transfer(db_name):
    j = Jai(url=URL, auth_key=AUTH_KEY)
    j.header = HEADER_TEST

    j_prod = Jai(url=URL, auth_key=AUTH_KEY)
    j_prod.header = {**HEADER_TEST, "environment": 'prod'}
    if db_name in j_prod.names:
        j_prod.delete_database(db_name)

    j.transfer(original_name=db_name,
               to_environment='prod',
               from_environment='default')

    assert j_prod.names == [db_name, 'titanic_ssupervised']

    j_prod.delete_database(db_name)
    assert j_prod.names == ['titanic_ssupervised']
