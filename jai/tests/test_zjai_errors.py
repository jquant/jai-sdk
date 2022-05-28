import json

import numpy as np
import pandas as pd
import pytest
from decouple import config

from jai import Jai
import os
from copy import deepcopy

from jai import Jai
from jai.core.validations import check_dtype_and_clean, check_name_lengths


@pytest.fixture(scope='session')
def bad_url_environ():
    # Remove JAI_URL from environment variables
    old_environ = deepcopy(os.environ)
    os.environ["JAI_URL"] = 'http://google.com'

    yield
    # restore initial values
    os.environ = old_environ


def test_names_exception(bad_url_environ):
    j = Jai()
    with pytest.raises(ValueError):
        j.names


def test_info_exception(bad_url_environ):
    j = Jai()
    with pytest.raises(ValueError):
        j.info


def test_generate_error():
    j = Jai()
    with pytest.raises(ValueError):
        j.generate_name(8, "prefix", "suffix")


def test_similar_exception_id(bad_url_environ):
    j = Jai()
    with pytest.raises(ValueError):
        j.similar(name="name", data=[0])


def test_similar_exception_data(bad_url_environ):
    j = Jai()
    with pytest.raises(ValueError):
        j.similar(name="name", data=["a"])


def test_predict_exception(bad_url_environ):
    j = Jai()
    with pytest.raises(ValueError):
        j.predict(name="name", data=["a"])


def test_ids_exception(bad_url_environ):
    j = Jai()
    with pytest.raises(ValueError):
        j.ids(name="name")


def test_temp_ids_exception(bad_url_environ):
    j = Jai()
    with pytest.raises(ValueError):
        j._temp_ids(name="name")


def test_is_valid_exception(bad_url_environ):
    j = Jai()
    with pytest.raises(ValueError):
        j.is_valid(name="name")


def test_fields_exception(bad_url_environ):
    j = Jai()
    with pytest.raises(ValueError):
        j.fields(name="name")


def test_delete_raw_data_exception(bad_url_environ):
    j = Jai()
    with pytest.raises(ValueError):
        j.delete_raw_data(name="name")


def test_delete_database_exception(bad_url_environ):
    j = Jai()
    with pytest.raises(ValueError):
        j.delete_database(name="name")


def test_status_exception(bad_url_environ):
    j = Jai()
    with pytest.raises(ValueError):
        j.status(max_tries=2, patience=4)


def test_similar_id_exceptions_invalid(bad_url_environ):
    j = Jai()
    with pytest.raises(TypeError):
        j._similar_id("test", id_item=dict())


def test_similar_id_exceptions(bad_url_environ):
    # we need to use a valid URL for this one
    j = Jai()
    with pytest.raises(ValueError):
        j._similar_id("test", id_item=[])


def test_similar_json_exception(bad_url_environ):
    j = Jai()
    with pytest.raises(ValueError):
        j._similar_json("test", data_json=dict())


def test_invalid_name_exception(bad_url_environ):
    # we need to use a valid URL for this one
    j = Jai()
    with pytest.raises(ValueError):
        j.get_dtype("test")


def test_check_dtype_and_clean_exception(bad_url_environ):
    j = Jai()
    with pytest.raises(TypeError):
        check_dtype_and_clean(data=dict(), db_type="SelfSupervised")

    db = np.array([])
    with pytest.raises(ValueError) as e:
        check_dtype_and_clean(data=db, db_type="SelfSupervised")
    assert e.value.args[0] == f"Inserted data is empty."

    db = np.array([[[1]]])
    with pytest.raises(ValueError) as e:
        check_dtype_and_clean(data=db, db_type="SelfSupervised")
    assert e.value.args[
        0] == f"Inserted 'np.ndarray' data has many dimensions ({db.ndim}). JAI only accepts up to 2-d inputs."


def test_predict_exception(bad_url_environ):
    j = Jai()
    with pytest.raises(ValueError):
        j._predict(name="test", data_json=dict())


def test_append_exception(bad_url_environ):
    j = Jai()
    with pytest.raises(ValueError):
        j._append(name="test")


def test_insert_json_exception(bad_url_environ):
    j = Jai()
    with pytest.raises(TypeError):
        j._insert_json(name="test", df_json=dict())


def test_insert_vector_json_exception():
    j = Jai()

    db = np.array([[1], [2]])
    with pytest.raises(ValueError) as e:
        j.insert_vectors(name="test", data=db, overwrite=True)
    assert e.value.args[
        0] == f"Data must be a DataFrame with at least 2 columns other than `id`. Current column(s):\n[0]"

    db = pd.DataFrame({'a': [1, 'a'], 'b': [1, 'c'], 'c': [1, np.nan]})
    with pytest.raises(ValueError) as e:
        j.insert_vectors(name="test", data=db, overwrite=True)
    assert e.value.args[
        0] == f"Columns ['a', 'b'] contains values types different from numeric."


def test_setup_database_exception(bad_url_environ):
    j = Jai()
    with pytest.raises(ValueError):
        j._setup(name="test", body={"db_type": "SelfSupervised"})


def test_embedding_exception(bad_url_environ):
    j = Jai()
    with pytest.raises(ValueError):
        j.embedding(name="test", data=dict())


def test_check_name_lengths_exception():
    # we need to use a valid URL for this one
    j = Jai()
    with pytest.raises(ValueError):
        check_name_lengths(name="test", cols=[j.generate_name(length=35)])


@pytest.mark.parametrize("name, batch_size, db_type",
                         [("test", 1024, "SelfSupervised")])
def test_check_ids_consistency_exception(name, batch_size, db_type):
    # we need to use a valid URL for this one
    j = Jai()

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


@pytest.mark.parametrize("name", ["invalid_test"])
def test_delete_tree(name):
    # we need to use a valid URL for this one
    j = Jai()
    with pytest.raises(IndexError):
        j._delete_tree(name)


def test_download_vectors_exception(bad_url_environ):
    j = Jai()
    with pytest.raises(ValueError):
        j.download_vectors(name="test")


@pytest.mark.parametrize('name', ['test_resolution'])
def test_filters(name):
    j = Jai()
    with pytest.raises(ValueError):
        j.filters(name)


@pytest.mark.parametrize("name, batch_size, db_type, max_insert_workers",
                         [("test", 1024, "SelfSupervised", "1")])
def test_max_insert_workers(name, batch_size, db_type, max_insert_workers):
    j = Jai()
    with pytest.raises(TypeError):
        j._insert_data(data={},
                       name=name,
                       batch_size=batch_size,
                       db_type=db_type,
                       max_insert_workers=max_insert_workers)
