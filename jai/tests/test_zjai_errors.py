from jai import Jai
import numpy as np
import pandas as pd
import pytest

INVALID_URL = 'http://google.com'
VALID_URL = 'http://localhost:8001'
AUTH_KEY = "sdk_test"


def test_names_exception():
    with pytest.raises(ValueError):
        j = Jai(url=INVALID_URL, auth_key=AUTH_KEY)
        j.names


def test_info_exception():
    with pytest.raises(ValueError):
        j = Jai(url=INVALID_URL, auth_key=AUTH_KEY)
        j.info


def test_similar_exception_id():
    with pytest.raises(ValueError):
        j = Jai(url=INVALID_URL, auth_key=AUTH_KEY)
        j.similar(name="name", data=[0])


def test_similar_exception_data():
    with pytest.raises(ValueError):
        j = Jai(url=INVALID_URL, auth_key=AUTH_KEY)
        j.similar(name="name", data=["a"])


def test_predict_exception():
    with pytest.raises(ValueError):
        j = Jai(url=INVALID_URL, auth_key=AUTH_KEY)
        j.predict(name="name", data=["a"])


def test_ids_exception():
    with pytest.raises(ValueError):
        j = Jai(url=INVALID_URL, auth_key=AUTH_KEY)
        j.ids(name="name")


def test_temp_ids_exception():
    with pytest.raises(ValueError):
        j = Jai(url=INVALID_URL, auth_key=AUTH_KEY)
        j._temp_ids(name="name")


def test_is_valid_exception():
    with pytest.raises(ValueError):
        j = Jai(url=INVALID_URL, auth_key=AUTH_KEY)
        j.is_valid(name="name")


def test_fields_exception():
    with pytest.raises(ValueError):
        j = Jai(url=INVALID_URL, auth_key=AUTH_KEY)
        j.fields(name="name")


def test_delete_raw_data_exception():
    with pytest.raises(ValueError):
        j = Jai(url=INVALID_URL, auth_key=AUTH_KEY)
        j.delete_raw_data(name="name")


def test_delete_database_exception():
    with pytest.raises(ValueError):
        j = Jai(url=INVALID_URL, auth_key=AUTH_KEY)
        j.delete_database(name="name")


def test_status_exception():
    with pytest.raises(ValueError):
        j = Jai(url=INVALID_URL, auth_key=AUTH_KEY)
        j.status(max_tries=2, patience=4)


def test_similar_id_exceptions():
    with pytest.raises(TypeError):
        j = Jai(url=INVALID_URL, auth_key=AUTH_KEY)
        j._similar_id("test", id_item=dict())

    with pytest.raises(ValueError):
        # we need to use a valid URL for this one
        j = Jai(url=VALID_URL, auth_key=AUTH_KEY)
        j._similar_id("test", id_item=[])


def test_similar_json_exception():
    with pytest.raises(ValueError):
        j = Jai(url=INVALID_URL, auth_key=AUTH_KEY)
        j._similar_json("test", data_json=dict())


def test_invalid_name_exception():
    with pytest.raises(ValueError):
        # we need to use a valid URL for this one
        j = Jai(url=VALID_URL, auth_key=AUTH_KEY)
        j._get_dtype("test")


def test_check_dtype_and_clean_exception():
    with pytest.raises(TypeError):
        j = Jai(url=INVALID_URL, auth_key=AUTH_KEY)
        j._check_dtype_and_clean(data=dict(), db_type="SelfSupervised")


def test_predict_exception():
    with pytest.raises(ValueError):
        j = Jai(url=INVALID_URL, auth_key=AUTH_KEY)
        j._predict(name="test", data_json=dict())


def test_append_exception():
    with pytest.raises(ValueError):
        j = Jai(url=INVALID_URL, auth_key=AUTH_KEY)
        j._append(name="test")


def test_insert_json_exception():
    with pytest.raises(ValueError):
        j = Jai(url=INVALID_URL, auth_key=AUTH_KEY)
        j._insert_json(name="test", df_json=dict())


def test_check_kwargs_exception():
    with pytest.raises(ValueError):
        j = Jai(url=INVALID_URL, auth_key=AUTH_KEY)
        j._check_kwargs(db_type="Supervised")


def test_setup_database_exception():
    with pytest.raises(ValueError):
        j = Jai(url=INVALID_URL, auth_key=AUTH_KEY)
        j._setup_database(name="test", db_type="SelfSupervised")


def test_embedding_exception():
    with pytest.raises(ValueError):
        j = Jai(url=INVALID_URL, auth_key=AUTH_KEY)
        j.embedding(name="test", data=dict())


def test_check_name_lengths_exception():
    with pytest.raises(ValueError):
        # we need to use a valid URL for this one
        j = Jai(url=VALID_URL, auth_key=AUTH_KEY)
        j._check_name_lengths(name="test", cols=[j.generate_name(length=35)])


@pytest.mark.parametrize("name, batch_size, db_type",
                         [("test", 1024, "SelfSupervised")])
def test_check_ids_consistency_exception(name, batch_size, db_type):
    # we need to use a valid URL for this one
    j = Jai(url=VALID_URL, auth_key=AUTH_KEY)

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
    j = Jai(url=VALID_URL, auth_key=AUTH_KEY)
    msg = f"Database '{name}' does not exist in your environment. Nothing to overwrite yet."
    assert j._delete_tree(name) == msg

def test_download_vectors_exception():
    with pytest.raises(ValueError):
        j = Jai(url=INVALID_URL, auth_key=AUTH_KEY)
        j.download_vectors(name="test")