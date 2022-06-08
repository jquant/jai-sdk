import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from decouple import config

from jai import Jai
import os
from copy import deepcopy


@pytest.fixture(scope="function")
def clean_environ():
    # Remove JAI_URL from environment variables
    old_environ = deepcopy(os.environ)
    os.environ.pop("JAI_URL", None)

    # Remove JAI_URL from config (.env, settings.ini)
    config("key", "anything")
    old_data = deepcopy(config.config.repository.data)
    config.config.repository.data.pop("JAI_URL", None)
    yield
    # restore initial values
    os.environ = old_environ
    config.config.repository.data = old_data


@pytest.fixture(scope="session")
def setup_npy_file():
    NPY_FILE = Path("jai/test_data/sdk_test_titanic_ssupervised.npy")
    img_file = np.load(NPY_FILE)
    return img_file


def test_url(clean_environ):
    j = Jai()
    assert j.url == "https://mycelia.azure-api.net"


def test_custom_url():
    j = Jai()
    j.url = "http://localhost:8001/"
    assert j.url == "http://localhost:8001"


@pytest.mark.parametrize("safe_mode", [False, True])
def test_names(safe_mode):
    j = Jai(safe_mode=safe_mode)
    assert j.names == [
        "test_insert_vector",
        "test_match",
        "test_resolution",
        "titanic_ssupervised",
    ], f"Failed names {j.names}"


@pytest.mark.parametrize("safe_mode", [False, True])
def test_info(safe_mode):
    j = Jai(safe_mode=safe_mode)

    assert isinstance(j.info, pd.DataFrame)


@pytest.mark.parametrize("safe_mode", [False, True])
def test_status(safe_mode):
    j = Jai(safe_mode=safe_mode)
    assert isinstance(j.status(), dict)


@pytest.mark.parametrize("length", [10, 15])
@pytest.mark.parametrize("prefix", ["", "pre_"])
@pytest.mark.parametrize("suffix", ["", "_fix"])
def test_generate_name(length, prefix, suffix):
    j = Jai()
    name = j.generate_name(length, prefix, suffix)
    assert len(name) == length, "generated name wrong."

    if prefix != "":
        assert name.startswith(prefix), "prefix not in generated name."

    if suffix != "":
        assert name.endswith(suffix), "suffix not in generated name."


@pytest.mark.parametrize("safe_mode", [False, True])
@pytest.mark.parametrize("name", ["titanic_ssupervised"])
def test_download_vectors(safe_mode, setup_npy_file, name):
    npy_file = setup_npy_file
    j = Jai(safe_mode=safe_mode)
    np.testing.assert_array_equal(npy_file, j.download_vectors(name=name))


@pytest.mark.parametrize("safe_mode", [False, True])
def test_user(safe_mode):
    j = Jai(safe_mode=safe_mode)
    user = j.user()
    assert user["memberRole"] == "dev"
    assert user["namespace"] == "testsdk"


@pytest.mark.parametrize("safe_mode", [False, True])
def test_environments(safe_mode):
    j = Jai(safe_mode=safe_mode)
    assert j.environments() == [
        {"key": "default", "id": "sdk/test", "name": "sdk_test"},
        {"id": "sdk/prod", "name": "sdk_prod"},
    ]


@pytest.mark.parametrize("safe_mode", [False, True])
@pytest.mark.parametrize("name", ["test_resolution"])
def test_describe(safe_mode, name):
    j = Jai(safe_mode=safe_mode)
    description = j.describe(name)
    description.pop("version")
    assert description == {
        "dtype": "TextEdit",
        "features": [{"dtype": "text", "name": "0"}],
        "has_filter": False,
        "model_hyperparams": {
            "batch_size": 128,
            "channel": 8,
            "embed_dim": 128,
            "epochs": 20,
            "k": 100,
            "maxl": 0,
            "mtc": False,
            "nb": 1385451,
            "nr": 1000,
            "nt": 1000,
            "random_append_train": False,
            "random_train": False,
            "shuffle_seed": 808,
            "test_batch_size": 1024,
        },
        "name": "test_resolution",
        "state": "active",
    }


@pytest.mark.parametrize("safe_mode", [False, True])
def test_rename(safe_mode):
    j = Jai(safe_mode=safe_mode)
    assert j.names == [
        "test_insert_vector",
        "test_match",
        "test_resolution",
        "titanic_ssupervised",
    ]

    j.rename(original_name="test_match", new_name="test_match_new")
    assert j.names == [
        "test_insert_vector",
        "test_match_new",
        "test_resolution",
        "titanic_ssupervised",
    ]

    j.rename(original_name="test_match_new", new_name="test_match")
    assert j.names == [
        "test_insert_vector",
        "test_match",
        "test_resolution",
        "titanic_ssupervised",
    ]


@pytest.mark.parametrize("safe_mode", [False, True])
@pytest.mark.parametrize("db_name", ["test_match"])
def test_transfer(safe_mode, db_name):
    j = Jai(safe_mode=safe_mode)
    assert j.names == [
        "test_insert_vector",
        "test_match",
        "test_resolution",
        "titanic_ssupervised",
    ]

    j_prod = Jai(safe_mode=safe_mode, environment="prod")
    assert j_prod.headers["environment"] == "prod"

    if db_name in j_prod.names:
        j_prod.delete_database(db_name)
    assert j_prod.names == ["titanic_ssupervised"]

    j.transfer(original_name=db_name, to_environment="prod", from_environment="default")

    assert j_prod.names == [db_name, "titanic_ssupervised"]

    j_prod.delete_database(db_name)
    assert j_prod.names == ["titanic_ssupervised"]
