import numpy as np
import pandas as pd
import pytest

from jai.task.explorer import Explorer

#! ADD test_import_dataset


def test_names():
    j = Explorer(safe_mode=True)
    assert j.names == ["test_match", "test_resolution"]


def test_info():
    j = Explorer(safe_mode=True)

    info = j.info()
    assert isinstance(info, pd.DataFrame)

    info = info.drop("last modified", axis=1).reset_index(drop=True)
    res = pd.DataFrame(
        {
            "name": ["test_match", "test_resolution"],
            "type": ["TextEdit", "TextEdit"],
            "dependencies": [[], []],
            "size": [15, 43],
            "embedding_dimension": [128, 128],
        }
    )
    assert np.array_equal(info.values, res.values)


@pytest.mark.parametrize("safe_mode", ["False", "True"])
def test_user(safe_mode):
    j = Explorer(safe_mode=safe_mode)
    user = j.user()

    res = {
        "userId": "jquantinho",
        "email": "jquantinho@jquant.com.br",
        "firstName": "jquantinho",
        "lastName": "jotinha",
        "memberRole": "dev",
        "namespace": "testsdk",
    }
    assert user == res


@pytest.mark.parametrize("safe_mode", ["False", "True"])
def test_environments(safe_mode):
    j = Explorer(safe_mode=safe_mode)
    envs = j.environments()

    res = [
        {"key": "default", "id": "testsdk/test", "name": "testsdk_test"},
        {"key": "prod", "id": "testsdk/prod", "name": "testsdk_prod"},
    ]

    assert envs == res


@pytest.mark.parametrize("safe_mode", ["False", "True"])
@pytest.mark.parametrize("name", ["test_match", "test_resolution"])
def test_describe(safe_mode, name):
    j = Explorer(safe_mode=safe_mode)

    desc = j.describe(name)
    res_keys = [
        "name",
        "displayName",
        "owner",
        "project",
        "dtype",
        "state",
        "version",
        "has_filter",
        "features",
        "model_hyperparams",
    ]
    assert list(desc.keys()) == res_keys


@pytest.mark.parametrize("safe_mode", [False, True])
def test_rename(safe_mode):
    j = Explorer(safe_mode=safe_mode)

    assert j.names == [
        "test_match",
        "test_resolution",
    ]

    j.rename(original_name="test_match", new_name="test_match_new")
    assert j.names == [
        "test_match_new",
        "test_resolution",
    ]

    j.rename(original_name="test_match_new", new_name="test_match")
    assert j.names == [
        "test_match",
        "test_resolution",
    ]


@pytest.mark.parametrize("db_name", ["test_match"])
def test_transfer(db_name):
    j = Explorer(safe_mode=True)

    assert j.names == [
        "test_match",
        "test_resolution",
    ]

    j_prod = Explorer(safe_mode=True, environment="prod")
    assert j_prod.names == []

    j.transfer(original_name=db_name, to_environment="prod", from_environment="default")

    assert j_prod.names == [db_name]
