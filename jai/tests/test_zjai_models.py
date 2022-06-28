import json

import numpy as np
import pandas as pd
import pytest
from decouple import config

from jai import Jai

MAX_SIZE = 50

np.random.seed(42)


@pytest.fixture(scope="session")
def setup_dataframe():
    TITANIC_TRAIN = (
        "https://raw.githubusercontent.com/rebeccabilbro/titanic/master/data/train.csv"
    )
    TITANIC_TEST = (
        "https://raw.githubusercontent.com/rebeccabilbro/titanic/master/data/test.csv"
    )

    train = pd.read_csv(TITANIC_TRAIN)
    test = pd.read_csv(TITANIC_TEST)
    return train, test


# =============================================================================
# Test Text
# =============================================================================
@pytest.mark.parametrize("safe_mode", [True])
@pytest.mark.parametrize(
    "name,dtype",
    [
        ("test_nlp", "Text"),
        ("test_fasttext", "FastText"),
        ("test_edittext", "TextEdit"),
    ],
)
def test_text(safe_mode, name, dtype, setup_dataframe):
    train, _ = setup_dataframe
    train = (
        train.rename(columns={"PassengerId": "id"})
        .set_index("id")["Name"]
        .iloc[:MAX_SIZE]
    )
    ids = train.index.tolist()
    query = train.loc[np.random.choice(ids, 10, replace=False)]

    j = Jai(safe_mode=safe_mode)

    if j.is_valid(name):
        j.delete_database(name)

    j.setup(name, train, db_type=dtype, overwrite=True)
    assert j.is_valid(name), f"valid name {name} after setup failed"

    assert j.ids(name) == [
        f"{len(ids)} items from {min(ids)} to {max(ids)}"
    ], "ids simple failed"
    assert sorted(j.ids(name, "complete")) == ids, "ids complete failed"

    result = j.similar(name, query)
    assert isinstance(result, list), "similar data result failed"

    # TODO: improve this
    j.fields(name)

    j.delete_database(name)
    assert not j.is_valid(name), "valid name after delete failed"


@pytest.mark.parametrize("safe_mode", [True])
@pytest.mark.parametrize("name,dtype", [("test_filter_nlp", "Text")])
def test_filter_text(safe_mode, name, dtype, setup_dataframe):
    train, _ = setup_dataframe
    train = (
        train.rename(columns={"PassengerId": "id"})
        .set_index("id")[["Name", "Embarked"]]
        .iloc[:MAX_SIZE]
    )
    ids = train.index.tolist()
    query = train.loc[np.random.choice(ids, 10, replace=False), "Name"]

    j = Jai(safe_mode=safe_mode)

    if j.is_valid(name):
        j.delete_database(name)

    j.setup(
        name,
        train,
        features={"Embarked": {"dtype": "filter"}},
        db_type=dtype,
        overwrite=True,
    )
    assert j.is_valid(name), f"valid name {name} after setup failed"

    assert j.filters(name) == ["_default", "S", "C", "Q"], "filters failed"

    assert j.ids(name) == [
        f"{len(ids)} items from {min(ids)} to {max(ids)}"
    ], "ids simple failed"
    assert sorted(j.ids(name, "complete")) == ids, "ids complete failed"

    result = j.similar(name, query)
    assert isinstance(result, list), "similar data result failed"

    result = j.similar(name, query, filters="Q")
    assert isinstance(result, list), "similar data result failed"

    result = j.similar(name, query, orient="flat")
    assert isinstance(result, list), "similar data result failed"

    result = j.similar(name, pd.Series(query.index))
    assert isinstance(result, list), "similar id series result failed"

    result = j.similar(name, query.index)
    assert isinstance(result, list), "similar id index result failed"

    result = j.similar(name, query.index.tolist())
    assert isinstance(result, list), "similar id list result failed"

    result = j.similar(name, query.index.values)
    assert isinstance(result, list), "similar id array result failed"

    # TODO: improve this
    j.fields(name)

    j.delete_database(name)
    assert not j.is_valid(name), "valid name after delete failed"


# =============================================================================
# Test Self-supervised
# =============================================================================
@pytest.mark.parametrize("safe_mode", [True])
def test_selfsupervised(setup_dataframe, safe_mode):
    name = "test_selfsupervised"

    train, _ = setup_dataframe
    train = train.drop(columns=["PassengerId"]).iloc[:MAX_SIZE]
    query = train.loc[np.random.choice(len(train), 10, replace=False)]

    j = Jai(safe_mode=safe_mode)

    if j.is_valid(name):
        j.delete_database(name)

    j.setup(
        name,
        train,
        db_type="SelfSupervised",
        hyperparams={"max_epochs": 3},
        overwrite=True,
        max_insert_workers=1,
    )

    assert j.is_valid(name), f"valid name {name} after setup failed"

    ids = train.index.tolist()
    assert j.ids(name) == [
        f"{len(ids)} items from {min(ids)} to {max(ids)}"
    ], "ids simple failed"
    assert j.ids(name, "complete") == ids, "ids complete failed"

    # TODO: improve this
    j.fields(name).items()

    result = j.similar(name, query)

    # try to use j.predict on a self-supervised database
    # this will raise an exception
    with pytest.raises(ValueError):
        j.predict(name, dict())

    assert isinstance(result, list), "similar result failed"

    # try to set up the same database again
    # without overwriting it
    with pytest.raises(KeyError):
        j.setup(name, train, db_type="SelfSupervised")

    j.delete_database(name)
    assert not j.is_valid(name), "valid name after delete failed"


# =============================================================================
# Test Supervised
# =============================================================================
@pytest.mark.parametrize("safe_mode", [True])
def test_supervised(setup_dataframe, safe_mode):
    name = "test_supervised"

    train, test = setup_dataframe
    train = train.rename(columns={"PassengerId": "id"}).iloc[:MAX_SIZE]
    test = test.rename(columns={"PassengerId": "id"}).iloc[:MAX_SIZE]
    query = test.loc[np.random.choice(len(test), 10, replace=False)]

    j = Jai(safe_mode=safe_mode)

    if j.is_valid(name):
        j.delete_database(name)

    j.fit(
        name,
        train,
        db_type="Supervised",
        overwrite=True,
        max_insert_workers=0,
        hyperparams={"max_epochs": 3},
        label={"task": "metric_classification", "label_name": "Survived"},
        split={"type": "stratified", "split_column": "Survived", "test_size": 0.2},
    )

    assert j.is_valid(name), f"valid name {name} after setup failed"

    ids = train["id"].tolist()
    assert j.ids(name) == [
        f"{len(ids)} items from {min(ids)} to {max(ids)}"
    ], "ids simple failed"
    assert j.ids(name, "complete") == ids, "ids complete failed"

    for k, from_api in j.fields(name).items():
        if k == "Survived":
            continue
        original = str(train[k].dtype)
        if original == "object":
            original = "string"
        assert (
            original == from_api
        ), "dtype from api {from_api} differ from data {original}"

    result = j.similar(name, query)
    assert isinstance(result, list), "similar result failed"

    result = j.predict(name, query)

    # since we have a supervised database already inplace
    # we test one of its exceptions
    with pytest.raises(ValueError):
        j.predict(name, dict())

    assert isinstance(result, list), "predict result failed"

    j.append(name, test)

    ids = train["id"].tolist() + test["id"].tolist()
    assert j.ids(name) == [
        f"{len(ids)} items from {min(ids)} to {max(ids)}"
    ], "ids simple failed"
    assert j.ids(name, "complete") == ids, "ids complete failed"

    # test _delete_tree method here
    j._delete_tree(name)
    assert not j.is_valid(name), "valid name after delete failed"


@pytest.mark.parametrize("name,safe_mode", [("test_recommendation", True)])
def test_recommendation(name, safe_mode):
    mock_db = pd.DataFrame(
        {"User": [0, 1, 2, 0, 1, 2, 1, 1, 0, 2], "Item": [2, 3, 1, 5, 1, 2, 4, 3, 2, 1]}
    )

    mock_users = pd.DataFrame({"User": [1, 2, 3], "id": [0, 1, 2]})
    mock_items = pd.DataFrame(
        {"id": [1, 2, 3, 4, 5], "Colour": ["black", "white", "green", "yellow", "blue"]}
    )
    data = {"users": mock_users, "items": mock_items, "main": mock_db}

    j = Jai(safe_mode=safe_mode)
    j.fit(
        name,
        data,
        db_type="RecommendationSystem",
        overwrite=True,
        pretrained_bases=[
            {"id_name": "User", "db_parent": "users"},
            {"id_name": "Item", "db_parent": "items"},
        ],
    )

    assert j.is_valid(name), f"valid name {name} after setup failed"
    assert j.is_valid("users"), f"valid name users after setup failed"
    assert j.is_valid("items"), f"valid name items after setup failed"

    users_ids = list(mock_users.index)
    j_users = Jai(safe_mode=safe_mode)
    assert j_users.ids("users") == [
        f"{len(users_ids)} items from {min(users_ids)} to {max(users_ids)}"
    ], "ids simple failed"
    assert j_users.ids("users", "complete") == users_ids, "ids complete failed"

    items_ids = list(mock_items["id"])
    j_trainer = Jai(safe_mode=safe_mode)
    assert j_trainer.ids("items") == [
        f"{len(items_ids)} items from {min(items_ids)} to {max(items_ids)}"
    ], "ids simple failed"
    assert j_trainer.ids("items", "complete") == items_ids, "ids complete failed"

    result = j.recommendation(name="users", data=mock_items, top_k=2)
    assert isinstance(result, list), "recommendation result failed"
    assert list(result[0].keys()) == ["query_id", "results"]

    result = j.recommendation(
        name="items", data=mock_users.index, top_k=2, orient="flat"
    )
    assert isinstance(result, list), "recommendation result failed"
    assert list(result[0].keys()) == ["query_id", "id", "distance"]

    j.delete_database(name)
    j.delete_database("users")
    j.delete_database("items")
