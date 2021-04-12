from jai import Jai
from .test_utils import setup_dataframe
import pandas as pd
import numpy as np
import pytest

URL = 'http://23.96.99.211:8001'
AUTH_KEY = "sdk_test"

np.random.seed(42)


# =============================================================================
# Test Text
# =============================================================================
@pytest.mark.parametrize("name,data,dtype",
                         [("test_nlp", "list", "Text"),
                          ("test_fasttext", "array", "FastText"),
                          ("test_edittext", "series", "TextEdit")])
def test_text(name, data, dtype, setup_dataframe):
    train, _ = setup_dataframe
    train = train.rename(columns={"PassengerId": "id"}).set_index("id")['Name']
    ids = train.index.tolist()
    query = train.loc[np.random.choice(ids, 10, replace=False)]

    if data == 'list':
        train = train.tolist()
        ids = list(range(len(train)))
    elif data == 'array':
        train = train.values
        ids = list(range(len(train)))
    else:
        pass

    j = Jai(url=URL, auth_key=AUTH_KEY)
    if j.is_valid(name):
        j.delete_database(name)

    j.setup(name, train, db_type=dtype, overwrite=True)
    assert j.is_valid(name), f"valid name {name} after setup failed"

    assert j.ids(name) == [f"{len(ids)} items from {min(ids)} to {max(ids)}"
                           ], 'ids simple failed'
    assert sorted(j.ids(name, 'complete')) == ids, "ids complete failed"

    result = j.similar(name, query)
    assert isinstance(result, list), "similar data result failed"

    result = j.similar(name, query.to_numpy())
    assert isinstance(result, list), "similar data (np.array) result failed"

    result = j.similar(name, pd.Series(query.index))
    assert isinstance(result, list), "similar id series result failed"

    result = j.similar(name, query.index)
    assert isinstance(result, list), "similar id index result failed"

    result = j.similar(name, query.index.tolist())
    assert isinstance(result, list), "similar id list result failed"

    result = j.similar(name, query.index.values)
    assert isinstance(result, list), "similar id array result failed"

    # try to use the fields method on a text database
    # this will raise an exception
    with pytest.raises(ValueError):
        j.fields(name)

    j.delete_database(name)
    assert not j.is_valid(name), "valid name after delete failed"


# =============================================================================
# Test Self-supervised
# =============================================================================
def test_selfsupervised(setup_dataframe):
    name = 'test_selfsupervised'

    train, _ = setup_dataframe
    train = train.drop(columns=["PassengerId"])
    query = train.loc[np.random.choice(len(train), 10, replace=False)]

    j = Jai(url=URL, auth_key=AUTH_KEY)
    if j.is_valid(name):
        j.delete_database(name)

    j.setup(name, train, db_type="SelfSupervised", overwrite=True)

    assert j.is_valid(name), f"valid name {name} after setup failed"

    ids = train.index.tolist()
    assert j.ids(name) == [f"{len(ids)} items from {min(ids)} to {max(ids)}"
                           ], 'ids simple failed'
    assert j.ids(name, 'complete') == ids, "ids complete failed"

    for k, from_api in j.fields(name).items():
        if k == 'id':
            continue
        original = str(train[k].dtype)
        if original == 'object':
            original = 'string'
        assert original == from_api, "dtype from api {from_api} differ from data {original}"

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
def test_supervised(setup_dataframe):
    name = 'test_supervised'

    train, test = setup_dataframe
    train = train.rename(columns={"PassengerId": "id"})
    test = test.rename(columns={"PassengerId": "id"})
    query = test.loc[np.random.choice(len(test), 10, replace=False)]

    j = Jai(url=URL, auth_key=AUTH_KEY)
    if j.is_valid(name):
        j.delete_database(name)

    j.setup(name,
            train,
            db_type="Supervised",
            overwrite=True,
            label={
                "task": "metric_classification",
                "label_name": "Survived"
            },
            split={
                "type": 'stratified',
                "split_column": "Survived",
                "test_size": .2
            })

    assert j.is_valid(name), f"valid name {name} after setup failed"

    ids = train['id'].tolist()
    assert j.ids(name) == [f"{len(ids)} items from {min(ids)} to {max(ids)}"
                           ], 'ids simple failed'
    assert j.ids(name, 'complete') == ids, "ids complete failed"

    for k, from_api in j.fields(name).items():
        if k == 'Survived':
            continue
        original = str(train[k].dtype)
        if original == 'object':
            original = 'string'
        assert original == from_api, "dtype from api {from_api} differ from data {original}"

    result = j.similar(name, query)
    assert isinstance(result, list), "similar result failed"

    result = j.predict(name, query)

    # since we have a supervised database already inplace
    # we test one of its exceptions
    with pytest.raises(ValueError):
        j.predict(name, dict())

    assert isinstance(result, list), "predict result failed"

    j.add_data(name, test)

    ids = train['id'].tolist() + test['id'].tolist()
    assert j.ids(name) == [f"{len(ids)} items from {min(ids)} to {max(ids)}"
                           ], 'ids simple failed'
    assert j.ids(name, 'complete') == ids, "ids complete failed"

    # test _delete_tree method here
    j._delete_tree(name)
    assert not j.is_valid(name), "valid name after delete failed"
