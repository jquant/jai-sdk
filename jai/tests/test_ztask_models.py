import numpy as np
import pandas as pd
import pytest

from jai.task.trainer import Trainer

np.random.seed(42)
MAX_SIZE = 50


@pytest.fixture(scope="session")
def setup_dataframe():
    TITANIC_TRAIN = "https://raw.githubusercontent.com/rebeccabilbro/titanic/master/data/train.csv"
    TITANIC_TEST = "https://raw.githubusercontent.com/rebeccabilbro/titanic/master/data/test.csv"

    train = pd.read_csv(TITANIC_TRAIN)
    test = pd.read_csv(TITANIC_TEST)
    return train, test


# =============================================================================
# Test Text
# =============================================================================
@pytest.mark.parametrize("safe_mode", [True])
@pytest.mark.parametrize("name,dtype", [("test_nlp", "Text"),
                                        ("test_fasttext", "FastText"),
                                        ("test_edittext", "TextEdit")])
def test_text(safe_mode, name, dtype, setup_dataframe):
    train, _ = setup_dataframe
    train = train.rename(columns={
        "PassengerId": "id"
    }).set_index("id")['Name'].iloc[:MAX_SIZE]
    ids = train.index.tolist()
    sample = train.loc[np.random.choice(ids, 10, replace=False)]

    trainer = Trainer(name=name, safe_mode=safe_mode)
    trainer.set_params(db_type=dtype)

    query = trainer.fit(train, overwrite=True)

    assert trainer.ids('simple') == [
        f"{len(ids)} items from {min(ids)} to {max(ids)}"
    ], 'ids simple failed'
    assert sorted(trainer.ids('complete')) == ids, "ids complete failed"
    assert trainer.is_valid(), f"valid name {name} after fit failed"

    # try to use the fields method on a text database
    # this will raise an exception
    with pytest.raises(ValueError):
        trainer.fields()

    result = query.similar(sample)
    assert isinstance(result, list), "similar data result failed"

    trainer.delete_database()
    assert not trainer.is_valid(), "valid name after delete failed"


@pytest.mark.parametrize("safe_mode", [True])
@pytest.mark.parametrize("name,dtype", [("test_filter_nlp", "Text")])
def test_filter_text(safe_mode, name, dtype, setup_dataframe):
    train, _ = setup_dataframe
    train = train.rename(columns={
        "PassengerId": "id"
    }).set_index("id")[['Name', 'Embarked']].iloc[:MAX_SIZE]
    ids = train.index.tolist()
    sample = train.loc[np.random.choice(ids, 10, replace=False), 'Name']

    trainer = Trainer(name=name, safe_mode=safe_mode)
    with pytest.raises(ValueError):
        trainer.setup_params

    trainer.set_params(db_type=dtype,
                       features={'Embarked': {
                           "dtype": "filter"
                       }})

    if trainer.is_valid():
        trainer.delete_database()

    query = trainer.fit(train, overwrite=True)
    assert trainer.is_valid(), f"valid name {name} after fit failed"

    assert trainer.filters() == ['_default', 'S', 'C', 'Q'], 'filters failed'

    assert trainer.ids('simple') == [
        f"{len(ids)} items from {min(ids)} to {max(ids)}"
    ], 'ids simple failed'
    assert sorted(trainer.ids('complete')) == ids, "ids complete failed"

    result = query.similar(sample)
    assert isinstance(result, list), "similar data result failed"

    result = query.similar(sample, filters="Q")
    assert isinstance(result, list), "similar data result failed"

    result = query.similar(sample, orient="flat")
    assert isinstance(result, list), "similar data result failed"

    result = query.similar(pd.Series(sample.index))
    assert isinstance(result, list), "similar id series result failed"

    result = query.similar(sample.index)
    assert isinstance(result, list), "similar id index result failed"

    result = query.similar(sample.index.tolist())
    assert isinstance(result, list), "similar id list result failed"

    result = query.similar(sample.index.values)
    assert isinstance(result, list), "similar id array result failed"

    # try to use the fields method on a text database
    # this will raise an exception
    with pytest.raises(ValueError):
        trainer.fields()

    trainer.delete_database()
    assert not trainer.is_valid(), "valid name after delete failed"


# =============================================================================
# Test Self-supervised
# =============================================================================
@pytest.mark.parametrize("safe_mode", [True])
def test_selfsupervised(setup_dataframe, safe_mode):
    name = 'test_selfsupervised'

    train, _ = setup_dataframe
    train = train.drop(columns=["PassengerId"]).iloc[:MAX_SIZE]
    sample = train.loc[np.random.choice(len(train), 10, replace=False)]

    trainer = Trainer(name=name, safe_mode=safe_mode)
    trainer.set_params(db_type="SelfSupervised", hyperparams={"max_epochs": 3})

    if trainer.is_valid():
        trainer.delete_database()

    query = trainer.fit(train, overwrite=True)

    assert trainer.is_valid(), f"valid name {name} after fit failed"

    ids = train.index.tolist()
    assert trainer.ids("simple") == [
        f"{len(ids)} items from {min(ids)} to {max(ids)}"
    ], 'ids simple failed'
    assert trainer.ids('complete') == ids, "ids complete failed"

    for k, from_api in trainer.fields().items():
        if k == 'id':
            continue
        original = str(train[k].dtype)
        if original == 'object':
            original = 'string'
        assert original == from_api, "dtype from api {from_api} differ from data {original}"

    result = query.similar(sample)

    # try to use j.predict on a self-supervised database
    # this will raise an exception
    with pytest.raises(ValueError):
        query.predict(dict())

    assert isinstance(result, list), "similar result failed"

    # try to set up the same database again
    # without overwriting it
    with pytest.raises(KeyError):
        trainer.fit(train)

    trainer.delete_database()
    assert not trainer.is_valid(), "valid name after delete failed"


# =============================================================================
# Test Supervised
# =============================================================================
@pytest.mark.parametrize("safe_mode", [True])
def test_supervised(setup_dataframe, safe_mode):
    name = 'test_supervised'

    train, test = setup_dataframe
    train = train.rename(columns={"PassengerId": "id"}).iloc[:MAX_SIZE]
    test = test.rename(columns={"PassengerId": "id"}).iloc[:MAX_SIZE]
    sample = test.loc[np.random.choice(len(test), 10, replace=False)]

    trainer = Trainer(name=name, safe_mode=safe_mode)
    trainer.set_params(db_type="Supervised",
                       hyperparams={"max_epochs": 3},
                       label={
                           "task": "metric_classification",
                           "label_name": "Survived"
                       },
                       split={
                           "type": 'stratified',
                           "split_column": "Survived",
                           "test_size": .2
                       })
    if trainer.is_valid():
        trainer.delete_database()

    query = trainer.fit(train, overwrite=True)

    assert trainer.is_valid(), f"valid name {name} after fit failed"

    ids = train['id'].tolist()
    assert trainer.ids("simple") == [
        f"{len(ids)} items from {min(ids)} to {max(ids)}"
    ], 'ids simple failed'
    assert trainer.ids('complete') == ids, "ids complete failed"

    for k, from_api in trainer.fields().items():
        if k == 'Survived':
            continue
        original = str(train[k].dtype)
        if original == 'object':
            original = 'string'
        assert original == from_api, "dtype from api {from_api} differ from data {original}"

    result = query.similar(sample)
    assert isinstance(result, list), "similar result failed"

    result = query.predict(sample)
    assert isinstance(result, list), "predict result failed"

    # since we have a supervised database already inplace
    # we test one of its exceptions
    with pytest.raises(ValueError):
        query.predict(dict())

    trainer.append(test)

    ids = train['id'].tolist() + test['id'].tolist()
    assert trainer.ids("simple") == [
        f"{len(ids)} items from {min(ids)} to {max(ids)}"
    ], 'ids simple failed'
    assert trainer.ids('complete') == ids, "ids complete failed"

    trainer.delete_database()
    assert not trainer.is_valid(), "valid name after delete failed"
