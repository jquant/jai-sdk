import json

import numpy as np
import pandas as pd
import pytest
from decouple import config

from jai import Jai

np.random.seed(42)


@pytest.fixture(scope="session")
def setup_dataframe():
    TITANIC_TRAIN = "https://raw.githubusercontent.com/rebeccabilbro/titanic/master/data/train.csv"
    TITANIC_TEST = "https://raw.githubusercontent.com/rebeccabilbro/titanic/master/data/test.csv"

    train = pd.read_csv(TITANIC_TRAIN)
    test = pd.read_csv(TITANIC_TEST)
    return train, test


# =============================================================================
# Test Embedding
# =============================================================================
@pytest.mark.parametrize("safe_mode", [True])
@pytest.mark.parametrize("name", ["test_embedding"])
def test_embedding(safe_mode, name, setup_dataframe):

    train, test = setup_dataframe
    train = train.rename(columns={
        "PassengerId": "id"
    }).set_index("id")['Name'].iloc[:10]
    test = test.rename(columns={
        "PassengerId": "id"
    }).set_index("id")['Name'].iloc[:10]

    j = Jai(safe_mode=safe_mode)

    if j.is_valid(name):
        j.delete_database(name)

    j.embedding(name, train, overwrite=True)
    assert j.is_valid(name), f"valid name {name} after train embedding"

    j.embedding(name, test)
    assert j.is_valid(name), f"valid name {name} after test embedding"

    j.delete_database(name)
    assert not j.is_valid(name), "valid name after delete failed"


# =============================================================================
# Test Fill
# =============================================================================
@pytest.mark.parametrize("safe_mode", [True])
@pytest.mark.parametrize("name", ["test_fill"])
def test_fill(safe_mode, name, setup_dataframe):

    train, test = setup_dataframe
    train = train.set_index("PassengerId").iloc[:10]
    test = test.set_index("PassengerId").iloc[:10]
    half = test.shape[0] // 2
    data = pd.concat([train, test.iloc[:half]])

    j = Jai(safe_mode=safe_mode)

    for n in j.names:
        if n.startswith(name):
            j.delete_database(n)

    x = j.fill(name, data, column="Survived")
    assert j.is_valid(name), f"valid name {name} after train fill"
    assert j.ids(name) == ['15 items from 1 to 896'], 'wrong ids values sanity'

    v = j.fill(name, test.iloc[half:], column="Survived")

    assert j.ids(name) == ['20 items from 1 to 901'], 'wrong ids values sanity'
    j.delete_database(name)
    assert not j.is_valid(name), "valid name after delete failed"


# =============================================================================
# Test Sanity
# =============================================================================
@pytest.mark.parametrize("safe_mode", [True])
@pytest.mark.parametrize("name", ["test_sanity"])
def test_sanity(safe_mode, name, setup_dataframe):

    train, test = setup_dataframe
    train = train.set_index("PassengerId").iloc[:50]
    test = test.set_index("PassengerId").iloc[:50]
    half = test.shape[0] // 2
    data = pd.concat([train, test.iloc[:half]]).drop(columns=['Survived'])

    j = Jai(safe_mode=safe_mode)

    for n in j.names:
        if n.startswith(name):
            j.delete_database(n)

    j.sanity(name, data)
    assert j.is_valid(name), f"valid name {name} after train sanity"

    v = j.sanity(name, test.iloc[half:])

    j.delete_database(name)
    assert not j.is_valid(name), "valid name after delete failed"


# =============================================================================
# Test Match Application
# =============================================================================
@pytest.mark.parametrize("safe_mode", [True])
@pytest.mark.parametrize("name", ["test_match"])
def test_match(safe_mode, name):

    A = [
        "Apple", "Watermelon", "Orange", "Nectarine", "Grape", "Lemon",
        "Blueberry", "Pomegranate", "Banana", "Papaya", "Pineapple",
        "Grapefruit", "Coconut", "Avocado", "Peach"
    ]

    B = [
        'Coconit', 'Pdach', 'Appld', 'Piheapplr', 'Banxna', 'Avocado', 'Grwpe'
    ]

    expected = [12, 14, 0, 10, 8, 13, 4]

    data_left = pd.Series(A)
    data_right = pd.Series(B)

    j = Jai(safe_mode=safe_mode)

    if j.is_valid(name):
        j.delete_database(name)
    ok = j.match(name,
                 data_left,
                 data_right,
                 top_k=15,
                 threshold=0.5,
                 original_data=True,
                 overwrite=True)

    assert ok['id_left'].tolist() == expected, "match failed"


# =============================================================================
# Test Resolution Application
# =============================================================================
@pytest.mark.parametrize("safe_mode", [True])
@pytest.mark.parametrize("name", ["test_resolution"])
def test_resolution(safe_mode, name):

    data = [
        "Apple", "Watermelon", "Orange", "Strawberry", "Nectarine", "Grape",
        "Blueberry", "Pomegranate", "Banana", "Raspberry", "Papaya",
        "Pineapple", "Lemon", "Grapefruit", "Coconut", "Avocado", "Peach",
        'Coconit', 'Pdach', 'Appld', 'Piheapplr', 'Banxna', 'Avocado', 'Grwpe',
        'Grapw', 'Bluebeffy', 'Banwna', 'Strzwherry', 'Gdapefruir',
        'Aatermelon', 'Piheaplle', 'Grzpe', 'Watermelon', 'Kemon', 'Bqnana',
        'Bljwberry', 'Rsspherry', 'Bahana', 'Watrrmeloh', 'Pezch', 'Blusberrt',
        'Grapegruit', 'Avocaeo'
    ]
    expected = np.arange(19)
    data = pd.Series(data)

    j = Jai(safe_mode=safe_mode)

    if j.is_valid(name):
        j.delete_database(name)
    ok = j.resolution(name, data, top_k=20, threshold=.4, original_data=True)
    assert ok['resolution_id'].isin(expected).all(), "resolution failed"
