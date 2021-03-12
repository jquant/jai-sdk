from jai import Jai
from pandas.api.types import infer_dtype
from .test_utils import setup_dataframe
import pandas as pd
import numpy as np
import pytest

URL = 'http://localhost:8001'
AUTH_KEY = "sdk_test"

np.random.seed(42)


# =============================================================================
# Test Embedding
# =============================================================================
@pytest.mark.parametrize("name", ["test_embedding"])
def test_embedding(name, setup_dataframe):

    train, test = setup_dataframe
    train = train.rename(columns={"PassengerId": "id"}).set_index("id")['Name']
    test = test.rename(columns={"PassengerId": "id"}).set_index("id")['Name']

    j = Jai(url=URL, auth_key=AUTH_KEY)
    if j.is_valid(name):
        j.delete_database(name)

    j.embedding(name, train, overwrite=True)
    assert j.is_valid(name), f"valid name {name} after train embedding"

    j.embedding(name, test)
    assert j.is_valid(name), f"valid name {name} after test embedding"

    j.delete_database(name)
    assert not j.is_valid(name), "valid name after delete failed"


# =============================================================================
# Test Match Application
# =============================================================================
@pytest.mark.parametrize("name", ["test_match"])
def test_match(name):

    A = [
        "Apple", "Watermelon", "Orange", "Strawberry", "Nectarine", "Grape",
        "Blueberry", "Pomegranate", "Banana", "Raspberry", "Papaya",
        "Pineapple", "Lemon", "Grapefruit", "Coconut", "Avocado", "Peach"
    ]

    B = [
        'Coconit', 'Pdach', 'Appld', 'Piheapplr', 'Banxna', 'Avocado', 'Grwpe',
        'Grapw', 'Bluebeffy', 'Banwna', 'Strzwherry', 'Gdapefruir',
        'Aatermelon', 'Piheaplle', 'Grzpe', 'Watermelon', 'Kemon', 'Bqnana',
        'Bljwberry', 'Gralefruig', 'Rsspherry', 'Bahana', 'Watrrmeloh',
        'Pezch', 'Blusberrt', 'Grapegruit', 'Avocaeo'
    ]

    expected = [
        14, 16, 0, 11, 8, 15, 5, 5, 6, 8, 3, 13, 1, 11, 5, 1, 12, 8, 6, 13, 9,
        8, 1, 16, 6, 13, 15
    ]

    data_left = pd.Series(A)
    data_right = pd.Series(B)

    j = Jai(url=URL, auth_key=AUTH_KEY)
    if j.is_valid(name):
        j.delete_database(name)
    ok = j.match(name,
                 data_left,
                 data_right,
                 top_k=15,
                 threshold=0.5,
                 original_data=True)

    assert ok['id_left'].tolist() == expected, "match failed"


# =============================================================================
# Test Resolution Application
# =============================================================================
@pytest.mark.parametrize("name", ["test_resolution"])
def test_resolution(name):

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

    j = Jai(url=URL, auth_key=AUTH_KEY)
    if j.is_valid(name):
        j.delete_database(name)
    ok = j.resolution(name, data, top_k=20, threshold=.4, original_data=True)
    assert ok['resolution_id'].isin(expected).all(), "resolution failed"
