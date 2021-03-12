from jai import Jai
from pandas.api.types import infer_dtype
from .test_utils import setup_dataframe
import pandas as pd
import numpy as np
import pytest

# URL = 'http://localhost:8001'
URL = 'http://23.96.99.211:8001'
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


@pytest.mark.parametrize("name", ["test_match"])
def test_embedding(name, setup_dataframe):

    A = [
        'Mandarin', 'Raspberry', 'Plum', 'Coconut', 'Kiwi', 'Grapefruit',
        'Grape', 'Lemon', 'Mandarin', 'Grape', 'Raspberry', 'Cherry', 'Plum',
        'Apple', 'Raspberry', 'Apricot', 'Watermelon', 'Blueberry',
        'Pineapple', 'Peach', 'Lime', 'Coconut', 'Mango', 'Grape', 'Avocado',
        'Apricot', 'Avocado', 'Pineapple', 'Avocado', 'Apple', 'Avocado',
        'Lemon', 'Lime', 'Cherry', 'Mandarin', 'Lime', 'Avocado', 'Papaya',
        'Mandarin', 'Apple', 'Apple', 'Pear', 'Papaya', 'Papaya', 'Apple',
        'Nectarine', 'Avocado', 'Apricot', 'Strawberry', 'Orange'
    ]

    B = [
        'Blyeberry', 'Otsnge', 'Mcngo', 'Wqtetmelob', 'Jaxofruif', 'Lear',
        'Leoon', 'Bludbecry', 'Kamgo', 'Aoricog', 'Zppke', 'Oaoaya', 'Appkr',
        'Chsrrt', 'Lapwya', 'Pescj', 'Plym', 'Xnerry', 'Avocarp', 'Mqhgo',
        'Nrctafije', 'Waterjepkn', 'Mwnearin', 'Apricov', 'Necgarinx',
        'Grapwfeuiy', 'Bsnaha', 'Apppe', 'Xtrswgerry', 'Apold', 'Peqr',
        'Nekon', 'Ljneaople', 'Hwnana', 'Mekoj', 'Oime', 'Lokegrahate',
        'Aoricit', 'Pineapoie', 'Avkcaeo', 'Avpvado', 'Cuerrg', 'Peqr',
        'Lsmin', 'Lemoj', 'Pomqgranatw', 'Aopls', 'Mxngi', 'Llmegranate',
        'Gfapd'
    ]
    expected = [
        17, 49, 22, 41, 7, 17, 22, 15, 13, 37, 13, 11, 37, 41, 2, 41, 24, 22,
        0, 15, 45, 5, 41, 13, 13, 41, 7, 22, 7, 20, 15, 18, 24, 24, 11, 41, 7,
        7, 13, 22, 6
    ]

    data_left = pd.Series(A)
    data_right = pd.Series(B)

    j = Jai(url=URL, auth_key=AUTH_KEY)
    if j.is_valid(name):
        j.delete_database(name)
    ok = j.match(name, data_left, data_right, top_k=40, original_data=True)

    assert ok['id_left'].tolist() == expected, "match failed"
