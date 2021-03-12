from jai import Jai
from pandas.api.types import infer_dtype
from .test_utils import setup_dataframe
import pandas as pd
import numpy as np
import pytest

URL = 'http://localhost:8001'
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


# =============================================================================
# Test Match Application
# =============================================================================
@pytest.mark.parametrize("name", ["test_match"])
def test_match(name):

    A = [
        "Apple", "Watermelon", "Orange", "Pear", "Cherry", "Strawberry",
        "Nectarine", "Grape", "Mango", "Blueberry", "Pomegranate", "Plum",
        "Banana", "Raspberry", "Mandarin", "Jackfruit", "Papaya", "Kiwi",
        "Pineapple", "Lime", "Lemon", "Apricot", "Grapefruit", "Melon",
        "Coconut", "Avocado", "Peach"
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
        9, 2, 8, 15, 3, 20, 9, 8, 21, 0, 16, 0, 4, 16, 3, 11, 3, 25, 23, 14,
        21, 6, 22, 12, 0, 0, 3, 23, 12, 23, 19, 21, 18, 25, 25, 4, 3, 20, 20,
        0, 8, 10, 7
    ]

    data_left = pd.Series(A)
    data_right = pd.Series(B)

    j = Jai(url=URL, auth_key=AUTH_KEY)
    if j.is_valid(name):
        j.delete_database(name)
    ok = j.match(name,
                 data_left,
                 data_right,
                 top_k=20,
                 threshold=.35,
                 original_data=True)

    assert ok['id_left'].tolist() == expected, "match failed"


# =============================================================================
# Test Resolution Application
# =============================================================================
@pytest.mark.parametrize("name", ["test_resolution"])
def test_resolution(name):

    data = [
        'Mandarin', 'Raspberry', 'Plum', 'Coconut', 'Kiwi', 'Grapefruit',
        'Grape', 'Lemon', 'Nectarine', 'Orange', 'Raspberry', 'Cherry', 'Plum',
        'Apple', 'Raspberry', 'Apricot', 'Watermelon', 'Blueberry', 'Banana',
        'Strawberry', 'Pineapple', 'Peach', 'Lime', 'Coconut', 'Mango',
        'Papaya', 'Pomegranate', 'Grape', 'Avocado', 'Apricot', 'Jackfruit',
        'Pineapple', 'Avocado', 'Apple', 'Avocado', 'Lemon', 'Lime', 'Cherry',
        'Mandarin', 'Lime', 'Avocado', 'Papaya', 'Mandarin', 'Apple', 'Apple',
        'Pear', 'Papaya', 'Papaya', 'Apple', 'Avocado', 'Kiwi', 'Plum',
        'Pomsgranate', 'Kiwi', 'Javkfruit', 'Apple', 'Peach', 'Melon', 'Kiwi',
        'Melon', 'Orangd', 'Cjerry', 'Cocknut', 'Watermslon', 'Mango', 'Mango',
        'Plum', 'Pinrapple', 'Chwrry', 'Peach', 'Banxna', 'Orxnge', 'Mandarim',
        'Pomegrabate', 'Mandafin', 'Xherry', 'Strawberty', 'Neftarine',
        'Mandqrin', 'Stfawberry', 'Apple', 'Apeicot', 'Avocwdo', 'Lime',
        'Palaya', 'Melon', 'Lemon', 'Apple', 'Lime', 'Blueberrt', 'Easpberry',
        'Qvocado', 'Strawbsrry', 'Apple', 'Psar', 'Pwar', 'Watermelom',
        'Peach', 'Prange', 'Kiqi'
    ]
    expected = np.arange(41)
    data = pd.Series(data)

    j = Jai(url=URL, auth_key=AUTH_KEY)
    if j.is_valid(name):
        j.delete_database(name)
    ok = j.resolution(name, data, top_k=20, threshold=.25, original_data=True)
    assert ok['resolution_id'].isin(expected).all(), "resolution failed"
