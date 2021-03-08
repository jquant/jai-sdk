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