from jai import Jai
from .test_utils import setup_dataframe
import pandas as pd
import numpy as np
import pytest
import json
from decouple import config

URL = 'http://localhost:8000'
AUTH_KEY = ""
HEADER_TEST = json.loads(config('HEADER_TEST'))

np.random.seed(42)


@pytest.mark.parametrize("name", [('test_insert_vector')])
def test_insert_vectors(name, setup_dataframe):

    data, _ = setup_dataframe
    data = data.rename(columns={"PassengerId": "id"}).set_index("id")
    data = data.select_dtypes(exclude='object')

    j = Jai(url=URL, auth_key=AUTH_KEY)
    j.header = HEADER_TEST

    df0 = data[:400]
    overwrite = False
    if name in set(j.info['name']):
        j.delete_database(name)

    batch_size = 200
    j.insert_vectors(name=name,
                     data=df0,
                     batch_size=batch_size,
                     overwrite=overwrite)
    j_info = j.info
    assert name in set(j_info['name'])
    assert 'Vector' == j_info[j_info['name'] == name]['type'].unique()

    n_ids = len(j.ids(name, mode='complete'))
    assert n_ids == df0.shape[0]

    overwrite = True
    df1 = data.to_numpy()
    j.insert_vectors(name=name,
                     data=df1,
                     batch_size=batch_size,
                     overwrite=overwrite)

    n_ids = len(j.ids(name, mode='complete'))
    assert n_ids == data.shape[0]

    overwrite = False
    with pytest.raises(KeyError) as e:
        j.insert_vectors(name=name,
                         data=data,
                         batch_size=batch_size,
                         overwrite=overwrite)
        assert e.value.args[0] == f"Database 'test_insert_vector' already exists in your environment." \
            "Set overwrite=True to overwrite it."
