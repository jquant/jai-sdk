from jai import Jai
import numpy as np
import pytest
import json
from decouple import config

URL = 'http://localhost:8001'
AUTH_KEY = ""
HEADER_TEST = json.loads(config('HEADER_TEST'))

np.random.seed(42)


@pytest.mark.parametrize("name", [('test_insert_vector')])
def test_insert_vectors(name, setup_dataframe):

    data, _ = setup_dataframe
    data = data.drop(columns="Cabin").dropna().rename(columns={
        "PassengerId": "id"
    }).set_index("id")
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


@pytest.mark.parametrize("name", [('test_insert_vector')])
def test_append_vectors(name, setup_dataframe):

    data, _ = setup_dataframe
    data = data.drop(columns=["PassengerId", "Cabin"]).dropna().reset_index(
        drop=True)
    data = data.select_dtypes(exclude='object')

    j = Jai(url=URL, auth_key=AUTH_KEY)
    j.header = HEADER_TEST

    df0 = data[:100].iloc[:, :3]
    j.insert_vectors(name=name, data=df0, overwrite=True)
    j_info = j.info
    length = j_info.loc[j_info['name'] == 'test_insert_vector',
                        'size'].values[0]
    assert length == 100

    df1 = data[100:200].iloc[:, :3]
    j.insert_vectors(name=name, data=df1, append=True)
    j_info = j.info
    length = j_info.loc[j_info['name'] == 'test_insert_vector',
                        'size'].values[0]
    assert length == 200


@pytest.mark.parametrize("name, pretrained",
                         [('test_insert_vector_ss', 'test_insert_vector')])
def test_pretrained_with_vectors(name, pretrained, setup_dataframe):

    data, _ = setup_dataframe
    data = data.drop(columns=["PassengerId", "Cabin"]).dropna().reset_index(
        drop=True)
    data = data.select_dtypes(exclude='object')

    j = Jai(url=URL, auth_key=AUTH_KEY)
    j.header = HEADER_TEST

    df0 = data.iloc[:200, 3:].reset_index().rename(columns={'index': 'myid'})
    df0.index = df0.index.rename('id')
    j.fit(name=name,
          db_type='SelfSupervised',
          data=df0,
          overwrite=True,
          pretrained_bases=[{
              'db_parent': pretrained,
              'id_name': 'myid'
          }])

    j_info = j.info
    dep = j_info.loc[j_info['name'] == 'test_insert_vector_ss',
                     'dependencies'].values[0]
    assert dep == ['test_insert_vector']

    j.delete_database('test_insert_vector_ss')
    j_info = j.info
    assert 'test_insert_vector_ss' not in j_info['name'].values