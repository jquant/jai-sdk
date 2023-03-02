import numpy as np
import pandas as pd
import pytest

from jai.task.vectors import Vectors

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


@pytest.mark.parametrize("safe_mode", [True])
@pytest.mark.parametrize("name", [("test_insert_vector")])
def test_insert_vectors(safe_mode, name, setup_dataframe):
    data, _ = setup_dataframe
    data = (
        data.drop(columns="Cabin")
        .dropna()
        .rename(columns={"PassengerId": "id"})
        .set_index("id")
    )
    data = data.select_dtypes(exclude="object")

    vectors = Vectors(name=name, safe_mode=safe_mode)

    df0 = data[:400]
    overwrite = False

    batch_size = 200
    res0 = vectors.insert_vectors(data=df0, batch_size=batch_size, overwrite=overwrite)
    assert len(res0) == 2
    assert res0[0] == res0[1]
    assert res0[0] == {
        "collection_name": "test_insert_vector",
        "vector_length": 200,
        "vector_dimension": 6,
        "message": "test_insert_vector vector insertion has finished.",
    }

    overwrite = True
    df1 = data.to_numpy()
    res1 = vectors.insert_vectors(data=df1, batch_size=batch_size, overwrite=overwrite)
    assert len(res1) == 4
    assert (res1[0] == res1[1]) and (res1[1] == res1[2]) and (res1[2] != res1[3])
    assert dict(res1[0]) == {
        "collection_name": "test_insert_vector",
        "vector_length": 200,
        "vector_dimension": 6,
        "message": "test_insert_vector vector insertion has finished.",
    }
    assert dict(res1[3]) == {
        "collection_name": "test_insert_vector",
        "vector_length": 112,
        "vector_dimension": 6,
        "message": "test_insert_vector vector insertion has finished.",
    }

    overwrite = False
    with pytest.raises(KeyError) as e:
        vectors.insert_vectors(data=data, batch_size=batch_size, overwrite=overwrite)
        assert (
            e.value.args[0]
            == f"Database 'test_insert_vector' already exists in your environment."
            "Set overwrite=True to overwrite it."
        )

    vectors.delete_database()


@pytest.mark.parametrize("safe_mode", [True])
@pytest.mark.parametrize("name", [("test_insert_vector")])
def test_append_vectors(safe_mode, name, setup_dataframe):
    data, _ = setup_dataframe
    data = data.drop(columns=["PassengerId", "Cabin"]).dropna().reset_index(drop=True)
    data = data.select_dtypes(exclude="object")

    vectors = Vectors(name=name, safe_mode=safe_mode)

    df0 = data[:100].iloc[:, :3]
    res0 = vectors.insert_vectors(data=df0, overwrite=True)

    assert len(res0) == 1
    assert res0[0] == {
        "collection_name": "test_insert_vector",
        "vector_length": 100,
        "vector_dimension": 3,
        "message": "test_insert_vector vector insertion has finished.",
    }

    df1 = data[100:200].iloc[:, :3]
    res1 = vectors.insert_vectors(data=df1, append=True)

    assert len(res1) == 1
    assert res1[0] == {
        "collection_name": "test_insert_vector",
        "vector_length": 100,
        "vector_dimension": 3,
        "message": "test_insert_vector vector insertion has finished.",
    }

    vectors.delete_database()
