import numpy as np
import pandas as pd
import pytest

from jai import Jai

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

    j = Jai(safe_mode=safe_mode)

    df0 = data[:400]
    overwrite = False
    if name in j.names:
        j.delete_database(name)

    batch_size = 200
    j.insert_vectors(name=name, data=df0, batch_size=batch_size, overwrite=overwrite)
    j_info = j.info
    assert name in j.names
    assert "Vector" == j_info[j_info["name"] == name]["type"].unique()

    n_ids = len(j.ids(name, mode="complete"))
    assert n_ids == df0.shape[0]

    overwrite = True
    df1 = data.to_numpy()
    j.insert_vectors(name=name, data=df1, batch_size=batch_size, overwrite=overwrite)

    n_ids = len(j.ids(name, mode="complete"))
    assert n_ids == data.shape[0]

    overwrite = False
    with pytest.raises(KeyError) as e:
        j.insert_vectors(
            name=name, data=data, batch_size=batch_size, overwrite=overwrite
        )
        assert (
            e.value.args[0]
            == f"Database 'test_insert_vector' already exists in your environment."
            "Set overwrite=True to overwrite it."
        )


@pytest.mark.parametrize("safe_mode", [True])
@pytest.mark.parametrize("name", [("test_insert_vector")])
def test_append_vectors(safe_mode, name, setup_dataframe):

    data, _ = setup_dataframe
    data = data.drop(columns=["PassengerId", "Cabin"]).dropna().reset_index(drop=True)
    data = data.select_dtypes(exclude="object")

    j = Jai(safe_mode=safe_mode)

    df0 = data[:100].iloc[:, :3]
    j.insert_vectors(name=name, data=df0, overwrite=True)
    j_info = j.info
    length = j_info.loc[j_info["name"] == "test_insert_vector", "size"].values[0]
    assert length == 100

    df1 = data[100:200].iloc[:, :3]
    j.insert_vectors(name=name, data=df1, append=True)
    j_info = j.info
    length = j_info.loc[j_info["name"] == "test_insert_vector", "size"].values[0]
    assert length == 200


@pytest.mark.parametrize("safe_mode", [True])
@pytest.mark.parametrize(
    "name, pretrained", [("test_insert_vector_ss", "test_insert_vector")]
)
def test_pretrained_with_vectors(safe_mode, name, pretrained, setup_dataframe):

    data, _ = setup_dataframe
    data = data.drop(columns=["PassengerId", "Cabin"]).dropna().reset_index(drop=True)
    data = data.select_dtypes(exclude="object")

    j = Jai(safe_mode=safe_mode)

    df0 = data.iloc[:200, 3:].reset_index().rename(columns={"index": "myid"})
    df0.index = df0.index.rename("id")
    j.fit(
        name=name,
        db_type="SelfSupervised",
        data=df0,
        overwrite=True,
        pretrained_bases=[{"db_parent": pretrained, "id_name": "myid"}],
    )

    j_info = j.info
    dep = j_info.loc[j_info["name"] == name, "dependencies"].values[0]
    assert dep == [pretrained]

    j.delete_database(name)
    assert name not in j.names

    j.delete_database(pretrained)
    assert pretrained not in j.names
