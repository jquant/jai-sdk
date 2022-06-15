import numpy as np
import pandas as pd
import pytest

from jai.task.trainer import Trainer
import matplotlib.pyplot as plt


class MockResponse():
    @staticmethod
    def _setup():
        return {
            'Task': 'Training Model',
            'Status': 'Job Created',
            'Description': 'Check status after some time!',
            'kwargs': {
                'db_type': '"SelfSupervised"'
            }
        }

    @staticmethod
    def _consistency():
        return True


@pytest.fixture(scope="session")
def setup_dataframe():
    TITANIC_TRAIN = "https://raw.githubusercontent.com/rebeccabilbro/titanic/master/data/train.csv"
    TITANIC_TEST = "https://raw.githubusercontent.com/rebeccabilbro/titanic/master/data/test.csv"

    train = pd.read_csv(TITANIC_TRAIN)
    test = pd.read_csv(TITANIC_TEST)
    return train, test


@pytest.mark.parametrize("safe_mode", [True])
@pytest.mark.parametrize("pname,pdb_type",
                         [('titanic_names', 'SelfSupervised')])
@pytest.mark.parametrize("name", ['titanic_self'])
def test__check_pretrained_bases(setup_dataframe, safe_mode, pname, pdb_type,
                                 name):
    data, _ = setup_dataframe
    cols = ['Name', 'Age', 'Sex']

    df_name = data[cols]
    parent_trainer = Trainer(name=pname, safe_mode=safe_mode)
    parent_trainer.set_params(db_type=pdb_type)
    parent_trainer.fit(df_name, overwrite=True)

    df_titanic = data.drop(columns=cols)
    trainer = Trainer(name=name, safe_mode=safe_mode)
    pretrained_bases = [{'db_parent': pname, 'id_name': 'PassengerId'}]
    res = trainer._check_pretrained_bases(df_titanic, pretrained_bases)
    assert res == list(range(df_titanic.shape[0]))

    df_dict = {'data': df_titanic}
    res = trainer._check_pretrained_bases(df_dict, pretrained_bases)
    assert res == list(range(df_titanic.shape[0]))

    parent_trainer.delete_database()


@pytest.mark.parametrize("name,safe_mode", [('test_report', True)])
def test_report(monkeypatch, setup_dataframe, name, safe_mode):
    data, _ = setup_dataframe

    trainer = Trainer(name, safe_mode=safe_mode)
    trainer.set_params('SelfSupervised')
    trainer.fit(data, overwrite=True)

    monkeypatch.setattr(plt, "show", lambda: None)

    trainer.report(verbose=2)

    res = trainer.report(return_report=True)
    assert list(res.keys()) == [
        'Auto scale batch size', 'Auto lr finder', 'Model Training',
        'Metrics Train', 'Metrics Validation', 'Model Evaluation',
        'Optimal Thresholds', 'Baseline Model', 'Loading from checkpoint'
    ]

    trainer.delete_database()


@pytest.mark.parametrize("name,safe_mode", [('mock_db', True)])
def test_dict_data(setup_dataframe, monkeypatch, name, safe_mode):
    data, _ = setup_dataframe
    data = {'data': data}

    trainer = Trainer(name, safe_mode=safe_mode)
    trainer.set_params(db_type='SelfSupervised')

    def mock__setup(*args, **kwargs):
        return MockResponse._setup()

    def mock__check_ids_consistency(*args, **kwargs):
        return MockResponse._consistency()

    monkeypatch.setattr(trainer, "_setup", mock__setup)
    monkeypatch.setattr(trainer, "_check_ids_consistency",
                        mock__check_ids_consistency)

    insert_res, setup_res = trainer.fit(data, frequency_seconds=0)
    assert setup_res == MockResponse._setup()
    assert len(insert_res) == 1
    assert dict(insert_res[0]) == {
        'Task': 'Adding new data for tabular setup',
        'Status': 'Completed',
        'Description': 'Insertion completed.',
        'Interrupted': False
    }


@pytest.mark.parametrize("name,safe_mode", [("test_recommendation", True)])
def test_recommendation(name, safe_mode):
    mock_db = pd.DataFrame({
        "User": [0, 1, 2, 0, 1, 2, 1, 1, 0, 2],
        "Item": [2, 3, 1, 5, 1, 2, 4, 3, 2, 1]
    })

    mock_users = pd.DataFrame({"User": [1, 2, 3], "id": [0, 1, 2]})
    mock_items = pd.DataFrame({
        "id": [2, 3, 1, 4, 5],
        "Colour": ['black', 'white', 'green', 'yellow', 'blue']
    })
    data = {'users': mock_users, 'items': mock_items, 'main': mock_db}

    trainer = Trainer(name=name, safe_mode=safe_mode)
    trainer.set_params(db_type="RecommendationSystem",
                       pretrained_bases=[{
                           "id_name": "User",
                           "db_parent": "users"
                       }, {
                           "id_name": "Item",
                           "db_parent": "items"
                       }])
    trainer.fit(data=data, overwrite=True)


def test_wrong_data_insertion():
    data = [1, 2, 3]

    trainer = Trainer(name='wrong_data')
    trainer.set_params(db_type='Text')
    with pytest.raises(ValueError):
        trainer.fit(data)