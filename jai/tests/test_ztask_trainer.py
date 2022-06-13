import numpy as np
import pandas as pd
import pytest

from jai.task.trainer import Trainer


@pytest.fixture(scope="session")
def setup_dataframe():
    TITANIC_TRAIN = "https://raw.githubusercontent.com/rebeccabilbro/titanic/master/data/train.csv"
    TITANIC_TEST = "https://raw.githubusercontent.com/rebeccabilbro/titanic/master/data/test.csv"

    train = pd.read_csv(TITANIC_TRAIN)
    test = pd.read_csv(TITANIC_TEST)
    return train, test


def test__check_pretrained_bases(setup_dataframe):
    pass