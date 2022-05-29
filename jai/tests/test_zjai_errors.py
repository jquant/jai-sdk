import json

import numpy as np
import pandas as pd
import pytest
from decouple import config

from jai import Jai
import os
from copy import deepcopy

from jai import Jai
from jai.core.validations import check_dtype_and_clean, check_name_lengths


def test_generate_error():
    j = Jai()
    with pytest.raises(ValueError):
        j.generate_name(8, "prefix", "suffix")


def test_insert_vector_json_exception():
    j = Jai()

    db = np.array([[1], [2]])
    with pytest.raises(ValueError) as e:
        j.insert_vectors(name="test", data=db, overwrite=True)
    assert e.value.args[
        0] == f"Data must be a DataFrame with at least 2 columns other than `id`. Current column(s):\n[0]"

    db = pd.DataFrame({'a': [1, 'a'], 'b': [1, 'c'], 'c': [1, np.nan]})
    with pytest.raises(ValueError) as e:
        j.insert_vectors(name="test", data=db, overwrite=True)
    assert e.value.args[
        0] == f"Columns ['a', 'b'] contains values types different from numeric."


def test_check_name_lengths_exception():
    # we need to use a valid URL for this one
    j = Jai()
    with pytest.raises(ValueError):
        check_name_lengths(name="test", cols=[j.generate_name(length=35)])


@pytest.mark.parametrize("name, batch_size, db_type",
                         [("test", 1024, "SelfSupervised")])
def test_check_ids_consistency_exception(name, batch_size, db_type):
    # we need to use a valid URL for this one
    j = Jai()

    # mock data
    r = 1100
    data = pd.DataFrame({
        "category": [str(i) for i in range(r)],
        "number": [i for i in range(r)]
    })

    # insert it
    j._insert_data(data=data,
                   name=name,
                   batch_size=batch_size,
                   db_type=db_type)

    # intentionally break it
    with pytest.raises(Exception):
        j._check_ids_consistency(name=name, data=data.iloc[:r - 5])


@pytest.mark.parametrize("name", ["invalid_test"])
def test_delete_tree(name):
    # we need to use a valid URL for this one
    j = Jai()
    with pytest.raises(IndexError):
        j._delete_tree(name)


@pytest.mark.parametrize('name', ['test_resolution'])
def test_filters(name):
    j = Jai()
    with pytest.raises(ValueError):
        j.filters(name)


@pytest.mark.parametrize("name, batch_size, db_type, max_insert_workers",
                         [("test", 1024, "SelfSupervised", "1")])
def test_max_insert_workers(name, batch_size, db_type, max_insert_workers):
    j = Jai()
    with pytest.raises(TypeError):
        j._insert_data(data={},
                       name=name,
                       batch_size=batch_size,
                       db_type=db_type,
                       max_insert_workers=max_insert_workers)
