import json
import re
import pandas as pd
import numpy as np

from typing import List
from pathlib import Path
from operator import itemgetter
from functools import cmp_to_key
from .classes import FieldName, PossibleDtypes

__all__ = ["data2json", "pbar_steps"]


def get_status_json(file_path="pbar_status.json"):
    pbar_status_path = Path(file_path)
    with open(pbar_status_path, 'r') as f:
        status_dict = json.load(f)
    return status_dict


def compare_regex(setup_task: str):
    return re.findall('\[(.*?)\]', setup_task)[0]


def pbar_steps(status: List = None, step: int = 0):
    PBAR_STATUS_PATH = Path(
        __file__).parent.parent / "auxiliar/pbar_status.json"
    setup_task = status['Description']

    try:
        db_type = compare_regex(setup_task)
        possible_tasks = get_status_json(PBAR_STATUS_PATH)[db_type]
        for index, task in enumerate(possible_tasks):
            pattern = re.compile(task)
            is_my_task = pattern.search(setup_task)
            if is_my_task:
                return index + 1, len(possible_tasks)
        return step, None
    except:
        return step, None


def series2json(data_series):
    data_series = data_series.copy()
    data_series.index.name = 'id'
    if data_series.index.duplicated().any():
        raise ValueError("Index must not contain duplicated values.")
    return data_series.reset_index().to_json(orient='records',
                                             date_format="iso")


def df2json(dataframe):
    dataframe = dataframe.copy()
    if 'id' not in dataframe.columns:
        dataframe.index.name = 'id'
        dataframe = dataframe.reset_index()
    if dataframe['id'].duplicated().any():
        raise ValueError("Index must not contain duplicated values.")
    return dataframe.to_json(orient='records', date_format="iso")


def data2json(data, dtype, filter_name=None, predict=False):
    if dtype in [
            PossibleDtypes.edit, PossibleDtypes.text, PossibleDtypes.fasttext,
            PossibleDtypes.image
    ]:
        if isinstance(data, (set, list, tuple, np.ndarray)):
            raise TypeError(
                "dtypes `set`, `list`, `tuple`, `np.ndarray` have been deprecated. Use pd.Series instead."
            )
        elif isinstance(data, pd.Series):
            return series2json(data)
        elif isinstance(data, pd.DataFrame):
            if data.shape[1] == 1:
                c = data.columns[0]
                return series2json(data[c])
            elif data.shape[1] == 2:
                if 'id' in data.columns:
                    data = data.set_index('id')
                    c = data.columns[0]
                    return series2json(data[c])
                elif filter_name in data.columns:
                    cols = data.columns.tolist()
                    cols.remove(filter_name)
                    return df2json(data)
                else:
                    raise ValueError(
                        "If data has 2 columns, one must be named 'id'.")
            elif data.shape[1] == 3:
                if 'id' in data.columns and filter_name in data.columns:
                    data = data.set_index('id')
                    cols = data.columns.tolist()
                    cols.remove(filter_name)
                    return df2json(data)
                else:
                    raise ValueError(
                        "If data has 2 columns, one must be named 'id'.")
            else:
                raise ValueError(
                    "Data must be a DataFrame with 1 column or 2 columns with one named 'id'."
                )
        else:
            raise NotImplementedError(
                f"type `{data.__class__.__name__}` is not accepted.")
    elif dtype == PossibleDtypes.selfsupervised:
        if isinstance(data, pd.DataFrame):
            if (data.columns != 'id').sum() >= 2:
                return df2json(data)
            else:
                raise ValueError(
                    f"Data must be a DataFrame with at least 2 columns other than 'id'. Current column(s):\n{data.columns.tolist()}"
                )
        else:
            raise NotImplementedError(
                f"type `{data.__class__.__name__}` is not implemented.")
    elif dtype == PossibleDtypes.supervised:
        if isinstance(data, pd.DataFrame):
            if ((data.columns != 'id').sum() >= 2 and not predict) or (
                (data.columns != 'id').sum() >= 1 and predict):
                return df2json(data)
            else:
                raise ValueError(
                    f"Data must be a DataFrame with at least {2 - predict} column(s) other than 'id'. Current column(s):\n{data.columns.tolist()}"
                )
        else:
            raise NotImplementedError(
                f"type `{data.__class__.__name__}` is not implemented.")
    elif dtype == "Unsupervised":
        raise ValueError(
            f"'Unsupervised' type has been replaced with {PossibleDtypes.selfsupervised} since version 0.6.0"
        )
    else:
        raise ValueError(f"dtype {dtype} not recognized.")


def cmp(x, y):
    """
    Replacement for built-in function cmp that was removed in Python 3

    Compare the two objects x and y and return an integer according to
    the outcome. The return value is negative if x < y, zero if x == y
    and strictly positive if x > y.

    https://portingguide.readthedocs.io/en/latest/comparisons.html#the-cmp-function
    """

    return (x > y) - (x < y)


def multikeysort(items, columns):
    """
    Sort a list of dictionaries.

    https://stackoverflow.com/a/1144405
    https://stackoverflow.com/a/73050

    Parameters
    ----------
    items : list of dictionaries
        list of dictionaries to be sorted.
    columns : list of strings
        list of key names to be sorted on the order of the sorting. add '-' at
        the start of the name if it should be sorted from high to low.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    comparers = [((itemgetter(col[1:].strip()), -1) if col.startswith('-') else
                  (itemgetter(col.strip()), 1)) for col in columns]

    def comparer(left, right):
        comparer_iter = (cmp(fn(left), fn(right)) * mult
                         for fn, mult in comparers)
        return next((result for result in comparer_iter if result), 0)

    return sorted(items, key=cmp_to_key(comparer))
