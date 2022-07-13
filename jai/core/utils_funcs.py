import warnings
from typing import Dict, Optional

import numpy as np
import pandas as pd
import psutil

from ..types.generic import PossibleDtypes

__all__ = ["build_name", "data2json", "resolve_db_type"]


def get_pcores(max_insert_workers: Optional[int] = None):

    if max_insert_workers is None:
        pcores = psutil.cpu_count(logical=False)
    elif not isinstance(max_insert_workers, int):
        raise TypeError(
            f"Variable 'max_insert_workers' must be 'None' or 'int' instance, not {max_insert_workers.__class__.__name__}."
        )
    elif max_insert_workers > 0:
        pcores = max_insert_workers
    else:
        pcores = 1
    return pcores


# Helper function to decide which kind of text model to use
def resolve_db_type(db_type, col):
    if isinstance(db_type, str):
        return db_type
    elif isinstance(db_type, dict) and col in db_type:
        return db_type[col]
    return "TextEdit"


def build_name(name: str, col: str):
    """
    Helper function to build the database names of columns that
    are automatically processed during 'sanity' and 'fill' methods

    Args:
        name (str): database's name
        col (srt): column name

    Returns:
        str: new database name
    """
    origin = name + "_" + col
    return origin.lower().replace("-", "_").replace(" ", "_")


def series2json(data_series):
    data_series = data_series.copy()
    data_series.index.name = "id"
    if data_series.index.duplicated().any():
        raise ValueError("Index must not contain duplicated values.")
    return data_series.reset_index().to_json(orient="records", date_format="iso")


def df2json(dataframe):
    dataframe = dataframe.copy()
    if "id" not in dataframe.columns:
        dataframe.index.name = "id"
        dataframe = dataframe.reset_index()
    if dataframe["id"].duplicated().any():
        raise ValueError("Index must not contain duplicated values.")
    return dataframe.to_json(orient="records", date_format="iso")


def data2json(
    data, dtype: PossibleDtypes, has_filter: bool = False, predict: bool = False
):
    one_column = (
        f"Data formats accepted for dtype {dtype} are:\n"
        "- pd.Series\n"
        "- pd.DataFrame with 1 column\n"
        "- pd.DataFrame with 2 columns, one must be named `id`\n"
        "- pd.DataFrame with 2 columns, one must be the filter column`\n"
        "- pd.DataFrame with 3 columns, two of them must be named `id` andthe filter column"
    )

    if dtype in [
        PossibleDtypes.edit,
        PossibleDtypes.text,
        PossibleDtypes.fasttext,
        PossibleDtypes.image,
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
                if "id" in data.columns:
                    data = data.set_index("id")
                    c = data.columns[0]
                    return series2json(data[c])
                elif has_filter:
                    return df2json(data)
            elif data.shape[1] == 3:
                if "id" in data.columns and has_filter:
                    return df2json(data.set_index("id"))
            raise ValueError(one_column)
        raise NotImplementedError(
            f"type `{data.__class__.__name__}` is not accepted.\n{one_column}"
        )
    elif dtype in [PossibleDtypes.recommendation, PossibleDtypes.recommendation_system]:
        if isinstance(data, pd.DataFrame):
            return df2json(data)
        raise NotImplementedError(
            f"type `{data.__class__.__name__}` is not implemented, use pd.DataFrame instead."
        )
    elif dtype == PossibleDtypes.selfsupervised:
        if isinstance(data, pd.DataFrame):
            count_except_id = (data.columns != "id").sum()
            if count_except_id >= 2:
                return df2json(data)

            raise ValueError(
                f"Data must be a DataFrame with at least 2 columns other than `id`. Current column(s):\n{data.columns.tolist()}"
            )
        raise NotImplementedError(
            f"type `{data.__class__.__name__}` is not implemented, use pd.DataFrame instead."
        )
    elif dtype == PossibleDtypes.supervised:
        if isinstance(data, pd.DataFrame):
            count_except_id = (data.columns != "id").sum()
            if count_except_id >= 2 - predict:
                return df2json(data)

            raise ValueError(
                f"Data must be a DataFrame with at least {2 - predict} column(s) other than `id`. Current column(s):\n{data.columns.tolist()}"
            )
        raise NotImplementedError(
            f"type `{data.__class__.__name__}` is not implemented, use pd.DataFrame instead."
        )
    elif dtype == "Unsupervised":
        raise ValueError(
            f"`Unsupervised` type has been replaced with `{PossibleDtypes.selfsupervised}`."
        )
    elif dtype == PossibleDtypes.vector:
        if isinstance(data, pd.DataFrame):
            count_except_id = (data.columns != "id").sum()
            if count_except_id >= 2:
                return df2json(data)

            raise ValueError(
                f"Data must be a DataFrame with at least 2 columns other than `id`. Current column(s):\n{data.columns.tolist()}"
            )

    raise ValueError(f"dtype {dtype} not recognized.")


def common_items(d1, d2):
    """
    It recursively compares the values of two dictionaries, and returns a new dictionary containing only
    the keys and values that are common to both dictionaries

    https://stackoverflow.com/a/38506628/10168941

    Args:
      d1: The first dictionary to compare.
      d2: The dictionary to compare against.

    Returns:
      A dictionary with the common keys and values.
    """
    result = {}
    for k in d1.keys() & d2.keys():
        v1 = d1[k]
        v2 = d2[k]
        if isinstance(v1, dict) and isinstance(v2, dict):
            result[k] = common_items(v1, v2)
        elif v1 == v2:
            result[k] = v1
        else:
            raise ValueError("values for common keys don't match")
    return result


def common_elements(l1, l2):
    """
    It takes two lists of dictionaries, and returns a list of dictionaries that have the same keys and
    values

    Args:
      l1: the list of dictionaries to be compared
      l2: the list of dictionaries that we want to compare against

    Returns:
      A list of dictionaries that have the same keys and values.
    """
    result = []
    for e1 in l1:
        match = False
        for e2 in l2:
            if e1.items() <= e2.items():
                match = True
                break

        if match:
            result.append(e1)
        else:
            raise ValueError("values for common keys don't match")
    return result


def print_args(response_kwargs, input_kwargs, verbose: int = 1):
    """
    It takes two dictionaries,
    one from the API response and one from the user input, and prints out the
    arguments that were recognized

    Args:
      response_kwargs: the parameters that are returned from the API
      input_kwargs: the parameters that you passed to the function
      verbose (int): If 1, prints out the recognised parameters, if 2,
      prints out everything that is used. Defaults to 1.
    """
    if verbose == 0:
        return

    all_args = verbose > 2

    warn_list = []
    print("\nRecognized fit arguments:")
    for key in input_kwargs.keys():
        value = response_kwargs.get(key, None)
        input = input_kwargs.get(key, None)

        if key == "split" and input is not None:
            value = response_kwargs["hyperparams"]["split"]

        if input is None:
            continue

        if isinstance(input, dict) and isinstance(value, dict):

            intersection = common_items(input, value)
            if not input.keys() == intersection.keys():
                warn_list.append(f"argument: `{key}`; values: ({input} != {value})")

            to_write = value if all_args else input
            m = max([len(s) for s in to_write] + [0])

            to_join = []
            for k, v in to_write.items():
                if isinstance(v, dict):
                    to_join.append(f"  * {k:{m}s}:\n")
                    for _k, _v in v.items():
                        to_join.append(f"    - {_k}: {_v}\n")
                else:
                    to_join.append(f"  * {k:{m}s}: {v}\n")
            value = "\n" + "".join(to_join)

        elif isinstance(input, list) and isinstance(value, list):
            intersection = common_elements(input, value)
            if input != intersection:
                warn_list.append(f"argument: `{key}`; values: ({input} != {value})")

            to_write = value if all_args else input

            to_join = []
            for v in to_write:
                if isinstance(v, dict):
                    first = True
                    for _k, _v in v.items():
                        if first:
                            to_join.append(f"  * {_k}: {_v}\n")
                            first = False
                        else:
                            to_join.append(f"    {_k}: {_v}\n")
                else:
                    to_join.append(f"  * {v}\n")

            value = "\n" + "".join(to_join)

        else:
            if input != value:
                warn_list.append(f"argument: `{key}`; values: ({input} != {value})")
            value = f"{value}\n"

        if value is not None:
            print(f"- {key}: {value}", end="")

    if len(warn_list):
        warn_str = "\n".join(warn_list)
        warnings.warn(
            "Values from input and from API response differ.\n" + warn_str, stacklevel=3
        )


def check_filters(data: pd.DataFrame, features: Dict[str, Dict]):
    """
    It returns `True` if any of the columns in the dataframe have a `dtype` of `filter` defined
    on features.

    Args:
      data (pd.DataFrame): the dataframe you want to filter
      features (Dict[str, Dict]): A dictionary of features, where the key is the column name and the
    value is a dictionary of feature properties.

    Returns:
      A boolean value.
    """
    return any(
        [
            feat["dtype"] == "filter"
            for col, feat in features.items()
            if col in data.columns
        ]
    )
