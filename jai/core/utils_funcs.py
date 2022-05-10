import numpy as np
import pandas as pd

from .types import PossibleDtypes

__all__ = ["build_name", "data2json", "resolve_db_type"]


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


def data2json(data,
              dtype: PossibleDtypes,
              filter_name: str = None,
              predict: bool = False):
    one_column = f"Data formats accepted for dtype {dtype} are:\n\
    - pd.Series\n\
    - pd.DataFrame with 1 column\n\
    - pd.DataFrame with 2 columns, one must be named `id`\n\
    - pd.DataFrame with 2 columns, one must be named `{filter_name}`\n\
    - pd.DataFrame with 3 columns, two of them must be named `id` and `{filter_name}`"

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
                    return df2json(data)
            elif data.shape[1] == 3:
                if 'id' in data.columns and filter_name in data.columns:
                    return df2json(data.set_index('id'))
            raise ValueError(one_column)
        raise NotImplementedError(
            f"type `{data.__class__.__name__}` is not accepted.\n{one_column}")
    elif dtype in [
            PossibleDtypes.recommendation, PossibleDtypes.recommendation_system
    ]:
        if isinstance(data, pd.DataFrame):
            return df2json(data)
        raise NotImplementedError(
            f"type `{data.__class__.__name__}` is not implemented, use pd.DataFrame instead."
        )
    elif dtype == PossibleDtypes.selfsupervised:
        if isinstance(data, pd.DataFrame):
            count_except_id = (data.columns != 'id').sum()
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
            count_except_id = (data.columns != 'id').sum()
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
            count_except_id = (data.columns != 'id').sum()
            if count_except_id >= 2:
                return df2json(data)

            raise ValueError(
                f"Data must be a DataFrame with at least 2 columns other than `id`. Current column(s):\n{data.columns.tolist()}"
            )

    raise ValueError(f"dtype {dtype} not recognized.")
