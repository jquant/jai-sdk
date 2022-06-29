import warnings
from typing import Dict, List, Union

import pandas as pd

__all__ = ["split", "split_recommendation"]


def split(dataframe, columns, sort: bool = False, prefix: str = "id_"):
    """
    Split columns from dataframe returning a dataframe with the unique values
    for each specified column and replacing the original column with the
    corresponding index of the new dataframe

    Parameters
    ----------
    dataframe : pd.DataFrame
        Dataframe to be factored.
    columns : str, list of str or dict
        Column to be separated from dataset.
        If column has multiple data, use a dict with the format column name as
        key and separator as value. Use `None` if no separator is needed.
    sort : bool, optional
        Sort values of the split data.
    prefix : str, optional
        prefix added to the splitted column names.

    Returns
    -------
    bases : list of pd.DataFrame
        list of dataframes with each base extracted.
    dataframe : pd.DataFrame
        original dataframe with columns replaced by the ids of the correlated
        base.

    Example
    -------
    >>> from jai.utilities import split
    ...
    >>> split_bases, main_base = split(df, ["split_column"])
    """
    dataframe = dataframe.copy()
    if isinstance(columns, str):
        columns = {columns: None}
    elif isinstance(columns, list):
        columns = {col: None for col in columns}

    na_columns = dataframe.isna().any(0).loc[columns.keys()]
    if na_columns.any():
        warnings.warn(
            f"Empty values will be represented with -1 as id values and cause\
            issues later, we recommend treating them before split.\n\
            Found empty values on the following columns:\n\
            - {'- '.join(na_columns.index[na_columns])}",
            stacklevel=3,
        )

    bases = {}
    for col, sep in columns.items():
        if sep is not None:
            values = dataframe[col].str.split(sep).explode().str.strip()
        else:
            values = dataframe[col]
        ids, uniques = pd.factorize(values, sort=sort)
        dataframe = dataframe.drop(columns=col)
        if sep is not None:
            dataframe[prefix + col] = (
                pd.DataFrame({"id": values.index, col: ids})
                .groupby("id")[col]
                .agg(lambda x: list(x))
            )
        else:
            dataframe[prefix + col] = ids
        base = pd.DataFrame(
            {col: uniques}, index=pd.Index(range(len(uniques)), name="id")
        )
        bases[col] = base

    return bases, dataframe


def split_recommendation(
    dataframe,
    split_config: Dict[str, List[str]],
    columns: str,
    as_index: Union[bool, Dict[str, str]] = False,
    sort: bool = False,
    prefix: str = "id_",
):
    """
    Split data into the 3 datasets for recommendation and also splits columns
    returning the datasets for pretrained bases and replacing the original
    column with the corresponding index of the new dataframe

    Parameters
    ----------
    dataframe : pd.DataFrame
        Dataframe to be factored.
    split_config : Dict[str, List[str]]
        Dictionary with **length 2**.
        - keys: db_names for each of the child Recommendation databases created
        on Recommendation System's setup.
        - values: list of columns of those databases.
    columns : str, list of str or dict
        Column to be separated from dataset.
        If column has multiple data, use a dict with the format column name as
        key and separator as value. Use `None` if no separator is needed.
    as_index : False or Dict[str, str]
        Dictionary with **length 2**:
        - keys: database name.
        - values: column name to be used as id for that database
    sort : bool
        sort values of the split data. See `split` function.
    prefix : str
        Prefix added to the splitted column names. See `split` function.
        Also used as prefix for de id columns of the child Recommendation databases.


    Returns
    -------
    main_bases : list of pd.DataFrame
        original dataframe with columns replaced by the ids of the correlated
        base.

    pretrained_bases : pd.DataFrame
        list of dataframes with each base extracted.

    Example
    -------
    >>> from jai.utilities import split
    ...
    >>> processed = predict2df(results)
    >>> pd.DataFrame(processed)
    """
    pretrained_bases, df_split = split(dataframe, columns, sort=sort, prefix=prefix)

    main_bases = {}
    for name, split_cols in split_config.items():
        split_cols = [prefix + col if col in columns else col for col in split_cols]
        df_out = df_split.loc[:, split_cols].drop_duplicates()
        if not as_index:
            df_out = df_out.reset_index().rename(columns={"index": prefix + name})
        else:
            df_out[prefix + name] = df_out[as_index[name]].copy()
        df_split = df_split.merge(df_out, left_on=split_cols, right_on=split_cols).drop(
            columns=split_cols
        )

        main_bases[name] = df_out.set_index(prefix + name)

    main_bases["main"] = df_split
    return main_bases, pretrained_bases
