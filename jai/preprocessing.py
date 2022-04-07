import pandas as pd
import warnings

__all__ = ["split"]


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
        sort values of the split data. 
    prefix : str, optional
        prefix added to the splitted column names.

    Returns
    -------
    bases : list of pd.DataFrame
        list of dataframes with each base extracted.
    dataframe : pd.DataFrame
        original dataframe with columns replaced by the ids of the correlated base.

    """
    dataframe = dataframe.copy()
    if isinstance(columns, str):
        columns = {columns: None}
    elif isinstance(columns, list):
        columns = {col: None for col in columns}

    na_columns = dataframe.isna().any(0).loc[columns.keys()]
    if na_columns.any():
        warnings.warn(
            f"Empty values will be represented with -1 as id values and cause issues later, we recommend treating them before split.\n\
            Found empty values on the following columns:\n\
            - {'- '.join(na_columns.index[na_columns])}",
            stacklevel=3)

    bases = {}
    for col, sep in columns.items():
        if sep is not None:
            values = dataframe[col].str.split(sep).explode().str.strip()
        else:
            values = dataframe[col]
        ids, uniques = pd.factorize(values, sort=sort)
        dataframe = dataframe.drop(columns=col)
        if sep is not None:
            dataframe[prefix + col] = pd.DataFrame({
                "id": values.index,
                col: ids
            }).groupby("id")[col].agg(lambda x: list(x))
        else:
            dataframe[prefix + col] = ids
        base = pd.DataFrame({col: uniques},
                            index=pd.Index(range(len(uniques)), name="id"))
        bases[col] = base

    return bases, dataframe


def split_recommendation(dataframe,
                         split_config: dict,
                         columns: str,
                         sort: bool = False,
                         prefix: str = "id_"):
    """
    Split data into the 3 datasets for recommendation and also splits columns 
    returning the datasets for pretrained bases and replacing the original column with the
    corresponding index of the new dataframe

    Parameters
    ----------
    dataframe : pd.DataFrame
        Dataframe to be factored.
    split_config : dict
        Dictionary with id names (prefix param will be added to those names) as keys and
        list of columns of those datasets as values. Must have length 2 and no common values.
    columns : str, list of str or dict
        Column to be separated from dataset.
        If column has multiple data, use a dict with the format column name as
        key and separator as value. Use `None` if no separator is needed.
    sort : bool
        sort values of the split data.
    prefix : str
        prefix added to the splitted column names.

    Returns
    -------
    main_bases : list of pd.DataFrame
        original dataframe with columns replaced by the ids of the correlated base.

    pretrained_bases : pd.DataFrame
        list of dataframes with each base extracted.
    """

    pretrained_bases, df_merge = split(dataframe,
                                       columns,
                                       sort=sort,
                                       prefix=prefix)

    main_bases = {}
    for name, split_cols in split_config.items():
        split_cols = [
            prefix + col if col in columns else col for col in split_cols
        ]
        df_out = df_merge.loc[:, split_cols].drop_duplicates().reset_index(
        ).rename(columns={'index': prefix + name})
        df_merge = df_merge.merge(df_out,
                                  left_on=split_cols,
                                  right_on=split_cols).drop(columns=split_cols)
        main_bases[name] = df_out.set_index(prefix + name)

    main_bases["main"] = df_merge
    return main_bases, pretrained_bases


def treat_unix(df_unix_col):
    """
    Transform the type of the unix timestamp column to datetime 
    returning a series that replaces the original 
    column.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Dataframe with only the unix column.
        
    Returns
    -------
    datime_col : column with the type altered to datetime that
        should substitute the unix timestamp column.
    """
    datime_col = pd.to_datetime(df_unix_col, unit="s")

    return datime_col