import pandas as pd

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
    columns : str, list of str or dict, optional
        Column to be separated from dataset.
        If column has multiple data, use a dict with the format column name as
        key and separator as value. Use `None` if no separator is needed.
    sort : bool
        sort values of the split data.
    prefix : str
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
