# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 21:39:24 2021

@author: Kazu
"""

import pandas as pd
import numpy as np
from .jai import Jai

__all__ = ['match', 'resolution', 'fill', 'sanity']


def match(data1, data2, auth_key, name=None):
    """
    Experimental

    Parameters
    ----------
    data1, data2 : text
        data to be matched.
    auth_key : TYPE
        Auth key for mycelia.
    name : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    dict
        each key is the id from data2 and the value is a list of ids from data1
        that match.

    Example
    -------
    >>> import pandas as pd
    >>> from jai.applications import match
    >>> from jai.auxiliar_funcs.utils_funcs import process_similar
    >>>
    >>> results = match(data1, data2, AUTH_KEY, name)
    >>> processed = process_similar(results, return_self=True)
    >>> pd.DataFrame(processed).sort_values('query_id')
    >>> # query_id is from data2 and id is from data 1
             query_id           id     distance
       0            1            2         0.11
       1            2            1         0.11
       2            3          NaN          NaN
       3            4          NaN          NaN
       4            5            5         0.15
    """
    top_k = 20
    j = Jai(auth_key)
    nt = np.clip(np.round(len(data1)/10, -3), 1000, 10000)
    if name is None:
        name = j.generate_name(20, prefix='sdk_', suffix='')
    print(f"name: {name}")
    if name not in j.names:
        j.setup(name, data1, batch_size=10000, db_type='TextEdit',
                hyperparams={"nt": nt})
        j.wait_setup(name, 20)
        j.delete_raw_data(name)
    return j.similar(name, data2, top_k=top_k, batch_size=10000)


def resolution(data, auth_key, name=None):
    """
    Experimental

    Parameters
    ----------
    data : text
        data to find duplicates.
    auth_key : TYPE
        DESCRIPTION.
    name : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    dict
        each key is the id and the value is a list of ids that are duplicates.

    Example
    -------
    >>> import pandas as pd
    >>> from jai.applications import resolution
    >>> from jai.auxiliar_funcs.utils_funcs import process_similar
    >>>
    >>> results = resolution(data, AUTH_KEY, name)
    >>> processed = process_similar(results, return_self=True)
    >>> pd.DataFrame(processed).sort_values('query_id')
             query_id           id     distance
       0            1            2         0.11
       1            2            1         0.11
       2            3          NaN          NaN
       3            4            5         0.15
    """
    top_k = 20
    j = Jai(auth_key)
    nt = np.clip(np.round(len(data)/10, -3), 1000, 10000)
    if name is None:
        name = j.generate_name(20, prefix='sdk_', suffix='')
    print(f"name: {name}")
    if name not in j.names:
        j.setup(name, data, batch_size=10000, db_type='TextEdit',
                hyperparams={"nt": nt})
        j.wait_setup(name, 20)
        j.delete_raw_data(name)
    return j.similar(name, data.index, top_k=top_k, batch_size=10000)


def fill(data, column:str, auth_key, name=None, **kwargs):
    """
    Experimental

    Parameters
    ----------
    data : pd.DataFrame
        data to fill NaN.
    column : str
        name of the column to be filled.
    auth_key : TYPE
        DESCRIPTION.
    name : TYPE, optional
        DESCRIPTION. The default is None.
    **kwargs : TYPE
        Extra args for supervised model.

    Returns
    -------
    list of dicts
        List of dicts with possible filling values for each id with column NaN.

    Example
    -------
    >>> import pandas as pd
    >>> from jai.applications import fill
    >>> from jai.auxiliar_funcs.utils_funcs import process_predict
    >>>
    >>> results = fill(data, COL_TO_FILL, AUTH_KEY, name)
    >>> processed = process_similar(results)
    >>> pd.DataFrame(processed).sort_values('id')
              id   sanity_prediction    confidence_level (%)
       0       1             value_1                    70.9
       1       4             value_1                    67.3
       2       7             value_1                    80.2
    """
    cat_threshold = 512
    data = data.copy()

    j = Jai(auth_key)
    if name is None:
        name = j.generate_name(20, prefix='sdk_', suffix='')

    cat = data.select_dtypes(exclude='number')
    pre = cat.columns[cat.nunique() > cat_threshold].tolist()
    prep_bases = []
    for col in pre:
        id_col = 'id_' + col
        values, inverse = np.unique(data[col], return_inverse=True)
        data[id_col] = inverse
        origin = embedding(values, auth_key, name=name + '_' + col)
        prep_bases.append({"id_name": id_col, "db_parent": origin})
    data = data.drop(columns=pre)

    vals = data[column].value_counts() < 2
    if vals.sum() > 0:
        eliminate = vals[vals].index.tolist()
        print(f"values {eliminate} from column {column} were removed for having less than 2 examples.")
        data.loc[data[column].isin(eliminate), column] = None

    print(f"name: {name}")
    label = {"task": "metric_classification",
             "label_name": column}
    split = {"type": 'stratified',
             "split_column": column,
             "test_size": .2}
    mycelia_bases = kwargs.get("mycelia_bases", [])
    mycelia_bases.extend(prep_bases)

    mask = data[column].isna()
    train = data.loc[~mask].copy()
    test = data.loc[mask].drop(columns=[column])

    if name not in j.names:
        j.setup(name, train, batch_size=10000, db_type='Supervised',
                hyperparams={"learning_rate": 0.001}, label=label, split=split,
                **kwargs)
        j.wait_setup(name, 20)
        j.delete_raw_data(name)

    return j.predict(name, test, predict_proba=True, batch_size=10000)


def sanity(data, auth_key, data_validate=None, columns_ref: list=None,
           name:str=None, frac:float= .1, random_seed=42, **kwargs):
    """
    Experimental

    Parameters
    ----------
    data : pd.DataFrame
        Data reference of sound data.
    auth_key : TYPE
        DESCRIPTION.
    data_validate : TYPE, optional
        Data to be checked if is valid or not. The default is None.
    columns_ref : list, optional
        Columns that can have inconsistencies. As default we use all non numeric
        columns. The default is None.
    name : str, optional
        DESCRIPTION. The default is None.
    frac : float, optional
        DESCRIPTION. The default is .1.
    random_seed : TYPE, optional
        DESCRIPTION. The default is 42.
    **kwargs : TYPE
        DESCRIPTION.

    Raises
    ------
    ValueError
        DESCRIPTION.

    Returns
    -------
    list of dicts
        Result of data is valid or not.

    Example
    -------
    >>> import pandas as pd
    >>> from jai.applications import sanity
    >>> from jai.auxiliar_funcs.utils_funcs import process_predict
    >>>
    >>> results = sanity(data, AUTH_KEY, 'sdk_4ddfcb1c1100de84')
    >>> processed = process_predict(results)
    >>> pd.DataFrame(processed).sort_values('id')
              id   sanity_prediction    confidence_level (%)
       0       1               Valid                    70.9
       1       4             Invalid                    67.3
       2       7             Invalid                    80.6
       3      13               Valid                    74.2
    """
    cat_threshold = 512
    np.random.seed(random_seed)
    target = ['is_valid', 'is_sound', 'valid', 'sanity', 'target']
    mask_target = np.logical_not(np.isin(np.array(target), data.columns))
    if not mask_target.any():
        raise ValueError(f"at least one of the values ({target}) should not be in columns.")
    target = np.array(target)[mask_target][0]
    data = data.copy()
    data_validate = data_validate.copy()

    j = Jai(auth_key)
    if name is None:
        name = j.generate_name(20, prefix='sdk_', suffix='')

    cat = data.select_dtypes(exclude='number')
    pre = cat.columns[cat.nunique() > cat_threshold].tolist()
    if columns_ref is None:
        columns_ref = cat.columns.tolist()

    n = len(data)
    prep_bases = []
    for col in pre:
        id_col = 'id_' + col
        emb = data[col].tolist() + data_validate[col].tolist()
        values, inverse = np.unique(emb, return_inverse=True)
        data[id_col] = inverse[:n]
        data_validate[id_col] = inverse[n:]
        origin = embedding(values, auth_key, name=name + '_' + col)
        prep_bases.append({"id_name": id_col, "db_parent": origin})
        columns_ref.remove(col)
        columns_ref.append(id_col)
    data = data.drop(columns=pre)
    data_validate = data_validate.drop(columns=pre)

    print(f"name: {name}")
    label = {"task": "metric_classification",
             "label_name": target}
    split = {"type": 'stratified',
             "split_column": target,
             "test_size": .2}
    mycelia_bases = kwargs.get("mycelia_bases", [])
    mycelia_bases.extend(prep_bases)

    def change(options, original):
        return np.random.choice(options[options!=original])

    # get a sample of the data and shuffle it
    sample = []
    for c in columns_ref:
        s = data.sample(frac=frac)
        uniques = s[c].unique()
        s[c] = [change(uniques, v) for v in s[c]]
        sample.append(s)
    sample = pd.concat(sample)

    # set target column values
    sample[target] = "Invalid"

    # set index of samples with different values as data
    idx = np.arange(len(data)+len(sample))
    mask_idx = np.logical_not(np.isin(idx, data.index))
    sample.index = idx[mask_idx][:len(sample)]

    data[target] = "Valid"
    train = pd.concat([data, sample])
    if data_validate is None:
        test = data.copy()
    else:
        test = data_validate.copy()


    if name not in j.names:
        j.setup(name, train, batch_size=10000, db_type='Supervised',
                  hyperparams={"learning_rate": 0.001}, label=label, split=split,
                  **kwargs)
        j.wait_setup(name, 20)
        j.delete_raw_data(name)

    results = j.predict(name, test, predict_proba=True, batch_size=10000)
    return results


def embedding(data, auth_key, name=None, db_type='FastText'):
    """
    Experimental
    Quick embedding for high numbers of categories in columns.

    Parameters
    ----------
    data : TYPE
        DESCRIPTION.
    auth_key : TYPE
        DESCRIPTION.
    name : TYPE, optional
        DESCRIPTION. The default is None.
    db_type : TYPE, optional
        DESCRIPTION. The default is 'FastText'.

    Returns
    -------
    name : str
        name of the base where the data was embedded.

    """
    j = Jai(auth_key)
    if name is None:
        name = j.generate_name(20, prefix='sdk_', suffix='')
    else:
        name = name.lower().replace('-', '_').replace(' ', '_')[:35]

    if db_type == "FastText":
        hyperparams={"minn": 0, "maxn": 0}
    elif db_type == "TextEdit":
        nt = np.clip(np.round(len(data)/10, -3), 1000, 10000)
        hyperparams={"nt": nt}
    else:
        hyperparams=None
    print(f"name: {name}")
    if name not in j.names:
        j.setup(name, data, batch_size=10000, db_type=db_type,
                hyperparams=hyperparams)
        j.wait_setup(name, 10)
        j.delete_raw_data(name)
    return name