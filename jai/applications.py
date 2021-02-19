# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 21:39:24 2021

@author: Kazu
"""

import pandas as pd
import numpy as np
from .jai import Jai

from .auxiliar_funcs.utils_funcs import process_similar


def match(data1, data2, auth_key, name=None, threshold=None, top_k=20):
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
    threshold : TYPE, optional
        DESCRIPTION. The default is None.
    top_k : TYPE, optional
        DESCRIPTION. The default is 20.

    Returns
    -------
    dict
        each key is the id from data2 and the value is a list of ids from data1
        that match.

    """

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
    results = j.similar(name, data2, top_k=top_k, batch_size=10000)
    return process_similar(results, threshold=threshold, return_self=True)



def resolution(data, auth_key, name=None, threshold=None, top_k=20):
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
    threshold : TYPE, optional
        DESCRIPTION. The default is None.
    top_k : TYPE, optional
        DESCRIPTION. The default is 20.

    Returns
    -------
    dict
        each key is the id and the value is a list of ids that are duplicates.

    """

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
    results = j.similar(name, data.index, top_k=top_k, batch_size=10000)
    return process_similar(results, threshold=threshold, return_self=False)


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
        DESCRIPTION.

    Returns
    -------
    list of dicts
        List of dicts with possible filling values for each id with column NaN.

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
        eliminate = vals[vals].index
        print(f"values {eliminate} from column {column} were removed for having less than 2 examples.")
        data.loc[data[column].isin(), column] = None

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

    """
    cat_threshold = 512
    np.random.seed(random_seed)
    target = ['is_valid', 'is_sound', 'valid', 'sanity', 'target']
    mask_target = np.logical_not(np.isin(np.array(target), data.columns))
    if mask_target.all():
        raise ValueError(f"at least one of the values ({target}) should not be in columns.")
    target = np.array(target)[mask_target][0]
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
    if columns_ref is None:
        columns_ref = cat.columns
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
        j.wait_setup(name, 5)
        j.delete_raw_data(name)
    return name