# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 21:39:24 2021

@author: Kazu
"""


import numpy as np
from jai import Jai

from auxiliar_funcs.utils_funcs import process_similar


def match(data1, data2, auth_key, name=None, threshold = None, top_k = 10):

    jai = Jai(auth_key)
    nt = np.clip(np.round(len(data1)/10, -3), 1000, 10000)
    if name is None:
        name = jai.generate_name(20, prefix='sdk_', suffix='')
    print(f"name: {name}")
    if name not in jai.names:
        jai.setup(name, data1, batch_size=10000, db_type='TextEdit',
                  hyperparams={"nt": nt})
        jai.wait_setup(name, 20)
        jai.delete_raw_data(name)
    results = jai.similar(name, data2, top_k=top_k, batch_size=10000)
    return process_similar(results, threshold=threshold, return_self=True)



def resolution(data, auth_key, name=None, threshold = None, top_k = 10):

    jai = Jai(auth_key)
    nt = np.clip(np.round(len(data)/10, -3), 1000, 10000)
    if name is None:
        name = jai.generate_name(20, prefix='sdk_', suffix='')
    print(f"name: {name}")
    if name not in jai.names:
        jai.setup(name, data, batch_size=10000, db_type='TextEdit',
                  hyperparams={"nt": nt})
        jai.wait_setup(name, 20)
        jai.delete_raw_data(name)
    results = jai.similar(name, data.index, top_k=top_k, batch_size=10000)
    return process_similar(results, threshold=threshold, return_self=False)


def fill(data, column, auth_key, name=None, **kwargs):
    cat_threshold = 512
    data = data.copy()

    jai = Jai(auth_key)
    if name is None:
        name = jai.generate_name(20, prefix='sdk_', suffix='')

    cat = data.select_dtypes(exclude='number')
    pre = cat.columns[cat.nunique() > cat_threshold].tolist()
    prep_bases = []
    for col in pre:
        id_col = 'id_' + col
        values, inverse = np.unique(data[col], return_inverse=True)
        data[id_col] = inverse
        origin = embedding(values, auth_key)
        prep_bases.append({"id_name": id_col, "db_parent": origin})
    data.drop(columns=pre)

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
    test = data.loc[mask].copy()

    if name not in jai.names:
        jai.setup(name, train, batch_size=10000, db_type='Supervised',
                  hyperparams={"learning_rate": 0.001}, label=label, split=split,
                  **kwargs)
        jai.wait_setup(name, 20)
        jai.delete_raw_data(name)

    results = jai.predict(name, test, predict_proba=True, batch_size=10000)

    return results



def embedding(data, auth_key, name=None, db_type='FastText'):
    jai = Jai(auth_key)
    if name is None:
        name = jai.generate_name(20, prefix='sdk_', suffix='')
    print(f"name: {name}")
    if name not in jai.names:
        jai.setup(name, data, batch_size=10000, db_type=db_type,
                  hyperparams={"minn": 0, "maxn": 0})
        jai.wait_setup(name, 5)
        jai.delete_raw_data(name)
    return name