# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 16:00:58 2021

@author: Kazu
"""


import pandas as pd
import numpy as np


def list2json(data_list, name='text'):
    index = pd.Index(range(len(data_list)), name='id')
    series = pd.Series(data_list, index=index, name=name)
    return series.reset_index().to_json(orient='records')


def series2json(data_series, name='text'):
    data_series.index.name = 'id'
    data_series.name = name
    return data_series.reset_index().to_json(orient='records')


def df2json(dataframe, col, name='text'):
    dataframe.index.name = 'id'
    series = dataframe[col].rename(columns={col: name})
    return series.reset_index().to_json(orient='records')


def data2json(data, name='text'):
    if isinstance(data, (set, list, tuple, np.ndarray)):
        return list2json(data, name=name)
    elif isinstance(data, pd.Series):
        return series2json(data, name=name)
    else:
        raise NotImplementedError(f"type {type(data)} is not implemented.")


def process_similar(results, threshold=5):
    map_duplicate = {}

    for k in results:
        i = k['query_id']
        df = pd.DataFrame(k['results'])
        similar = df.loc[df['distance'] < threshold, 'id'].values
        map_duplicate[i] = min(similar)

    return pd.Series(map_duplicate)
