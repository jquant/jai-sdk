# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 16:00:58 2021

@author: Kazu
"""


import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import Callable


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


def process_similar(results, threshold: int = 5, only_duplicated: bool = False,
                    priority: str = 'closest', validator: Callable = None):
    """
    Process the output from the similar methods.

    Parameters
    ----------
    results : List of Dicts.
        output from similar methods.
    threshold : int, optional
        value for the distance threshold. The default is 5.
    only_duplicated : bool, optional
        option to return all correspondences or only duplicated occurrences.
        The default is False.
    priority : str, optional
        rule to choose the duplicated values, possible options are "min", "max"
        and "closest". The default is 'closest'.
        - "min": chooses the lowest id value.
        - "max": chooses the largest id value.
        - "closest": chooses the chooses the id value with the smallest distance,
        unless the id is equal to the queried id.
    validator : Callable, optional
        function that receive an array of ints and returns an array of bools
        of the same lenght as the input. Used as an extra filter to the id
        values. The default is None.

    Raises
    ------
    ValueError
        If validator is not valid.
    NotImplementedError
        If priority inputed is not implemented.

    Returns
    -------
    pd.Series
        mapping the query id to the similar value.

    """
    if validator is not None:
        if not isinstance(validator, Callable):
            raise ValueError(f"validator {validator} is not Callable type.")
        dummy_array = pd.DataFrame(results[0]['results'])['id'].iloc[:3].values
        mask = validator(dummy_array)
        msg = f"Callable validator {validator} must return a boolean array with same lenght as the input."
        if not isinstance(mask, np.ndarray) or mask.dtype != bool:
            raise ValueError(msg)

    map_duplicate = {}
    for k in tqdm(results, desc="Processing Similar"):
        i = k['query_id']
        df = pd.DataFrame(k['results']).sort_values(by='distance')
        distances = df['distance'].values
        similar = df.loc[distances < threshold, 'id'].values
        if validator is not None:
            similar = similar[validator(similar)]
        if only_duplicated and len(similar) <= 1:
            continue
        elif len(similar) == 0:
            map_duplicate[i] = None
        elif priority == 'min':
            map_duplicate[i] = min(similar)
        elif priority == 'max':
            map_duplicate[i] = max(similar)
        elif priority == 'closest':
            map_duplicate[i] = similar[0] if len(
                similar) <= 1 or similar[0] != i else similar[1]
        else:
            raise NotImplementedError(
                f"priority {priority} is not recognized.")

    return pd.Series(map_duplicate)
