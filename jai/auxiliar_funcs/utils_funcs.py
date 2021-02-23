# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 16:00:58 2021

@author: Kazu
"""

import base64
import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import List
from pathlib import Path
from PIL import Image

from operator import itemgetter
from functools import cmp_to_key
from .classes import FieldName, PossibleDtypes


__all__ = ['data2json', 'process_predict', 'process_similar']


def read_image_folder(image_folder: str = None, images: List = None, ignore_corrupt=False,
                      extentions=["*.png", "*.jpg", "*.jpeg"]):

    if image_folder is not None:
        images = Path(image_folder).iterdir()
    elif images is not None:
        pass
    else:
        raise ValueError(
            "must pass the folder of the images or a list with the paths of each image.")

    temp_img = []
    ids = []
    corrupted_files = []
    for i, filename in enumerate(tqdm(images)):
        if filename.suffix in extentions:
            try:
                im = Image.open(filename)
                im.verify()  # I perform also verify, don't know if he sees other types o defects
                im.close()  # reload is necessary in my case
                im = Image.open(filename)
                im.transpose(Image.FLIP_LEFT_RIGHT)
                im.close()
                with open(filename, "rb") as image_file:
                    encoded_string = base64.b64encode(
                        image_file.read()).decode("utf-8")
                temp_img.append(encoded_string)
                ids.append(i)
            except KeyboardInterrupt:
                raise KeyboardInterrupt
            except:
                if ignore_corrupt:
                    corrupted_files.append(filename)
                    continue
                else:
                    raise ValueError(f"file {filename} seems to be corrupted.")
    if len(corrupted_files) > 0:
        print("Here are the files that seem to be corrupted:")
        [print(f"{f}") for f in corrupted_files]
    index = pd.Index(ids, name='id')
    return pd.Series(temp_img, index=index, name='image_base64')


def list2json(data_list, name):
    index = pd.Index(range(len(data_list)), name='id')
    series = pd.Series(data_list, index=index, name=name)
    return series.reset_index().to_json(orient='records')


def series2json(data_series, name):
    data_series = data_series.copy()
    data_series.index.name = 'id'
    data_series.name = name
    if data_series.index.duplicated().any():
        raise ValueError("Index must not contain duplicated values.")
    return data_series.reset_index().to_json(orient='records')


def df2json(dataframe):
    dataframe = dataframe.copy()
    if 'id' not in dataframe.columns:
        dataframe.index.name = 'id'
        dataframe = dataframe.reset_index()
    if dataframe.index.duplicated().any():
        raise ValueError("Index must not contain duplicated values.")
    return dataframe.to_json(orient='records')


def data2json(data, dtype):
    if (dtype == PossibleDtypes.edit or dtype == PossibleDtypes.text
        or dtype == PossibleDtypes.fasttext):
        if isinstance(data, (set, list, tuple, np.ndarray)):
            return list2json(data, name=FieldName.text)
        elif isinstance(data, pd.Series):
            return series2json(data, name=FieldName.text)
        elif isinstance(data, pd.DataFrame):
            if data.shape[1] == 1:
                c = data.columns[0]
                return series2json(data[c], name=FieldName.text)
            else:
                raise ValueError("Data must be a DataFrame with one column.")
        else:
            raise NotImplementedError(f"type {type(data)} is not implemented.")
    elif dtype == PossibleDtypes.image:
        if isinstance(data, (set, list, tuple, np.ndarray)):
            return list2json(data, name=FieldName.image)
        if isinstance(data, pd.Series):
            return series2json(data, name=FieldName.image)
        else:
            raise NotImplementedError(f"type {type(data)} is not implemented.")
    elif dtype == PossibleDtypes.supervised or dtype == PossibleDtypes.unsupervised:
        if isinstance(data, pd.DataFrame):
            return df2json(data)
        else:
            raise NotImplementedError(f"type {type(data)} is not implemented.")
    else:
        raise ValueError(f"dtype {dtype} not recognized.")


def process_similar(results, threshold=None, return_self: bool = True,
                    skip_null: bool = True):
    """
    Process the output from the similar methods.

    Parameters
    ----------
    results : List of Dicts.
        output from similar methods.
    threshold : int, optional
        value for the distance threshold. The default is None.
        if set to None, takes a random 1% of the results and uses the 10%
        quantile of the distances distributions as the threshold.
    return_self : bool, optional
        option to return the queried id from the query result or not. The default is True.
    skip_null: bool, optional
        option to skip ids without similar results. The default is True.

    Raises
    ------
    NotImplementedError
        If priority inputed is not implemented.

    Returns
    -------
    pd.Series
        mapping the query id to the similar value.

    """
    if threshold is None:
        samples = np.random.randint(0, len(results), len(results)//(100))
        distribution = []
        for s in tqdm(samples, desc="Fiding threshold"):
            d = [l['distance'] for l in results[s]['results'][1:]]
            distribution.extend(d)
        threshold = np.quantile(distribution, .1)
    print(f"threshold: {threshold}\n")

    similar = []
    for q in tqdm(results.copy(), desc='Process'):
        sort = multikeysort(q['results'], ['distance', 'id'])
        zero, one = sort[0], sort[1]
        if zero['distance'] <= threshold and (zero['id'] != q['query_id'] or return_self):
            zero['query_id'] =  q['query_id']
            similar.append(zero)
        elif one['distance'] <= threshold:
            one['query_id'] = q['query_id']
            similar.append(one)
        elif not skip_null:
            mock = {"query_id":  q['query_id'], "id": None, "distance": None}
            similar.append(mock)
        else:
            continue
    return similar


def process_predict(predicts):
    example = predicts[0]['predict']
    if isinstance(example, dict):
        predict_proba = True
    elif isinstance(example, str):
        predict_proba = False
    else:
        raise ValueError(f"Unexpected predict type. {type(example)}")

    sanity_check = []
    for query in tqdm(predicts, desc='Predict all ids'):
        if predict_proba == False:
            sanity_check.append(query)
        else:
            predict = max(query['predict'], key=query['predict'].get)
            confidence_level = round(query['predict'][predict]*100, 2)
            sanity_check.append({'id': query['id'],
                                 'sanity_prediction': predict,
                                 'confidence_level (%)': confidence_level})
    return sanity_check

# https://stackoverflow.com/a/1144405
# https://stackoverflow.com/a/73050
def cmp(x, y):
    """
    Replacement for built-in function cmp that was removed in Python 3

    Compare the two objects x and y and return an integer according to
    the outcome. The return value is negative if x < y, zero if x == y
    and strictly positive if x > y.

    https://portingguide.readthedocs.io/en/latest/comparisons.html#the-cmp-function
    """

    return (x > y) - (x < y)

def multikeysort(items, columns):
    """
    Sort a list of dictionaries.

    Parameters
    ----------
    items : list of dictionaries
        list of dictionaries to be sorted.
    columns : list of strings
        list of key names to be sorted on the order of the sorting. add '-' at
        the start of the name if it should be sorted from high to low.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    comparers = [
        ((itemgetter(col[1:].strip()), -1) if col.startswith('-') else (itemgetter(col.strip()), 1))
        for col in columns
    ]
    def comparer(left, right):
        comparer_iter = (
            cmp(fn(left), fn(right)) * mult
            for fn, mult in comparers
        )
        return next((result for result in comparer_iter if result), 0)
    return sorted(items, key=cmp_to_key(comparer))