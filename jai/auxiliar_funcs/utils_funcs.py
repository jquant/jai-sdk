# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 16:00:58 2021

@author: Kazu
"""

import base64
import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import Callable, List
from pathlib import Path
from PIL import Image
from .classes import FieldName, PossibleDtypes


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
                    skip_null: bool = True, validator: Callable = None):
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

    if threshold is None:
        samples = np.random.randint(0, len(results), len(results)//(100))
        distribution = []
        for s in tqdm(samples, desc="Fiding threshold"):
            d = [l['distance'] for l in results[s]['results'][1:]]
            distribution.extend(d)
        threshold = np.quantile(distribution, .1)
    print(f"threshold: {threshold}\n")

    map_duplicate = {}
    for k in tqdm(results, desc="Processing Similar"):
        i = k['query_id']
        similar = [l['id'] for l in k['results'] if l['distance']<=threshold]
        if validator is not None:
            similar = similar[validator(similar)]
        if not return_self and i in similar:
            similar.remove(i)

        if similar is None or len(similar) == 0:
            if skip_null:
                continue
            else:
                map_duplicate[i] = None
        else:
            map_duplicate[i] = similar

    return map_duplicate
