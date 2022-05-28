import json
import secrets
import time
from fnmatch import fnmatch
from io import BytesIO
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from pandas.api.types import is_integer_dtype, is_numeric_dtype
from sklearn.model_selection import StratifiedShuffleSplit
from tqdm import tqdm, trange

from jai.utilities import (filter_resolution, filter_similar, predict2df)

from .base import TaskBase
from ..core.utils_funcs import build_name, data2json, resolve_db_type
from ..core.validations import (check_response, check_dtype_and_clean,
                                check_name_lengths, kwargs_validation)
from ..types.generic import Mode, PossibleDtypes
from ..types.responses import (
    EnvironmentsResponse, UserResponse, Report1Response, Report2Response,
    AddDataResponse, StatusResponse, InfoResponse, SetupResponse,
    SimilarNestedResponse, PredictResponse, ValidResponse,
    InsertVectorResponse, RecNestedResponse, FlatResponse)

from pydantic import HttpUrl

from typing import Any, Optional, Dict, List
import sys

if sys.version < '3.8':
    from typing_extensions import Literal
else:
    from typing import Literal

__all__ = ["Query"]


class Query(TaskBase):
    """
    Base class for communication with the Mycelia API.

    Used as foundation for more complex applications for data validation such
    as matching tables, resolution of duplicated values, filling missing values
    and more.

    """

    def __init__(self,
                 name: str,
                 environment: str = "default",
                 env_var: str = "JAI_AUTH",
                 safe_mode: bool = False,
                 verbose: int = 1,
                 batch_size: int = 16384):
        """
        Initialize the Jai class.

        An authorization key is needed to use the Mycelia API.

        Parameters
        ----------

        Returns
        -------
            None

        """
        super(Query, self).__init__(name=name,
                                    environment=environment,
                                    env_var=env_var,
                                      verbose=verbose,
                                    safe_mode=safe_mode)

        self.batch_size = batch_size
        if not self.is_valid():
            raise ValueError(
                "Generic Error Message")  # TODO: Database not does not exist

    def _generate_batch(self, data, is_id: bool = False, desc: str = None):

        for i in trange(0, len(data), self.batch_size, desc=desc):
            if is_id:
                if isinstance(data, pd.Series):
                    yield data.iloc[i:i + self.batch_size].tolist()
                elif isinstance(data, pd.Index):
                    yield data[i:i + self.batch_size].tolist()
                else:
                    yield data[i:i + self.batch_size].tolist()
            else:
                if isinstance(data, (pd.Series, pd.DataFrame)):
                    _batch = data.iloc[i:i + self.batch_size]
                else:
                    _batch = data[i:i + self.batch_size]
                yield data2json(_batch, dtype=self.db_type, predict=True)

    def similar(self,
                data,
                top_k: int = 5,
                orient: str = "nested",
                filters=None):
        """
        Query a database in search for the `top_k` most similar entries for each
        input data passed as argument.

        Args
        ----
        data : list, np.ndarray, pd.Series or pd.DataFrame
            Data to be queried for similar inputs in your database.
        top_k : int
            Number of k similar items that we want to return. `Default is 5`.
        orient : "nested" or "flat"
            Changes the output format. `Default is "nested"`.

        Return
        ------
        results : list of dicts
            A list with a dictionary for each input value identified with
            'query_id' and 'result' which is a list with 'top_k' most similar
            items dictionaries, each dictionary has the 'id' from the database
            previously setup and 'distance' in between the correspondent 'id'
            and 'query_id'.

        Example
        -------
        >>> name = 'chosen_name'
        >>> DATA_ITEM = # data in the format of the database
        >>> TOP_K = 3
        >>> j = Jai(AUTH_KEY)
        >>> df_index_distance = j.similar(name, DATA_ITEM, TOP_K)
        >>> print(pd.DataFrame(df_index_distance['similarity']))
           id  distance
        10007       0.0
        45568    6995.6
         8382    7293.2
        """

        if isinstance(data, list):
            data = np.array(data)
        is_id = is_integer_dtype(data)
        results = []
        for _batch in self._generate_batch(data, is_id=is_id, desc="Similar"):
            if is_id:
                res = self._similar_id(self.name,
                                       _batch,
                                       top_k=top_k,
                                       orient=orient,
                                       filters=filters)
            else:
                res = self._similar_json(self.name,
                                         _batch,
                                         top_k=top_k,
                                         orient=orient,
                                         filters=filters)
            if orient == "flat":
                if self.safe_mode:
                    res = check_response(FlatResponse, res, list_of=True)
                results.extend(res)
            else:
                if self.safe_mode:
                    res = check_response(SimilarNestedResponse, res).dict()
                results.extend(res["similarity"])
        return results

    def recommendation(self,
                       data,
                       top_k: int = 5,
                       orient: str = "nested",
                       filters=None):
        """
        Query a database in search for the `top_k` most recommended entries for each
        input data passed as argument.

        Args
        ----
        data : list, np.ndarray, pd.Series or pd.DataFrame
            Data to be queried for recommendation in your database.
        top_k : int
            Number of k recommendations that we want to return. `Default is 5`.
        orient : "nested" or "flat"
            Changes the output format. `Default is "nested"`.

        Return
        ------
        results : list of dicts
            A list with a dictionary for each input value identified with
            'query_id' and 'result' which is a list with 'top_k' most recommended
            items dictionaries, each dictionary has the 'id' from the database
            previously setup and 'distance' in between the correspondent 'id'
            and 'query_id'.

        Example
        -------
        >>> name = 'chosen_name'
        >>> DATA_ITEM = # data in the format of the database
        >>> TOP_K = 3
        >>> j = Jai(AUTH_KEY)
        >>> df_index_distance = j.recommendation(name, DATA_ITEM, TOP_K)
        >>> print(pd.DataFrame(df_index_distance['recommendation']))
           id  distance
        10007       0.0
        45568    6995.6
         8382    7293.2
        """
        if isinstance(data, list):
            data = np.array(data)

        is_id = is_integer_dtype(data)

        results = []
        for _batch in self._generate_batch(data,
                                           is_id=is_id,
                                           desc="Recommendation"):
            if is_id:
                res = self._recommendation_id(self.name,
                                              _batch,
                                              top_k=top_k,
                                              orient=orient,
                                              filters=filters)
            else:
                res = self._recommendation_json(self.name,
                                                _batch,
                                                top_k=top_k,
                                                orient=orient,
                                                filters=filters)
            if orient == "flat":
                if self.safe_mode:
                    res = check_response(FlatResponse, res, list_of=True)
                results.extend(res)
            else:
                if self.safe_mode:
                    res = check_response(RecNestedResponse, res).dict()
                results.extend(res["recommendation"])
        return results

    def predict(self,
                data,
                predict_proba: bool = False,
                as_frame: bool = False):
        """
        Predict the output of new data for a given database.

        Args
        ----
        data : pd.Series or pd.DataFrame
            Data to be queried for similar inputs in your database.
        predict_proba : bool
            Whether or not to return the probabilities of each prediction is
            it's a classification. `Default is False`.

        Return
        ------
        results : list of dicts
            List of dictionaries with 'id' of the inputed data and 'predict'
            as predictions for the data passed as input.

        Example
        ----------
        >>> name = 'chosen_name'
        >>> DATA_ITEM = # data in the format of the database
        >>> j = Jai(AUTH_KEY)
        >>> preds = j.predict(name, DATA_ITEM)
        >>> print(preds)
        [{"id":0, "predict": "class1"}, {"id":1, "predict": "class0"}]

        >>> preds = j.predict(name, DATA_ITEM, predict_proba=True)
        >>> print(preds)
        [{"id": 0 , "predict"; {"class0": 0.1, "class1": 0.6, "class2": 0.3}}]
        """
        if self.db_type != "Supervised":
            raise ValueError("predict is only available to dtype Supervised.")
        if not isinstance(data, (pd.Series, pd.DataFrame)):
            raise ValueError(
                f"data must be a pandas Series or DataFrame. (data type `{data.__class__.__name__}`)"
            )

        results = []
        for _batch in self._generate_batch(data, desc="Predict"):
            res = self._predict(self.name, _batch, predict_proba=predict_proba)
            if self.safe_mode:
                res = check_response(PredictResponse, res, list_of=True)
            results.extend(res)

        return predict2df(results) if as_frame else results