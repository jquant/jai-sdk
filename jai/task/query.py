import numpy as np
import pandas as pd
from io import BytesIO
from pandas.api.types import is_integer_dtype
from tqdm import trange

from jai.utilities import predict2df

from .base import TaskBase
from ..core.utils_funcs import data2json
from ..core.validations import check_response
from ..types.generic import Mode
from ..types.responses import (
    FieldsResponse,
    SimilarNestedResponse,
    PredictResponse,
    RecNestedResponse,
    FlatResponse,
    DescribeResponse,
)
from typing import Any, Dict, List, Union
import requests

from pydantic import HttpUrl
import sys

if sys.version < "3.8":
    from typing_extensions import Literal
else:
    from typing import Literal

__all__ = ["Query"]


class Query(TaskBase):
    """
    Query task class.

    An authorization key is needed to use the Jai API.

    Parameters
    ----------
    name : str
        String with the name of a database in your JAI environment.
    environment : str
        Jai environment id or name to use. Defaults to "default"
    env_var : str
        The environment variable that contains the JAI authentication token.
        Defaults to "JAI_AUTH".
    verbose : int
        The level of verbosity. Defaults to 1
    safe_mode : bool
        When safe_mode is True, responses from Jai API are validated.
        If the validation fails, the current version you are using is probably incompatible with the current API version.
        We advise updating it to a newer version. If the problem persists and you are on the latest SDK version, please open an issue so we can work on a fix.
    batch_size : int
        Size of the batch to split data sent to the API. It won't change results,
        but a value too small could increase the total process time and a value too large could
        exceed the data limit of the request. Defaults to 16384.

    """

    def __init__(
        self,
        name: str,
        environment: str = "default",
        env_var: str = "JAI_AUTH",
        verbose: int = 1,
        safe_mode: bool = False,
        batch_size: int = 16384,
    ):
        super(Query, self).__init__(
            name=name,
            environment=environment,
            env_var=env_var,
            verbose=verbose,
            safe_mode=safe_mode,
        )

        self.batch_size = batch_size
        if not self.is_valid():
            raise ValueError(
                f"Unable to instantiate Query object because collection with name `{name}` does not exist."
            )

    def check_features(self, columns: List[str], name: str = None):
        def _check_fields(
            fields: Dict[str, FieldsResponse], columns: List[str], mapping: str = "id"
        ):
            not_found = []
            for f in fields[mapping]["fields"]:
                if f["name"] == "id" or f["dtype"] in ["filter", "label"]:
                    pass
                elif f["name"] not in columns and f["dtype"] == "embedding":
                    if len(_check_fields(fields, columns=columns, mapping=f["name"])):
                        not_found.append(f["name"])
                elif f["name"] not in columns:
                    not_found.append(f["name"])
            return not_found

        if name is None:
            name = self.name

        # column validation
        # TODO: typing validation is very complex
        fields = {v["mapping"]: v for v in self.fields()}
        return _check_fields(fields, columns=columns)

    def _generate_batch(
        self,
        data: Union[list, np.ndarray, pd.Index, pd.Series, pd.DataFrame],
        desc: str = None,
    ):
        """
        Breaks data into batches to avoid exceeding data transmission on requests.

        Args:
            data (list, np.ndarray, pd.Index, pd.Series or pd.DataFrame):
               - Use list, np.ndarray or pd.Index for id.
               - Use pd.Series or pd.Dataframe for raw data.
            desc (str, optional): Label for the progress bar. Defaults to None.

        Yields:
            json: data in json format for request.
        """
        if isinstance(data, list):
            data = np.array(data)

        if isinstance(data, (np.ndarray, pd.Index)):
            is_id = True
            # index values
            ids = self.ids(mode="complete")
            inverted_in = np.isin(data, ids, invert=True)
            if inverted_in.sum() > 0:
                missing = data[inverted_in].tolist()
                raise KeyError(
                    f"Id values must belong to the set of Ids from database {self.name}.\n"
                    f"Missing: {missing}"
                )
        elif isinstance(data, (pd.Series, pd.DataFrame)):
            is_id = False

            columns = data.columns if isinstance(data, pd.DataFrame) else [data.name]

            not_found = self.check_features(self.name, columns)
            if len(not_found):
                raise ValueError("")

        else:
            raise ValueError(
                "Data must be `list`, `np.array`, `pd.Index`, `pd.Series` or `pd.DataFrame`"
            )

        for i in trange(0, len(data), self.batch_size, desc=desc):
            if is_id:
                yield is_id, data[i : i + self.batch_size].tolist()
            else:
                _batch = data.iloc[i : i + self.batch_size]
                yield is_id, data2json(_batch, dtype=self.db_type, predict=True)

    def similar(
        self,
        data: Union[list, np.ndarray, pd.Index, pd.Series, pd.DataFrame],
        top_k: int = 5,
        orient: str = "nested",
        filters=None,
    ):
        """
        Query a database in search for the `top_k` most similar entries for each
        input data passed as argument.

        Args
        ----
        data : list, np.ndarray, pd.Index, pd.Series or pd.DataFrame
            Data to be queried for similar inputs in your database.
            - Use list, np.ndarray or pd.Index for id.
            - Use pd.Series or pd.Dataframe for raw data.
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

        results = []
        for is_id, _batch in self._generate_batch(data, desc="Similar"):
            if is_id:
                res = self._similar_id(
                    self.name, _batch, top_k=top_k, orient=orient, filters=filters
                )
            else:
                res = self._similar_json(
                    self.name, _batch, top_k=top_k, orient=orient, filters=filters
                )
            if orient == "flat":
                if self.safe_mode:
                    res = check_response(FlatResponse, res, list_of=True)
                results.extend(res)
            else:
                if self.safe_mode:
                    res = check_response(SimilarNestedResponse, res).dict()
                results.extend(res["similarity"])
        return results

    def recommendation(
        self,
        data: Union[list, np.ndarray, pd.Index, pd.Series, pd.DataFrame],
        top_k: int = 5,
        orient: str = "nested",
        filters=None,
    ):
        """
        Query a database in search for the `top_k` most recommended entries for each
        input data passed as argument.

        Args
        ----
        data : list, np.ndarray, pd.Index, pd.Series or pd.DataFrame
            Data to be queried for recommendation in your database.
            - Use list, np.ndarray or pd.Index for id.
            - Use pd.Series or pd.Dataframe for raw data.
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

        results = []
        for is_id, _batch in self._generate_batch(data, desc="Recommendation"):
            if is_id:
                res = self._recommendation_id(
                    self.name, _batch, top_k=top_k, orient=orient, filters=filters
                )
            else:
                res = self._recommendation_json(
                    self.name, _batch, top_k=top_k, orient=orient, filters=filters
                )
            if orient == "flat":
                if self.safe_mode:
                    res = check_response(FlatResponse, res, list_of=True)
                results.extend(res)
            else:
                if self.safe_mode:
                    res = check_response(RecNestedResponse, res).dict()
                results.extend(res["recommendation"])
        return results

    def predict(
        self,
        data: Union[pd.Series, pd.DataFrame],
        predict_proba: bool = False,
        as_frame: bool = False,
    ):
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
        for _, _batch in self._generate_batch(data, desc="Predict"):
            res = self._predict(self.name, _batch, predict_proba=predict_proba)
            if self.safe_mode:
                res = check_response(PredictResponse, res, list_of=True)
            results.extend(res)

        return predict2df(results) if as_frame else results

    def fields(self):
        """
        Get the table fields for a Supervised/SelfSupervised database.

        Args
        ----
        name : str
            String with the name of a database in your JAI environment.

        Return
        ------
        response : dict
            Dictionary with table fields.

        Example
        -------
        >>> name = 'chosen_name'
        >>> j = Jai(AUTH_KEY)
        >>> fields = j.fields(name=name)
        >>> print(fields)
        {'id': 0, 'feature1': 0.01, 'feature2': 'string', 'feature3': 0}
        """
        fields = self._fields(self.name)
        if self.safe_mode:
            return check_response(
                Dict[
                    str,
                    Literal[
                        "int32",
                        "int64",
                        "float32",
                        "float64",
                        "string",
                        "embedding",
                        "label",
                        "datetime",
                    ],
                ],
                fields,
            )
        return fields

    def download_vectors(self):
        """
        Download vectors from a particular database.

        Args
        ----
        name : str
            String with the name of a database in your JAI environment.

        Return
        ------
        vector : np.array
            Numpy array with all vectors.

        Example
        -------
        >>> name = 'chosen_name'
        >>> j = Jai(AUTH_KEY)
        >>> vectors = j.download_vectors(name=name)
        >>> print(vectors)
        [[ 0.03121682  0.2101511  -0.48933393 ...  0.05550333  0.21190546  0.19986008]
        [-0.03121682 -0.21015109  0.48933393 ...  0.2267401   0.11074653  0.15064166]
        ...
        [-0.03121682 -0.2101511   0.4893339  ...  0.00758727  0.15916921  0.1226602 ]]
        """
        url = self._download_vectors(self.name)
        if self.safe_mode:
            url = check_response(HttpUrl, url)
        r = requests.get(url)
        return np.load(BytesIO(r.content))

    def filters(self):
        """
        Gets the valid values of filters.

        Return
        ------
        response : list of strings
            List of valid filter values.
        """
        filters = self._filters(self.name)
        if self.safe_mode:
            return check_response(List[str], filters)
        return filters

    def ids(self, mode: Mode = "complete"):
        """
        Get id information of a given database.

        Args
        mode : str, optional

        Return
        -------
        response: list
            List with the actual ids (mode: 'complete') or a summary of ids
            ('simple') of the given database.

        Example
        ----------
        >>> name = 'chosen_name'
        >>> j = Jai(AUTH_KEY)
        >>> ids = j.ids(name)
        >>> print(ids)
        ['891 items from 0 to 890']
        """
        ids = self._ids(self.name, mode)
        if self.safe_mode:
            return check_response(List[Any], ids)
        return ids
