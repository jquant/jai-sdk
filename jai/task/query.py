import concurrent
from io import BytesIO
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import requests
from pydantic import HttpUrl
from tqdm.auto import tqdm

from jai.utilities import predict2df

from ..core.utils_funcs import data2json, get_pcores
from ..types.generic import Mode
from ..types.responses import FieldsResponse
from .base import TaskBase

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
        Defaults to False.
    batch_size : int
        Size of the batch to split data sent to the API. It won't change results,
        but a value too small could increase the total process time and a value too large could
        exceed the data limit of the request. Defaults to 16384.

    """

    def __init__(
        self,
        name: str,
        auth_key: str = None,
        environment: str = "default",
        env_var: str = "JAI_AUTH",
        verbose: int = 1,
        safe_mode: bool = False,
        batch_size: int = 16384,
    ):
        super(Query, self).__init__(
            name=name,
            auth_key=auth_key,
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
        """
        It checks if the columns you want to use in your model match the expected from the API.

        Args
        ----
        columns (List[str]): list of columns name to be used.
        name (str): The name of the model. Leave this as is.

        Returns
        -------
        A list of names of columns that were not found.
        """

        def _check_fields(
            fields: Dict[str, FieldsResponse], columns: List[str], mapping: str = "id"
        ):
            not_found = []
            for f in fields[mapping]["fields"]:
                if f["name"] == "id" or f["type"] in ["filter", "label"]:
                    pass
                elif f["name"] not in columns and f["type"] == "embedding":
                    if len(_check_fields(fields, columns=columns, mapping=f["name"])):
                        not_found.append(f["name"])
                elif f["name"] not in columns:
                    not_found.append(f["name"])
            return not_found

        if name is None:
            name = self.name

        # column validation
        # TODO: typing validation is very complex
        fields = self._fields(name)
        fields = {v["mapping"]: v for v in fields}
        columns = [c.replace(".", "_") for c in columns]
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
            if self.safe_mode:
                if desc == "Recommendation":
                    description = self.describe()
                    twin_name = description["twin_base"]
                    ids = self._ids(twin_name, mode="complete")
                else:
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

            if self.safe_mode:
                columns = (
                    data.columns if isinstance(data, pd.DataFrame) else [data.name]
                )

                if desc == "Recommendation":
                    description = self.describe()
                    twin_name = description["twin_base"]
                    not_found = self.check_features(columns, name=twin_name)
                else:
                    not_found = self.check_features(columns)

                if len(not_found):
                    str_missing = "`, `".join(not_found)
                    raise ValueError(
                        f"The following columns were not found in data:\n"
                        f"- `{str_missing}`"
                    )

        else:
            raise ValueError(
                "Data must be `list`, `np.array`, `pd.Index`, `pd.Series` or `pd.DataFrame`"
            )
        db_type = self.db_type
        for i in range(0, len(data), self.batch_size):
            if is_id:
                yield is_id, data[i : i + self.batch_size].tolist()
            else:
                _batch = data.iloc[i : i + self.batch_size]
                yield is_id, data2json(_batch, dtype=db_type, predict=True)

    def similar(
        self,
        data: Union[list, np.ndarray, pd.Index, pd.Series, pd.DataFrame],
        top_k: int = 5,
        orient: str = "nested",
        filters: List[str] = None,
        max_workers: Optional[int] = None,
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
        filters : List of strings
            Filters to use on the similarity query. `Default is None`.
        max_workers : bool
            Number of workers to use to parallelize the process. If None, use all workers. `Defaults to None.`

        Return
        ------
        results : list of dicts
            A list with a dictionary for each input value identified with
            'query_id' and 'result' which is a list with 'top_k' most similar
            items dictionaries, each dictionary has the 'id' from the database
            previously setup and 'distance' in between the correspondent 'id'
            and 'query_id'.

        """
        description = "Similar"
        pcores = get_pcores(max_workers)

        dict_futures = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=pcores) as executor:
            for i, (is_id, _batch) in enumerate(
                self._generate_batch(data, desc=description)
            ):
                if is_id:
                    task = executor.submit(
                        self._similar_id,
                        self.name,
                        _batch,
                        top_k=top_k,
                        orient=orient,
                        filters=filters,
                    )
                else:
                    task = executor.submit(
                        self._similar_json,
                        self.name,
                        _batch,
                        top_k=top_k,
                        orient=orient,
                        filters=filters,
                    )
                dict_futures[task] = i

            with tqdm(total=len(dict_futures), desc=description) as pbar:
                results = []
                for future in concurrent.futures.as_completed(dict_futures):
                    res = future.result()
                    results.extend(res)
                    pbar.update(1)

        return results

    def recommendation(
        self,
        data: Union[list, np.ndarray, pd.Index, pd.Series, pd.DataFrame],
        top_k: int = 5,
        orient: str = "nested",
        filters: List[str] = None,
        max_workers: Optional[int] = None,
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
        filters : List of strings
            Filters to use on the similarity query. `Default is None`.
        max_workers : bool
            Number of workers to use to parallelize the process. If None, use all workers. `Defaults to None.`

        Return
        ------
        results : list of dicts
            A list with a dictionary for each input value identified with
            'query_id' and 'result' which is a list with 'top_k' most recommended
            items dictionaries, each dictionary has the 'id' from the database
            previously setup and 'distance' in between the correspondent 'id'
            and 'query_id'.
        """
        description = "Recommendation"

        pcores = get_pcores(max_workers)

        dict_futures = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=pcores) as executor:
            for i, (is_id, _batch) in enumerate(
                self._generate_batch(data, desc=description)
            ):
                if is_id:
                    task = executor.submit(
                        self._recommendation_id,
                        self.name,
                        _batch,
                        top_k=top_k,
                        orient=orient,
                        filters=filters,
                    )
                else:
                    task = executor.submit(
                        self._recommendation_json,
                        self.name,
                        _batch,
                        top_k=top_k,
                        orient=orient,
                        filters=filters,
                    )
                dict_futures[task] = i

            with tqdm(total=len(dict_futures), desc=description) as pbar:
                results = []
                for future in concurrent.futures.as_completed(dict_futures):
                    res = future.result()
                    results.extend(res)
                    pbar.update(1)
        return results

    def predict(
        self,
        data: Union[pd.Series, pd.DataFrame],
        predict_proba: bool = False,
        as_frame: bool = False,
        max_workers: Optional[int] = None,
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
        as_frame : bool
            Whether or not to return the result of prediction as a DataFrame or list. `Default is False`.
        max_workers : bool
            Number of workers to use to parallelize the process. If None, use all workers. `Defaults to None.`


        Return
        ------
        results : list of dicts
            List of dictionaries with 'id' of the inputed data and 'predict'
            as predictions for the data passed as input.

        """
        if self.db_type != "Supervised":
            raise ValueError("predict is only available to dtype Supervised.")
        if not isinstance(data, (pd.Series, pd.DataFrame)):
            raise ValueError(
                f"data must be a pandas Series or DataFrame. (data type `{data.__class__.__name__}`)"
            )

        description = "Predict"
        pcores = get_pcores(max_workers)

        dict_futures = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=pcores) as executor:
            for i, (_, _batch) in enumerate(
                self._generate_batch(data, desc=description)
            ):
                task = executor.submit(
                    self._predict, self.name, _batch, predict_proba=predict_proba
                )
                dict_futures[task] = i

            with tqdm(total=len(dict_futures), desc=description) as pbar:
                results = []
                for future in concurrent.futures.as_completed(dict_futures):
                    res = future.result()
                    results.extend(res)
                    pbar.update(1)

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
        >>> from jai import Query
        ...
        >>> q = Query(name)
        >>> q.fields()
        """
        return self._fields(self.name)

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
        >>> from jai import Query
        ...
        >>> q = Query(name)
        >>> q.download_vectors()
        >>> print(vectors)
        [[ 0.03121682  0.2101511  -0.48933393 ...  0.05550333  0.21190546  0.19986008]
        [-0.03121682 -0.21015109  0.48933393 ...  0.2267401   0.11074653  0.15064166]
        ...
        [-0.03121682 -0.2101511   0.4893339  ...  0.00758727  0.15916921  0.1226602 ]]
        """
        return self._download_vectors(self.name)

    def filters(self):
        """
        Gets the valid values of filters.

        Return
        ------
        response : list of strings
            List of valid filter values.
        """
        return self._filters(self.name)

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
        >>> from jai import Query
        ...
        >>> q = Query(name)
        >>> q.ids()
        >>> print(ids)
        ['891 items from 0 to 890']
        """
        return self._ids(self.name, mode)
