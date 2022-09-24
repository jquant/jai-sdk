import concurrent
import json
import secrets
import time
from fnmatch import fnmatch
from typing import List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from pandas.api.types import is_numeric_dtype
from sklearn.model_selection import StratifiedShuffleSplit
from tqdm.auto import tqdm, trange

from jai.utilities import filter_resolution, filter_similar, predict2df

from ..types.generic import Mode, PossibleDtypes
from .base import BaseJai
from .utils_funcs import build_name, data2json, get_pcores, print_args, resolve_db_type
from .validations import check_dtype_and_clean, check_name_lengths, kwargs_validation

__all__ = ["Jai"]


class Jai(BaseJai):
    """
    General class for communication with the Jai API.

    Used as foundation for more complex applications for data validation such
    as matching tables, resolution of duplicated values, filling missing values
    and more.

    An authorization key is needed to use the Jai API.

    Contains the implementation of most functionalities from the API.

    Parameters
    ----------
    environment : str
        Jai environment id or name to use. Defaults to "default"
    env_var : str
        Name of the Environment Variable to get the value of your auth key.
        Defaults to "JAI_AUTH".
    safe_mode : bool
        When safe_mode is True, responses from Jai API are validated.
        If the validation fails, the current version you are using is probably incompatible with the current API version.
        We advise updating it to a newer version. If the problem persists and you are on the latest SDK version, please open an issue so we can work on a fix.
        Defaults to False.

    """

    def __init__(
        self,
        auth_key: str = None,
        environment: str = "default",
        env_var: str = "JAI_AUTH",
        safe_mode: bool = False,
    ):

        super(Jai, self).__init__(
            auth_key=auth_key,
            environment=environment,
            env_var=env_var,
            safe_mode=safe_mode,
        )

    @property
    def names(self):
        """
        Retrieves databases already created for the provided Auth Key.

        Return
        ------
            List with the sorted names of the databases created so far.

        Example
        -------
        >>> j.names
        ['jai_database', 'jai_selfsupervised', 'jai_supervised']

        """
        return self._info(mode="names")

    @property
    def info(self):
        """
        Get name and type of each database in your environment.

        Return
        ------
        pandas.DataFrame
            Pandas dataframe with name, type, creation date and parent
            databases of each database in your environment.

        Example
        -------
        >>> j.info
                                db_name           db_type
        0                  jai_database              Text
        1            jai_selfsupervised    SelfSupervised
        2                jai_supervised        Supervised
        """
        info = self._info()
        df_info = pd.DataFrame(info).rename(
            columns={
                "name": "name",
                "type": "type",
                "version": "last modified",
                "parents": "dependencies",
            }
        )
        if len(df_info) == 0:
            return df_info
        return df_info.sort_values(by="name")

    def status(self, max_tries=5, patience=5):
        """
        Get the status of your JAI environment when training.

        Return
        ------
        response : dict
            A `JSON` file with the current status of the training tasks.

        Example
        -------
        >>> j.status()
        {
            "Task": "Training",
            "Status": "Completed",
            "Description": "Training of database YOUR_DATABASE has ended."
        }
        """
        for _ in range(max_tries):
            try:
                return self._status()
            except BaseException:
                time.sleep(patience)
        return self._status()

    @staticmethod
    def get_auth_key(email: str, firstName: str, lastName: str, company: str = ""):
        """
        Request an auth key to use JAI-SDK with.

        **This method will be deprecated.**
        Please use `get_auth_key` function.

        Args
        ----------
        `email`: str
            A valid email address where the auth key will be sent to.
        `firstName`: str
            User's first name.
        `lastName`: str
            User's last name.
        `company`: str
            User's company.

        Return
        ----------
        `response`: dict
            A Response object with whether or not the auth key was created.
        """
        url = "https://mycelia.azure-api.net/clone"
        body = {
            "email": email,
            "firstName": firstName,
            "lastName": lastName,
            "company": company,
        }
        response = requests.put(url + "/auth", json=body)
        return response

    def user(self):
        """
        User information.

        Returns:
            dict:
            - userId: str
            - email: str
            - firstName: str
            - lastName: str
            - memberRole: str
            - namespace: srt
        """
        return self._user()

    def environments(self):
        """
        Return names of available environments.
        """
        return self._environments()

    def generate_name(self, length: int = 8, prefix: str = "", suffix: str = ""):
        """

        Generate a random string. You can pass a prefix and/or suffix. In this case,
        the generated string will be a concatenation of `prefix + random + suffix`.

        Args
        ----
        length : int
            Length for the desired string. `Default is 8`.
        prefix : str
            Prefix of your string. `Default is empty`.
        suffix  : str
            Suffix of your string. `Default is empty`.

        Returns
        -------
        str
            A random string.

        Example
        ----------
        >>> j.generate_name()
        13636a8b
        >>> j.generate_name(length=16, prefix="company")
        companyb8bbd445d

        """
        len_prefix = len(prefix)
        len_suffix = len(suffix)

        if length <= len_prefix + len_suffix:
            raise ValueError(
                f"length ({length}) should be larger than {len_prefix+len_suffix} for prefix and suffix inputed."
            )
        if length >= 32:
            raise ValueError(f"length ({length}) should be smaller than 32.")

        length -= len_prefix + len_suffix
        code = secrets.token_hex(length)[:length].lower()
        name = str(prefix) + str(code) + str(suffix)
        names = self.names

        while name in names:
            code = secrets.token_hex(length)[:length].lower()
            name = str(prefix) + str(code) + str(suffix)

        return name

    def fields(self, name: str):
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
        >>> j = Jai()
        >>> fields = j.fields(name=name)
        >>> print(fields)
        {'id': 0, 'feature1': 0.01, 'feature2': 'string', 'feature3': 0}
        """
        return self._fields(name)

    def describe(self, name: str):
        """
        Get the database hyperparameters and parameters of a specific database.

        Args
        ----
        name : str
            String with the name of a database in your JAI environment.

        Return
        ------
        response : dict
            Dictionary with database description.
        """
        return self._describe(name)

    def get_dtype(self, name: str):
        """
        Return the database type.

        Parameters
        ----------
        name : str
            String with the name of a database in your JAI environment.

        Raises
        ------
        ValueError
            If the name is not valid.

        Returns
        -------
        db_type : str
            The name of the type of the database.

        """
        return self.describe(name)["dtype"]

    def download_vectors(self, name: str):
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
        >>> j = Jai()
        >>> vectors = j.download_vectors(name=name)
        >>> print(vectors)
        [[ 0.03121682  0.2101511  -0.48933393 ...  0.05550333  0.21190546  0.19986008]
        [-0.03121682 -0.21015109  0.48933393 ...  0.2267401   0.11074653  0.15064166]
        ...
        [-0.03121682 -0.2101511   0.4893339  ...  0.00758727  0.15916921  0.1226602 ]]
        """
        return self._download_vectors(name)

    def filters(self, name: str):
        """
        Gets the valid values of filters.

        Args
        ----
        name : str
            String with the name of a database in your JAI environment.

        Return
        ------
        response : list of strings
            List of valid filter values.
        """
        return self._filters(name)

    def update_database(self, name: str, display_name: str = None, project: str = None):
        return self._update_database(
            name=name, display_name=display_name, project=project
        )

    def similar(
        self,
        name: str,
        data: Union[list, np.ndarray, pd.Index, pd.Series, pd.DataFrame],
        top_k: int = 5,
        orient: str = "nested",
        filters: List[str] = None,
        max_workers: Optional[int] = None,
        batch_size: int = 16384,
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
        batch_size : int
            Size of batches to send the data. `Default is 16384`.

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
        >>> j = Jai()
        >>> df_index_distance = j.similar(name, DATA_ITEM, TOP_K)
        >>> print(pd.DataFrame(df_index_distance['similarity']))
           id  distance
        10007       0.0
        45568    6995.6
         8382    7293.2
        """
        description = "Similar"

        pcores = get_pcores(max_workers)
        dtype = self.get_dtype(name)

        if isinstance(data, list):
            data = np.array(data)

        if isinstance(data, (np.ndarray, pd.Index)):
            is_id = True
        elif isinstance(data, (pd.Series, pd.DataFrame)):
            is_id = False
        else:
            raise ValueError(
                "Data must be `list`, `np.array`, `pd.Index`, `pd.Series` or `pd.DataFrame`"
            )

        dict_futures = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=pcores) as executor:
            for i, b in enumerate(range(0, len(data), batch_size)):
                if is_id:
                    _batch = data[b : b + batch_size].tolist()
                    task = executor.submit(
                        self._similar_id,
                        name,
                        _batch,
                        top_k=top_k,
                        orient=orient,
                        filters=filters,
                    )
                else:
                    _batch = data2json(
                        data.iloc[b : b + batch_size], dtype=dtype, predict=True
                    )
                    task = executor.submit(
                        self._similar_json,
                        name,
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
        name: str,
        data: Union[list, np.ndarray, pd.Index, pd.Series, pd.DataFrame],
        top_k: int = 5,
        orient: str = "nested",
        filters: List[str] = None,
        max_workers: Optional[int] = None,
        batch_size: int = 16384,
    ):
        """
        Query a database in search for the `top_k` most recommended entries for each
        input data passed as argument.

        Args
        ----
        name : str
            String with the name of a database in your JAI environment.
        data : list, np.ndarray, pd.Series or pd.DataFrame
            Data to be queried for recommendation in your database.
        top_k : int
            Number of k recommendations that we want to return. `Default is 5`.
        orient : "nested" or "flat"
            Changes the output format. `Default is "nested"`.
        filters : List of strings
            Filters to use on the similarity query. `Default is None`.
        max_workers : bool
            Number of workers to use to parallelize the process. If None, use all workers. `Defaults to None.`
        batch_size : int
            Size of batches to send the data. `Default is 16384`.

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
        >>> j = Jai()
        >>> df_index_distance = j.recommendation(name, DATA_ITEM, TOP_K)
        >>> print(pd.DataFrame(df_index_distance['recommendation']))
           id  distance
        10007       0.0
        45568    6995.6
         8382    7293.2
        """
        description = "Recommendation"

        pcores = get_pcores(max_workers)
        dtype = self.get_dtype(name)

        if isinstance(data, list):
            data = np.array(data)

        if isinstance(data, (np.ndarray, pd.Index)):
            is_id = True
        elif isinstance(data, (pd.Series, pd.DataFrame)):
            is_id = False
        else:
            raise ValueError(
                "Data must be `list`, `np.array`, `pd.Index`, `pd.Series` or `pd.DataFrame`"
            )

        dict_futures = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=pcores) as executor:
            for i, b in enumerate(range(0, len(data), batch_size)):
                if is_id:
                    _batch = data[b : b + batch_size].tolist()
                    task = executor.submit(
                        self._recommendation_id,
                        name,
                        _batch,
                        top_k=top_k,
                        orient=orient,
                        filters=filters,
                    )
                else:
                    _batch = data2json(
                        data.iloc[b : b + batch_size], dtype=dtype, predict=True
                    )
                    task = executor.submit(
                        self._recommendation_json,
                        name,
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
        name: str,
        data,
        predict_proba: bool = False,
        as_frame: bool = False,
        batch_size: int = 16384,
        max_workers: Optional[int] = None,
    ):
        """
        Predict the output of new data for a given database.

        Args
        ----
        name : str
            String with the name of a database in your JAI environment.
        data : pd.Series or pd.DataFrame
            Data to be queried for similar inputs in your database.
        predict_proba : bool
            Whether or not to return the probabilities of each prediction is
            it's a classification. `Default is False`.
        batch_size : int
            Size of batches to send the data. `Default is 16384`.
        max_workers : bool
            Number of workers to use to parallelize the process. If None, use all workers. `Defaults to None.`

        Return
        ------
        results : list of dicts
            List of dictionaries with 'id' of the inputed data and 'predict'
            as predictions for the data passed as input.

        Example
        ----------
        >>> name = 'chosen_name'
        >>> DATA_ITEM = # data in the format of the database
        >>> j = Jai()
        >>> preds = j.predict(name, DATA_ITEM)
        >>> print(preds)
        [{"id":0, "predict": "class1"}, {"id":1, "predict": "class0"}]

        >>> preds = j.predict(name, DATA_ITEM, predict_proba=True)
        >>> print(preds)
        [{"id": 0 , "predict"; {"class0": 0.1, "class1": 0.6, "class2": 0.3}}]
        """
        dtype = self.get_dtype(name)
        if dtype != "Supervised":
            raise ValueError("predict is only available to dtype Supervised.")
        if not isinstance(data, (pd.Series, pd.DataFrame)):
            raise ValueError(
                f"data must be a pandas Series or DataFrame. (data type `{data.__class__.__name__}`)"
            )

        description = "Predict"
        pcores = get_pcores(max_workers)
        dict_futures = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=pcores) as executor:
            for i, b in enumerate(range(0, len(data), batch_size)):
                _batch = data2json(
                    data.iloc[b : b + batch_size], dtype=dtype, predict=True
                )
                task = executor.submit(
                    self._predict, name, _batch, predict_proba=predict_proba
                )
                dict_futures[task] = i

            with tqdm(total=len(dict_futures), desc=description) as pbar:
                results = []
                for future in concurrent.futures.as_completed(dict_futures):
                    res = future.result()
                    results.extend(res)
                    pbar.update(1)

        return predict2df(results) if as_frame else results

    def ids(self, name: str, mode: Mode = "simple"):
        """
        Get id information of a given database.

        Args
        mode : str, optional

        Return
        -------
        response: list
            List with the actual ids (mode: 'complete') or a summary of ids
            ('simple'/'summarized') of the given database.

        Example
        ----------
        >>> name = 'chosen_name'
        >>> j = Jai()
        >>> ids = j.ids(name)
        >>> print(ids)
        ['891 items from 0 to 890']
        """
        return self._ids(name, mode)

    def is_valid(self, name: str):
        """
        Check if a given name is a valid database name (i.e., if it is
        in your environment).

        Args
        ----
        `name`: str
            String with the name of a database in your JAI environment.

        Return
        ------
        response: bool
            True if name is in your environment. False, otherwise.

        Example
        -------
        >>> name = 'chosen_name'
        >>> j = Jai()
        >>> check_valid = j.is_valid(name)
        >>> print(check_valid)
        True
        """
        return self._is_valid(name)

    def setup(
        self,
        name: str,
        data,
        db_type: str,
        batch_size: int = 16384,
        max_insert_workers: Optional[int] = None,
        frequency_seconds: int = 1,
        verbose: int = 1,
        **kwargs,
    ):
        """
        Insert data and train model. This is JAI's crème de la crème.

        Args
        ----
        name : str
            Database name.
        data : pandas.DataFrame or pandas.Series
            Data to be inserted and used for training.
        db_type : str
            Database type.
            {RecommendationSystem, Supervised, SelfSupervised, Text,
            FastText, TextEdit, Image}
        batch_size : int
            Size of batch to insert the data.`Default is 16384 (2**14)`.
        max_insert_workers : int
            Number of workers to use in the insert data process. `Default is None`.
        frequency_seconds : int
            Time in between each check of status. `Default is 10`.
        verbose : int
            Level of information to retrieve to the user. `Default is 1`.
        **kwargs
            Parameters that should be passed as a dictionary in compliance
            with the API methods. In other words, every kwarg argument
            should be passed as if it were in the body of a POST method.
            **To check all possible kwargs in Jai.setup method,
            you can check the** :ref:`Fit Kwargs <source/reference/fit_kwargs:fit kwargs (keyword arguments)>` **section**.

        Return
        ------
        insert_response : dict
            Dictionary of responses for each data insertion.
        setup_response : dict
            Setup response telling if the model started training.

        Example
        -------
        >>> name = 'chosen_name'
        >>> data = # data in pandas.DataFrame format
        >>> j = Jai()
        >>> _, setup_response = j.setup(
                name=name,
                data=data,
                db_type="Supervised",
                label={
                    "task": "metric_classification",
                    "label_name": "my_label"
                }
            )
        >>> print(setup_response)
        {
            "Task": "Training",
            "Status": "Started",
            "Description": "Training of database chosen_name has started."
        }
        """
        overwrite = kwargs.get("overwrite", False)
        if name in self.names:
            if overwrite:
                self.delete_database(name)
            else:
                raise KeyError(
                    f"Database '{name}' already exists in your environment.\
                        Set overwrite=True to overwrite it."
                )

        if isinstance(data, (pd.Series, pd.DataFrame)):

            # make sure our data has the correct type and is free of NAs
            data = check_dtype_and_clean(data=data, db_type=db_type)

            # insert data
            self._delete_raw_data(name)
            insert_responses = self._insert_data(
                data=data,
                name=name,
                db_type=db_type,
                batch_size=batch_size,
                has_filter=any(
                    [
                        feat["dtype"] == "filter"
                        for feat in kwargs.get("features", {}).values()
                    ]
                ),
                max_insert_workers=max_insert_workers,
            )

        elif isinstance(data, dict):

            # loop insert
            for key, value in data.items():

                # make sure our data has the correct type and is free of NAs
                value = check_dtype_and_clean(data=value, db_type=db_type)

                if key == "main":
                    key = name

                # insert data
                self._delete_raw_data(key)
                insert_responses = self._insert_data(
                    data=value,
                    name=key,
                    db_type=db_type,
                    batch_size=batch_size,
                    has_filter=any(
                        [
                            feat["dtype"] == "filter"
                            for feat in kwargs.get("features", {}).values()
                        ]
                    ),
                    max_insert_workers=max_insert_workers,
                    predict=False,
                )
        else:
            ValueError(
                "Data must be a pd.Series, pd.Dataframe or a dictionary of pd.DataFrames."
            )

        # train model
        body = kwargs_validation(db_type=db_type, **kwargs)
        setup_response = self._setup(name, body, overwrite)
        print_args(
            {k: json.loads(v) for k, v in setup_response["kwargs"].items()},
            dict(db_type=db_type, **kwargs),
            verbose=verbose,
        )

        if frequency_seconds >= 1:
            self.wait_setup(name=name, frequency_seconds=frequency_seconds)

            if db_type in [
                PossibleDtypes.selfsupervised,
                PossibleDtypes.supervised,
                PossibleDtypes.recommendation_system,
            ]:
                self.report(name, verbose)

        return insert_responses, setup_response

    def fit(self, *args, **kwargs):
        """
        Another name for setup.
        """
        return self.setup(*args, **kwargs)

    def rename(self, original_name: str, new_name: str):
        return self._rename(original_name=original_name, new_name=new_name)

    def transfer(
        self,
        original_name: str,
        to_environment: str,
        new_name: str = None,
        from_environment: str = "default",
    ):
        return self._transfer(
            original_name=original_name,
            to_environment=to_environment,
            new_name=new_name,
            from_environment=from_environment,
        )

    def import_database(
        self,
        database_name: str,
        owner_id: str,
        owner_email: str,
        import_name: str = None,
    ):
        return self._import_database(
            database_name=database_name,
            owner_id=owner_id,
            owner_email=owner_email,
            import_name=import_name,
        )

    def add_data(
        self, name: str, data, batch_size: int = 16384, frequency_seconds: int = 1
    ):
        """
        Insert raw data and extract their latent representation.

        This method should be used when we already setup up a database using `setup()`
        and want to create the vector representations of new data
        using the model we already trained for the given database.

        Args
        ----
        name : str
            String with the name of a database in your JAI environment.
        data : pandas.DataFrame or pandas.Series
            Data to be inserted and used for training.
        batch_size : int
            Size of batch to send the data. `Default is 16384`.
        frequency_seconds : int
            Time in between each check of status. `Default is 10`.

        Return
        -------
        insert_responses: dict
            Dictionary of responses for each batch. Each response contains
            information of whether or not that particular batch was successfully inserted.
        """
        # delete data reamains
        self.delete_raw_data(name)

        # get the db_type
        describe = self.describe(name)
        db_type = describe["dtype"]

        # make sure our data has the correct type and is free of NAs
        data = check_dtype_and_clean(data=data, db_type=db_type)

        # insert data
        self._delete_raw_data(name)
        insert_responses = self._insert_data(
            data=data,
            name=name,
            db_type=db_type,
            batch_size=batch_size,
            has_filter=describe["has_filter"],
            predict=True,
        )

        # add data per se
        add_data_response = self._append(name=name)

        if frequency_seconds >= 1:
            self.wait_setup(name=name, frequency_seconds=frequency_seconds)

        return insert_responses, add_data_response

    def append(
        self, name: str, data, batch_size: int = 16384, frequency_seconds: int = 1
    ):
        """
        Another name for add_data
        """
        return self.add_data(
            name=name,
            data=data,
            batch_size=batch_size,
            frequency_seconds=frequency_seconds,
        )

    def report(self, name, verbose: int = 2, return_report: bool = False):
        """
        Get a report about the training model.

        Parameters
        ----------
        name : str
            String with the name of a database in your JAI environment.
        verbose : int, optional
            Level of description. The default is 2.
            Use verbose 2 to get the loss graph, verbose 1 to get only the
            metrics result.

        Returns
        -------
        dict
            Dictionary with the information.

        """
        dtype = self.get_dtype(name)
        if dtype not in [
            PossibleDtypes.selfsupervised,
            PossibleDtypes.supervised,
            PossibleDtypes.recommendation_system,
        ]:
            return None

        result = self._report(name, verbose)

        if return_report:
            return result

        if "Model Training" in result.keys():
            plots = result["Model Training"]

            plt.plot(*plots["train"], label="train loss")
            plt.plot(*plots["val"], label="val loss")
            plt.title("Training Losses")
            plt.legend()
            plt.xlabel("epoch")
            plt.show()

        print("\nSetup Report:")
        print(result["Model Evaluation"]) if result.get(
            "Model Evaluation", None
        ) is not None else None
        print()
        print(result["Loading from checkpoint"].split("\n")[1]) if result.get(
            "Loading from checkpoint", None
        ) is not None else None

    def wait_setup(self, name: str, frequency_seconds: int = 1):
        """
        Wait for the setup (model training) to finish

        Placeholder method for scripts.

        Args
        ----
        name : str
            String with the name of a database in your JAI environment.
        frequency_seconds : int, optional
            Number of seconds apart from each status check. `Default is 5`.

        Return
        ------
        None.
        """

        def get_numbers(sts):
            curr_step, max_iterations = (
                sts["Description"].split("Iteration: ")[1].strip().split(" / ")
            )
            return int(curr_step), int(max_iterations)

        status = self.status()[name]
        starts_at, max_steps = status["CurrentStep"], status["TotalSteps"]

        step = starts_at
        aux = 0
        sleep_time = frequency_seconds
        try:
            with tqdm(
                total=max_steps,
                desc="JAI is working",
                bar_format="{l_bar}{bar}|{n_fmt}/{total_fmt} [{elapsed}]",
            ) as pbar:
                while status["Status"] != "Task ended successfully.":
                    if status["Status"] == "Something went wrong.":
                        raise BaseException(status["Description"])
                    elif fnmatch(status["Description"], "*Iteration:*"):
                        # create a second progress bar to track
                        # training progress
                        _, max_iterations = get_numbers(status)
                        with tqdm(
                            total=max_iterations, desc=f"[{name}] Training", leave=False
                        ) as iteration_bar:
                            while fnmatch(status["Description"], "*Iteration:*"):
                                curr_step, _ = get_numbers(status)
                                step_update = curr_step - iteration_bar.n
                                if step_update > 0:
                                    iteration_bar.update(step_update)
                                    sleep_time = frequency_seconds
                                else:
                                    sleep_time += frequency_seconds
                                time.sleep(sleep_time)
                                status = self.status()[name]
                            # training might stop early, so we make the progress bar appear
                            # full when early stopping is reached -- peace of mind
                            iteration_bar.update(max_iterations - iteration_bar.n)

                    if (step == starts_at) and (aux == 0):
                        pbar.update(starts_at)
                    else:
                        pbar.update(step - starts_at)
                        starts_at = step

                    step = status["CurrentStep"]
                    time.sleep(frequency_seconds)
                    status = self.status()[name]
                    aux += 1

                if (starts_at != max_steps) and aux != 0:
                    pbar.update(max_steps - starts_at)
                elif (starts_at != max_steps) and aux == 0:
                    pbar.update(max_steps)

        except KeyboardInterrupt:
            print("\n\nInterruption caught!\n\n")
            response = self._cancel_setup(name)
            raise KeyboardInterrupt(response)

        response = self._delete_status(name)
        return status

    def delete_ids(self, name, ids):
        """
        Delete the specified ids from database.

        Args
        ----
        name : str
            String with the name of a database in your JAI environment.

        ids : list
            List of ids to be removed from database.

        Return
        -------
        response : dict
            Dictionary with the API response.
        """
        return self._delete_ids(name, ids)

    def delete_raw_data(self, name: str):
        """
        Delete raw data. It is good practice to do this after training a model.

        Args
        ----
        name : str
            String with the name of a database in your JAI environment.

        Return
        -------
        response : dict
            Dictionary with the API response.

        Example
        ----------
        >>> name = 'chosen_name'
        >>> j = Jai()
        >>> j.delete_raw_data(name=name)
        'All raw data from database 'chosen_name' was deleted!'
        """
        return self._delete_raw_data(name)

    def delete_database(self, name: str):
        """
        Delete a database and everything that goes with it (I thank you all).

        Args
        ----
        name : str
            String with the name of a database in your JAI environment.

        Return
        ------
        response : dict
            Dictionary with the API response.

        Example
        -------
        >>> name = 'chosen_name'
        >>> j = Jai()
        >>> j.delete_database(name=name)
        'Bombs away! We nuked database chosen_name!'
        """
        return self._delete_database(name)

    # Helper function to delete the whole tree of databases related with
    # database 'name'
    def _delete_tree(self, name):
        df = self.info
        if len(df) == 0:
            return

        bases_to_del = df.loc[df["name"] == name, "dependencies"].values[0]
        bases_to_del.append(name)
        total = len(bases_to_del)
        for i, base in enumerate(bases_to_del):
            try:
                msg = self.delete_database(base)
            except:
                msg = f"Database '{base}' does not exist in your environment."
            print(f"({i+1} out of {total}) {msg}")

    def embedding(
        self,
        name: str,
        data,
        db_type="TextEdit",
        batch_size: int = 16384,
        frequency_seconds: int = 1,
        hyperparams=None,
        overwrite=False,
    ):
        """
        Quick embedding for high numbers of categories in columns.

        Parameters
        ----------
        name: str
            String with the name of a database in your JAI environment.
        data : pd.Series
            Data for your text based model.
        db_type : str, optional
            type of model to be trained. The default is 'TextEdit'.
        hyperparams: optional
            See setup documentation for the db_type used.

        Returns
        -------
        name : str
            name of the base where the data was embedded.

        """
        if isinstance(data, pd.Series):
            data = data.copy()
        else:
            raise ValueError(
                f"data must be a Series. data is `{data.__class__.__name__}`"
            )

        ids = data.index

        if db_type == "TextEdit":
            hyperparams = {"nt": np.clip(np.round(len(data) / 10, -3), 1000, 10000)}

        if name not in self.names or overwrite:
            self.setup(
                name,
                data,
                db_type=db_type,
                batch_size=batch_size,
                overwrite=overwrite,
                frequency_seconds=frequency_seconds,
                hyperparams=hyperparams,
            )
        else:
            missing = ids[~np.isin(ids, self.ids(name, "complete"))]
            if len(missing) > 0:
                self.add_data(
                    name,
                    data.loc[missing],
                    batch_size=batch_size,
                    frequency_seconds=frequency_seconds,
                )
        return ids

    def match(
        self,
        name: str,
        data_left,
        data_right,
        top_k: int = 100,
        batch_size: int = 16384,
        threshold: float = None,
        original_data: bool = False,
        db_type="TextEdit",
        hyperparams=None,
        overwrite: bool = False,
    ):
        """
        Match two datasets with their possible equal values.

        Queries the data right to get the similar results in data left.

        Parameters
        ----------
        name: str
            String with the name of a database in your JAI environment.
        data_left, data_right : pd.Series
            data to be matched.
        top_k : int, optional
            Number of similars to query. Default is 100.
        threshold : float, optional
            Distance threshold to decide if the result is the same item or not.
            Smaller distances give more strict results. Default is None.
            The threshold is automatically set by default, but may need manual
            setting for more accurate results.
        original_data : bool, optional
            If True, returns the values of the original data along with the ids.
            Default is False.
        db_type : str, optional
            type of model to be trained. The default is 'TextEdit'.
        hyperparams: dict, optional
            See setup documentation for the db_type used.
        overwrite : bool, optional
            If True, then the model is always retrained. Default is False.

        Returns
        -------
        pd.DataFrame
            Returns a dataframe with the matching ids of data_left and data_right.

        Example
        -------
        >>> import pandas as pd
        >>> from jai.processing import process_similar
        >>>
        >>> j = Jai()
        >>> match = j.match(name, data1, data2)
        >>> match
                  id_left     id_right     distance
           0            1            2         0.11
           1            2            1         0.11
           2            3          NaN          NaN
           3            4          NaN          NaN
           4            5            5         0.15
        """
        self.embedding(
            name,
            data_left,
            db_type=db_type,
            batch_size=batch_size,
            hyperparams=hyperparams,
            overwrite=overwrite,
        )
        similar = self.similar(name, data_right, top_k=top_k, batch_size=batch_size)
        processed = filter_similar(similar, threshold=threshold, return_self=True)
        match = pd.DataFrame(processed).sort_values("query_id")
        match = match.rename(columns={"id": "id_left", "query_id": "id_right"})
        if original_data:
            match["data_letf"] = data_left.loc[match["id_left"]].to_numpy(copy=True)
            match["data_rigth"] = data_right.loc[match["id_right"]].to_numpy(copy=True)

        return match

    def resolution(
        self,
        name: str,
        data,
        top_k: int = 20,
        batch_size: int = 16384,
        threshold: float = None,
        return_self: bool = True,
        original_data: bool = False,
        db_type="TextEdit",
        hyperparams=None,
        overwrite=False,
    ):
        """
        Experimental

        Find possible duplicated values within the data.

        Parameters
        ----------
        name: str
            String with the name of a database in your JAI environment.
        data : pd.Series
            data to find duplicates.
        top_k : int, optional
            Number of similars to query. Default is 100.
        threshold : float, optional
            Distance threshold to decide if the result is the same item or not.
            Smaller distances give more strict results. Default is None.
            The threshold is automatically set by default, but may need manual
            setting for more accurate results.
        original_data : bool, optional
            If True, returns the ids when resolution_id is the same as id.
            Default is True.
        original_data : bool, optional
            If True, returns the values of the original data along with the ids.
            Default is False.
        db_type : str, optional
            type of model to be trained. The default is 'TextEdit'.
        hyperparams: dict, optional
            See setup documentation for the db_type used.
        overwrite : bool, optional
            If True, then the model is always retrained. Default is False.

        Returns
        -------
        pd.DataFrame
            Each id with its resolution id. More columns depending on parameters.

        Example
        -------
        >>> import pandas as pd
        >>> from jai.processing import process_similar
        >>>
        >>> j = Jai()
        >>> results = j.resolution(name, data)
        >>> results
          id  resolution_id
           0              0
           1              0
           2              0
           3              3
           4              3
           5              5
        """
        ids = self.embedding(
            name,
            data,
            db_type=db_type,
            batch_size=batch_size,
            hyperparams=hyperparams,
            overwrite=overwrite,
        )
        simliar = self.similar(name, ids, top_k=top_k, batch_size=batch_size)
        connect = filter_resolution(
            simliar, threshold=threshold, return_self=return_self
        )
        r = pd.DataFrame(connect).set_index("id").sort_index()

        if original_data:
            r["Original"] = data.loc[r.index.values].to_numpy(copy=True)
            r["Resolution"] = data.loc[r["resolution_id"].values].to_numpy(copy=True)

        return r

    def fill(
        self,
        name: str,
        data,
        column: str,
        batch_size: int = 16384,
        db_type="TextEdit",
        **kwargs,
    ):
        """
        Experimental

        Fills the column in data with the most likely value given the other columns.

        Only works with categorical columns. Can not fill missing values for
        numerical columns.

        Parameters
        ----------
        name: str
            String with the name of a database in your JAI environment.
        data : pd.DataFrame
            data to fill NaN.
        column : str
            name of the column to be filled.
        db_type : str or dict
            which db_type to use for embedding high dimensional categorical columns.
            If a string is provided, we assume that all columns will be embedded using that db_type;
            if a dict-like structure {"col1": "TextEdit", "col2": "FastText", ...} is provided, we embed the
            specified columns with their respective db_types, and columns not in dict are by default embedded
            with "TextEdit"
        **kwargs : TYPE
            Extra args for supervised model. See setup method.

        Returns
        -------
        list of dicts
            List of dicts with possible filling values for each id with column NaN.

        Example
        -------
        >>> import pandas as pd
        >>> from jai.processing import predict2df
        >>>
        >>> j = Jai()
        >>> results = j.fill(name, data, COL_TO_FILL)
        >>> processed = predict2df(results)
        >>> pd.DataFrame(processed).sort_values('id')
                  id   sanity_prediction    confidence_level (%)
           0       1             value_1                    70.9
           1       4             value_1                    67.3
           2       7             value_1                    80.2
        """
        if "id" in data.columns:
            data = data.set_index("id")
        data = data.copy()
        cat_threshold = kwargs.get("cat_threshold", 512)
        overwrite = kwargs.get("overwrite", False)
        as_frame = kwargs.get("as_frame", False)

        # delete tree of databases derived from 'name',
        # including 'name' itself
        if overwrite:
            self._delete_tree(name)

        if column in data.columns:
            vals = data.loc[:, column].value_counts() < 2
            if vals.sum() > 0:
                eliminate = vals[vals].index.tolist()
                print(
                    f"values {eliminate} from column {column} were removed for having less than 2 examples."
                )
                data.loc[data[column].isin(eliminate), column] = None
        else:
            data.loc[:, column] = None

        cat = data.select_dtypes(exclude="number")

        if name not in self.names:
            mask = data.loc[:, column].isna()
            train = data.loc[~mask].copy()
            test = data.loc[mask].drop(columns=[column])

            # first columns to include are the ones that satisfy
            # the cat_threshold
            pre = cat.columns[cat.nunique() > cat_threshold].tolist()

            # check if db_type is a dict and has some keys in it
            # that do not satisfy the cat_threshold, but must be
            # processed anyway
            if isinstance(db_type, dict):
                pre.extend([item for item in db_type.keys() if item in cat.columns])

            # we make `pre` a set to ensure it has
            # unique column names
            pre = set(pre)

            prep_bases = []

            # check if database and column names will not overflow the 32-character
            # concatenation limit
            check_name_lengths(name, pre)

            for col in pre:
                id_col = "id_" + col
                origin = build_name(name, col)

                # find out which db_type to use for this particular column
                curr_db_type = resolve_db_type(db_type, col)

                train[id_col] = self.embedding(origin, train[col], db_type=curr_db_type)
                test[id_col] = self.embedding(origin, test[col], db_type=curr_db_type)

                prep_bases.append({"id_name": id_col, "db_parent": origin})
            train = train.drop(columns=pre)
            test = test.drop(columns=pre)

            label = {"task": "metric_classification", "label_name": column}
            split = {"type": "stratified", "split_column": column, "test_size": 0.2}

            pretrained_bases = kwargs.get(
                "pretrained_bases", kwargs.get("mycelia_bases", [])
            )
            kwargs["pretrained_bases"] = pretrained_bases
            kwargs.pop("mycelia_bases", None)
            if not kwargs["pretrained_bases"]:
                del kwargs["pretrained_bases"]

            self.setup(
                name,
                train,
                db_type="Supervised",
                batch_size=batch_size,
                label=label,
                split=split,
                **kwargs,
            )
        else:
            drop_cols = []
            for col in cat.columns:
                id_col = "id_" + col
                origin = build_name(name, col)

                if origin in self.names:
                    data[id_col] = self.embedding(origin, data[col])
                    drop_cols.append(col)
            if column in data.columns:
                drop_cols.append(column)
            test = data.drop(columns=drop_cols)

        ids_test = test.index
        missing_test = ids_test[~np.isin(ids_test, self.ids(name, "complete"))]
        if len(missing_test) > 0:
            self.add_data(name, test.loc[missing_test], batch_size=batch_size)
        return self.predict(
            name, test, predict_proba=True, batch_size=batch_size, as_frame=as_frame
        )

    def sanity(
        self,
        name: str,
        data,
        batch_size: int = 16384,
        columns_ref: list = None,
        db_type="TextEdit",
        **kwargs,
    ):
        """
        Experimental

        Validates consistency in the columns (columns_ref).

        Parameters
        ----------
        name: str
            String with the name of a database in your JAI environment.
        data : pd.DataFrame
            Data reference of sound data.
        columns_ref : list, optional
            Columns that can have inconsistencies. As default we use all non numeric columns.
        db_type : str or dict
            which db_type to use for embedding high dimensional categorical columns.
            If a string is provided, we assume that all columns will be embedded using that db_type;
            if a dict-like structure {"col1": "TextEdit", "col2": "FastText", "col3": "Text", ...} is provided,
            we embed the specified columns with their respective db_types,
            and columns not in dict are by default embedded with "TextEdit"
        kwargs :
            Extra args for supervised model except label and split. See setup method. Also:

            * **frac** (float):
                Percentage of the orignal dataframe to be shuffled to create
                invalid samples for each column in columns_ref. `Default is 0.1`.
            * **random_seed** (int):
                random seed. `Default is 42`.
            * **cat_threshold** (int):
                threshold for processing categorical columns with fasttext model.
                `Default is 512`.
            * **target** (str):
                target validation column. If target is already in data, shuffling is skipped.
                `Default is "is_valid"`.


        Returns
        -------
        list of dicts
            Result of data is valid or not.

        Example
        -------
        >>> import pandas as pd
        >>> from jai.processing import predict2df
        >>>
        >>> j = Jai()
        >>> results = j.sanity(name, data)
        >>> processed = predict2df(results)
        >>> pd.DataFrame(processed).sort_values('id')
                  id   sanity_prediction    confidence_level (%)
           0       1               Valid                    70.9
           1       4             Invalid                    67.3
           2       7             Invalid                    80.6
           3      13               Valid                    74.2
        """
        if "id" in data.columns:
            data = data.set_index("id")
        data = data.copy()

        frac = kwargs.get("frac", 0.1)
        random_seed = kwargs.get("random_seed", 42)
        cat_threshold = kwargs.get("cat_threshold", 512)
        target = kwargs.get("target", "is_valid")
        overwrite = kwargs.get("overwrite", False)
        as_frame = kwargs.get("as_frame", False)

        # delete tree of databases derived from 'name',
        # including 'name' itself
        if overwrite:
            self._delete_tree(name)

        SKIP_SHUFFLING = target in data.columns

        np.random.seed(random_seed)

        cat = data.select_dtypes(exclude="number")

        if name not in self.names:
            # first columns to include are the ones that satisfy
            # the cat_threshold
            pre = cat.columns[cat.nunique() > cat_threshold].tolist()

            # check if db_type is a dict and has some keys in it
            # that do not satisfy the cat_threshold, but must be
            # processed anyway
            if isinstance(db_type, dict):
                pre.extend([item for item in db_type.keys() if item in cat.columns])

            # we make `pre` a set to ensure it has
            # unique column names
            pre = set(pre)

            if columns_ref is None:
                columns_ref = cat.columns.tolist()
            elif not isinstance(columns_ref, list):
                columns_ref = columns_ref.tolist()

            prep_bases = []

            # check if database and column names will not overflow the 32-character
            # concatenation limit
            check_name_lengths(name, pre)

            for col in pre:
                id_col = "id_" + col
                origin = build_name(name, col)

                # find out which db_type to use for this particular column
                curr_db_type = resolve_db_type(db_type, col)
                data[id_col] = self.embedding(origin, data[col], db_type=curr_db_type)
                prep_bases.append({"id_name": id_col, "db_parent": origin})

                if col in columns_ref:
                    columns_ref.remove(col)
                    columns_ref.append(id_col)

            data = data.drop(columns=pre)

            if not SKIP_SHUFFLING:

                def change(options, original):
                    return np.random.choice(options[options != original])

                # get a sample of the data and shuffle it
                sample = []
                strat_split = StratifiedShuffleSplit(
                    n_splits=1, test_size=frac, random_state=0
                )
                for c in columns_ref:
                    indexes = []
                    # We try to get a stratified sample on each column.
                    # However, stratified does not work with NaN values, so
                    # we need to drop them before getting the samples
                    try:
                        _, indexes = next(
                            strat_split.split(
                                data.dropna(subset=[c]), data.dropna(subset=[c])[c]
                            )
                        )
                        s = data.dropna(subset=[c]).iloc[indexes].copy()
                    except:
                        pass

                    # due to dropping NaN values, the number of samples might
                    # fall short the desired amount we want;
                    # in this case, we give up the stratified strategy and simply
                    # sample our database randomly.
                    # This 'if' statement below will also hold true if
                    # stratified sampling could not work for whatever
                    # reason (for instance, all samples in a given column are different)
                    if len(indexes) < int(np.floor(data.shape[0] * frac)):
                        s = data.sample(frac=frac)

                    uniques = s[c].unique()
                    s.loc[:, c] = [change(uniques, v) for v in s[c]]
                    sample.append(s)
                sample = pd.concat(sample)

                # set target column values
                sample[target] = "Invalid"

                # set index of samples with different values as data
                sample.index = 10 ** int(np.log10(data.shape[0]) + 2) + np.arange(
                    len(sample)
                )
                data[target] = "Valid"
                train = pd.concat([data, sample])
            else:
                train = data.copy()

            label = {"task": "metric_classification", "label_name": target}
            split = {"type": "stratified", "split_column": target, "test_size": 0.2}

            pretrained_bases = kwargs.get(
                "pretrained_bases", kwargs.get("mycelia_bases", [])
            )
            kwargs["pretrained_bases"] = pretrained_bases
            kwargs.pop("mycelia_bases", None)
            if not kwargs["pretrained_bases"]:
                del kwargs["pretrained_bases"]

            self.setup(
                name,
                train,
                db_type="Supervised",
                batch_size=batch_size,
                label=label,
                split=split,
                **kwargs,
            )
        else:

            drop_cols = []
            for col in cat.columns:
                id_col = "id_" + col
                origin = build_name(name, col)

                if origin in self.names:
                    data[id_col] = self.embedding(origin, data[col])
                    drop_cols.append(col)

            data = data.drop(columns=drop_cols)

            ids = data.index
            missing = ids[~np.isin(ids, self.ids(name, "complete"))]

            if len(missing) > 0:
                self.add_data(name, data.loc[missing], batch_size=batch_size)
        return self.predict(
            name, data, predict_proba=True, batch_size=batch_size, as_frame=as_frame
        )

    def insert_vectors(
        self,
        data,
        name,
        batch_size: int = 10000,
        overwrite: bool = False,
        append: bool = False,
    ):
        """
        Insert raw vectors database directly into JAI without any need of fit.

        Args
        -----
        data : pd.DataFrame, pd.Series or np.ndarray
            Database data to be inserted.
        name : str
            String with the name of a database in your JAI environment.
        batch_size : int, optional
            Size of batch to send the data.
        overwrite : bool, optional
            If True, then the vector database is always recriated. Default is False.
        append : bool, optional
            If True, then the inserted data will be added to the existent database. Default is False.

        Return
        ------
        insert_responses : dict
            Dictionary of responses for each batch. Each response contains
            information of whether or not that particular batch was successfully inserted.
        """

        if name in self.names:
            if overwrite:
                create_new_collection = True
                self.delete_database(name)
            elif not overwrite and append:
                create_new_collection = False
            else:
                raise KeyError(
                    f"Database '{name}' already exists in your environment."
                    f"Set overwrite=True to overwrite it or append=True to add new data to your database."
                )
        else:
            # delete data remains
            create_new_collection = True
            self.delete_raw_data(name)

        # make sure our data has the correct type and is free of NAs
        data = check_dtype_and_clean(data=data, db_type=PossibleDtypes.vector)

        # Check if all values are numeric
        non_num_cols = [
            x for x in data.columns.tolist() if not is_numeric_dtype(data[x])
        ]
        if non_num_cols:
            raise ValueError(
                f"Columns {non_num_cols} contains values types different from numeric."
            )

        insert_responses = {}
        for i, b in enumerate(trange(0, len(data), batch_size, desc="Insert Vectors")):
            _batch = data.iloc[b : b + batch_size]
            data_json = data2json(_batch, dtype=PossibleDtypes.vector, predict=False)

            if i == 0 and create_new_collection is True:
                response = self._insert_vectors_json(name, data_json, overwrite=True)

            else:
                response = self._insert_vectors_json(name, data_json, overwrite=False)

            insert_responses[i] = response

        return insert_responses
