import concurrent
import json
import warnings
from copy import copy
from typing import List, Optional

import psutil
import requests
from decouple import config
from tqdm import tqdm

from ..core.utils_funcs import data2json
from ..core.validations import check_response
from ..types.generic import Mode
from ..types.responses import InsertDataResponse
from .authentication import get_authentication
from .decorators import raise_status_error

__all__ = ["BaseJai"]


class BaseJai(object):
    """
    Base class for requests with the Mycelia API.
    """

    def __init__(
        self,
        environment: str = "default",
        env_var: str = "JAI_AUTH",
        url_var: str = "JAI_URL",
    ):
        """
        Initialize the Jai class.

        An authorization key is needed to use the Mycelia API.

        Parameters
        ----------
        auth_key : str
            Authorization key for the use of the API.
        url : str, optional
            Param used for development purposes. `Default is None`.

        Returns
        -------
            None

        """
        auth_key = get_authentication(env_var)
        self.headers = {"Auth": auth_key, "environment": environment}

        self.url = config(url_var, default="https://mycelia.azure-api.net")

    @property
    def url(self):
        """
        Get name and type of each database in your environment.
        """
        return self.__url

    @url.setter
    def url(self, value):
        """
        Set url.
        """
        self.__url = value[:-1] if value.endswith("/") else value

    @raise_status_error(200)
    def _user(self):
        """
        Get name and type of each database in your environment.
        """
        return requests.get(url=self.url + f"/user", headers=self.headers)

    @raise_status_error(200)
    def _environments(self):
        """
        Get name of environments available.
        """
        return requests.get(url=self.url + f"/environments", headers=self.headers)

    @raise_status_error(200)
    def _info(self, mode="complete", get_size=True):
        """
        Get name and type of each database in your environment.
        """
        get_size = json.dumps(get_size)
        return requests.get(
            url=self.url + f"/info?mode={mode}&get_size={get_size}",
            headers=self.headers,
        )

    @raise_status_error(200)
    def _status(self):
        """
        Get the status of your JAI environment when training.
        """
        return requests.get(self.url + "/status", headers=self.headers)

    @raise_status_error(200)
    def _delete_status(self, name):
        """
        Remove database from status. Used when processing ended.
        """
        return requests.delete(
            self.url + f"/status?db_name={name}", headers=self.headers
        )

    @raise_status_error(200)
    def _download_vectors(self, name: str):
        """
        Download vectors from a particular database.

        Args
        ----
        name : str
            String with the name of a database in your JAI environment.
        """
        return requests.get(self.url + f"/key/{name}", headers=self.headers)

    @raise_status_error(200)
    def _filters(self, name):
        """
        Gets the valid values of filters.

        Args
        ----
        name : str
            String with the name of a database in your JAI environment.
        """
        return requests.get(self.url + f"/filters/{name}", headers=self.headers)

    @raise_status_error(200)
    def _similar_id(
        self,
        name: str,
        id_item: list,
        top_k: int = 5,
        orient: str = "nested",
        filters=None,
    ):
        """
        Creates a list of dicts, with the index and distance of the k items most similars given an id.
        This is a protected method.

        Args
        ----
        name : str
            String with the name of a database in your JAI environment.

        id_item : list
            List of ids of the item the user is looking for.

        top_k : int
            Number of k similar items we want to return. `Default is 5`.

        orient : "nested" or "flat"
            Changes the output format. `Default is "nested"`.

        Return
        ------
        response : dict
            Dictionary with the index and distance of `the k most similar items`.
        """

        if not isinstance(id_item, list):
            raise TypeError(
                f"id_item param must be int or list, `{id_item.__class__.__name__}` found."
            )

        filtering = (
            "" if filters is None else "".join(["&filters=" + s for s in filters])
        )
        url = self.url + f"/similar/id/{name}?top_k={top_k}&orient={orient}" + filtering
        return requests.put(url, headers=self.headers, json=id_item)

    @raise_status_error(200)
    def _similar_json(
        self, name: str, data_json, top_k: int = 5, orient: str = "nested", filters=None
    ):
        """
        Creates a list of dicts, with the index and distance of the k items most similars given a JSON data entry.
        This is a protected method

        Args
        ----
        name : str
            String with the name of a database in your JAI environment.

        data_json : dict (JSON)
            Data in JSON format. Each input in the dictionary will be used to
            search for the `top_k` most similar entries in the database.

        top_k : int
            Number of k similar items we want to return. `Default is 5`.

        orient : "nested" or "flat"
            Changes the output format. `Default is "nested"`.

        Return
        ------
        response : dict
            Dictionary with the index and distance of `the k most similar
            items`.
        """
        filtering = (
            "" if filters is None else "".join(["&filters=" + s for s in filters])
        )
        url = (
            self.url + f"/similar/data/{name}?top_k={top_k}&orient={orient}" + filtering
        )
        header = copy(self.headers)
        header["Content-Type"] = "application/json"
        return requests.put(url, headers=header, data=data_json)

    @raise_status_error(200)
    def _recommendation_id(
        self,
        name: str,
        id_item: list,
        top_k: int = 5,
        orient: str = "nested",
        filters=None,
    ):
        """
        Creates a list of dicts, with the index and distance of the k items
        most similars given an id. This is a protected method.

        Args
        ----
        name : str
            String with the name of a database in your JAI environment.

        id_item : list
            List of ids of the item the user is looking for.

        top_k : int
            Number of k similar items we want to return. `Default is 5`.

        orient : "nested" or "flat"
            Changes the output format. `Default is "nested"`.

        Return
        ------
        response : dict
            Dictionary with the index and distance of `the k most similar
            items`.
        """

        if not isinstance(id_item, list):
            raise TypeError(
                f"id_item param must be int or list, \
                    `{id_item.__class__.__name__}` found."
            )

        filtering = (
            "" if filters is None else "".join(["&filters=" + s for s in filters])
        )
        url = (
            self.url
            + f"/recommendation/id/{name}?top_k={top_k}&orient={orient}"
            + filtering
        )
        return requests.put(
            url,
            headers=self.headers,
            json=id_item,
        )

    @raise_status_error(200)
    def _recommendation_json(
        self, name: str, data_json, top_k: int = 5, orient: str = "nested", filters=None
    ):
        """
        Creates a list of dicts, with the index and distance of the k items most similars given a JSON data entry.
        This is a protected method

        Args
        ----
        name : str
            String with the name of a database in your JAI environment.

        data_json : dict (JSON)
            Data in JSON format. Each input in the dictionary will be used to search for the `top_k` most
            similar entries in the database.

        top_k : int
            Number of k similar items we want to return. `Default is 5`.

        orient : "nested" or "flat"
            Changes the output format. `Default is "nested"`.

        Return
        ------
        response : dict
            Dictionary with the index and distance of `the k most similar items`.
        """
        filtering = (
            "" if filters is None else "".join(["&filters=" + s for s in filters])
        )
        url = (
            self.url
            + f"/recommendation/data/{name}?top_k={top_k}&orient={orient}"
            + filtering
        )
        header = copy(self.headers)
        header["Content-Type"] = "application/json"
        return requests.put(url, headers=header, data=data_json)

    @raise_status_error(200)
    def _predict(self, name: str, data_json, predict_proba: bool = False):
        """
        Predict the output of new data for a given database by calling its
        respecive API method. This is a protected method.

        Args
        ----
        name : str
            String with the name of a database in your JAI environment.
        data_json : JSON file (dict)
            Data to be inferred by the previosly trained model.
        predict_proba : bool
            Whether or not to return the probabilities of each prediction. `Default is False`.

        Return
        -------
        results : dict
            Dictionary of predctions for the data passed as parameter.
        """
        url = self.url + f"/predict/{name}?predict_proba={predict_proba}"

        header = copy(self.headers)
        header["Content-Type"] = "application/json"
        return requests.put(url, headers=header, data=data_json)

    @raise_status_error(200)
    def _ids(self, name: str, mode: Mode = "simple"):
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
        >>> j = Jai(AUTH_KEY)
        >>> ids = j.ids(name)
        >>> print(ids)
        ['891 items from 0 to 890']
        """
        return requests.get(self.url + f"/id/{name}?mode={mode}", headers=self.headers)

    @raise_status_error(200)
    def _is_valid(self, name: str):
        """
        Check if a given name is a valid database name (i.e., if it is in your environment).

        Args
        ----
        `name`: str
            String with the name of a database in your JAI environment.

        Return
        ------
        response: bool
            True if name is in your environment. False, otherwise.
        """
        return requests.get(self.url + f"/validation/{name}", headers=self.headers)

    @raise_status_error(201)
    def _rename(self, original_name: str, new_name: str):
        """
        Get name and type of each database in your environment.
        """
        body = {"original_name": original_name, "new_name": new_name}
        return requests.patch(
            url=self.url + f"/rename", headers=self.headers, json=body
        )

    @raise_status_error(200)
    def _transfer(
        self,
        original_name: str,
        to_environment: str,
        new_name: str = None,
        from_environment: str = "default",
    ):
        """
        Get name and type of each database in your environment.
        """
        body = {
            "from_environment": from_environment,
            "to_environment": to_environment,
            "original_name": original_name,
            "new_name": new_name,
        }
        return requests.post(
            url=self.url + f"/transfer", headers=self.headers, json=body
        )

    @raise_status_error(200)
    def _import_database(
        self,
        database_name: str,
        owner_id: str,
        owner_email: str,
        import_name: str = None,
    ):
        """
        Get name and type of each database in your environment.
        """
        body = {"database_name": database_name, "import_name": import_name}
        return requests.post(
            url=self.url + f"/import?userId={owner_id}&email={owner_email}",
            headers=self.headers,
            json=body,
        )

    @raise_status_error(202)
    def _append(self, name: str):
        """
        Add data to a database that has been previously trained.
        This is a protected method.

        Args
        ----
        name : str
            String with the name of a database in your JAI environment.

        Return
        ------
        response : dict
            Dictionary with the API response.
        """
        return requests.patch(self.url + f"/data/{name}", headers=self.headers)

    @raise_status_error(200)
    def _insert_json(self, name: str, data_json):
        """
        Insert data in JSON format. This is a protected method.

        Args
        ----
        name : str
            String with the name of a database in your JAI environment.
        data_json : dict
            Data in JSON format.

        Return
        ------
        response : dict
            Dictionary with the API response.
        """
        header = copy(self.headers)
        header["Content-Type"] = "application/json"
        return requests.post(self.url + f"/data/{name}", headers=header, data=data_json)

    @raise_status_error(200)
    def _check_parameters(
        self,
        db_type: str,
        hyperparams=None,
        features=None,
        num_process: dict = None,
        cat_process: dict = None,
        datetime_process: dict = None,
        pretrained_bases: list = None,
        label: dict = None,
        split: dict = None,
    ):
        body = {
            "db_type": db_type,
            "hyperparams": hyperparams,
            "features": features,
            "num_process": num_process,
            "cat_process": cat_process,
            "datetime_process": datetime_process,
            "pretrained_bases": pretrained_bases,
            "label": label,
            "split": split,
        }
        return requests.put(self.url + "/parameters", headers=self.headers, json=body)

    @raise_status_error(202)
    def _setup(self, name: str, body, overwrite=False):
        """
        Call the API method for database setup.
        This is a protected method.

        Args
        ----
        name : str
            String with the name of a database in your JAI environment.
        db_type : str
            Database type (Supervised, SelfSupervised, Text...)
        overwrite : bool
            [Optional] Whether of not to overwrite the given database. `Default is False`.
        **kwargs:
            Any parameters the user wants to (or needs to) set for the given datase. Please
            refer to the API methods to see the possible arguments.

        Return
        -------
        response : dict
            Dictionary with the API response.
        """
        overwrite = json.dumps(overwrite)
        return requests.post(
            self.url + f"/setup/{name}?overwrite={overwrite}",
            headers=self.headers,
            json=body,
        )

    @raise_status_error(200)
    def _report(self, name, verbose: int = 2):
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
        return requests.get(
            self.url + f"/report/{name}?verbose={verbose}", headers=self.headers
        )

    @raise_status_error(200)
    def _temp_ids(self, name: str, mode: Mode = "complete"):
        """
        Get id information of a RAW database (i.e., before training). This is a protected method

        Args
        ----
        name : str
            String with the name of a database in your JAI environment.
        mode : str, optional
            Level of detail to return. Possible values are 'simple', 'summarized' or 'complete'.

        Return
        -------
        response: list
            List with the actual ids (mode: 'complete') or a summary of ids
            ('simple'/'summarized') of the given database.
        """
        return requests.get(
            self.url + f"/setup/ids/{name}?mode={mode}", headers=self.headers
        )

    @raise_status_error(200)
    def _fields(self, name: str):
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
        """
        return requests.get(self.url + f"/fields/{name}", headers=self.headers)

    @raise_status_error(200)
    def _describe(self, name: str):
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
        return requests.get(self.url + f"/describe/{name}", headers=self.headers)

    @raise_status_error(200)
    def _cancel_setup(self, name: str):
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
        return requests.post(self.url + f"/cancel/{name}", headers=self.headers)

    @raise_status_error(200)
    def _delete_ids(self, name, ids):
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

        Example
        ----------
        >>> name = 'chosen_name'
        >>> j = Jai(AUTH_KEY)
        >>> j.delete_raw_data(name=name)
        'All raw data from database 'chosen_name' was deleted!'
        """
        return requests.delete(
            self.url + f"/entity/{name}", headers=self.headers, json=ids
        )

    @raise_status_error(200)
    def _delete_raw_data(self, name: str):
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
        >>> j = Jai(AUTH_KEY)
        >>> j.delete_raw_data(name=name)
        'All raw data from database 'chosen_name' was deleted!'
        """
        return requests.delete(self.url + f"/data/{name}", headers=self.headers)

    @raise_status_error(200)
    def _delete_database(self, name: str):
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
        >>> j = Jai(AUTH_KEY)
        >>> j.delete_database(name=name)
        'Bombs away! We nuked database chosen_name!'
        """
        return requests.delete(self.url + f"/database/{name}", headers=self.headers)

    @raise_status_error(201)
    def _insert_vectors_json(self, name: str, data_json, overwrite: bool = False):
        """
        Insert data in JSON format. This is a protected method.
        Args
        ----
        name : str
            String with the name of a database in your JAI environment.
        data_json : dict
            Data in JSON format.
        Return
        ------
        response : dict
            Dictionary with the API response.
        """
        header = copy(self.headers)
        header["Content-Type"] = "application/json"
        return requests.post(
            self.url + f"/vector/{name}?overwrite={overwrite}",
            headers=header,
            data=data_json,
        )

    @raise_status_error(200)
    def _linear_train(
        self,
        name: str,
        data_dict,
        y,
        task,
        learning_rate: float = None,
        l2: float = 0.1,
        model_parameters: dict = None,
        pretrained_bases: list = None,
        overwrite: bool = False,
    ):
        """
        Insert data in JSON format. This is a protected method.
        Args
        ----
        name : str
            String with the name of a database in your JAI environment.
        data_dict : dict
            Data in list of dicts format.
        Return
        ------
        response : dict
            Dictionary with the API response.
        """
        if model_parameters is None:
            model_parameters = {}
        if pretrained_bases is None:
            pretrained_bases = []

        return requests.post(
            self.url + f"/linear/batch/{name}?overwrite={overwrite}",
            headers=self.headers,
            json={
                "X": data_dict,
                "y": y,
                "hyperparams": {
                    "task": task,
                    "learning_rate": learning_rate,
                    "l2": l2,
                    "model": model_parameters,
                },
                "pretrained_bases": pretrained_bases,
            },
        )

    @raise_status_error(200)
    def _linear_learn(self, name: str, data_dict, y: list):
        """
        Insert data in JSON format. This is a protected method.
        Args
        ----
        name : str
            String with the name of a database in your JAI environment.
        data_dict : dict
            Data in list of dicts format.
        Return
        ------
        response : dict
            Dictionary with the API response.
        """
        return requests.post(
            self.url + f"/linear/learn/{name}",
            headers=self.headers,
            json={"X": data_dict, "y": y},
        )

    @raise_status_error(200)
    def _linear_predict(self, name: str, data_dict: list, predict_proba: bool = False):
        """
        Insert data in JSON format. This is a protected method.
        Args
        ----
        name : str
            String with the name of a database in your JAI environment.
        data_dict : dict
            Data in list of dicts format.
        predict_proba : bool
            Makes probability predictions for classification models.

        Return
        ------
        response : dict
            Dictionary with the API response.
        """

        return requests.put(
            self.url + f"/linear/predict/{name}?predict_proba={predict_proba}",
            headers=self.headers,
            json=data_dict,
        )

    def _insert_data(
        self,
        data,
        name,
        db_type,
        batch_size,
        max_insert_workers: Optional[int] = None,
        has_filter: bool = False,
        predict: bool = False,
    ):
        """
        Insert raw data for training. This is a protected method.

        Args
        ----------
        name : str
            String with the name of a database in your JAI environment.
        db_type : str
            Database type (Supervised, SelSupervised, Text...)
        batch_size : int
            Size of batch to send the data.
        predict : bool
            Allows table type data to have only one column for predictions,
            if False, then tables must have at least 2 columns. `Default is False`.

        Return
        ------
        insert_responses : dict
            Dictionary of responses for each batch. Each response contains
            information of whether or not that particular batch was successfully inserted.
        """
        if max_insert_workers is None:
            pcores = psutil.cpu_count(logical=False)
        elif not isinstance(max_insert_workers, int):
            raise TypeError(
                f"Variable 'max_insert_workers' must be 'None' or 'int' instance, not {max_insert_workers.__class__.__name__}."
            )
        elif max_insert_workers > 0:
            pcores = max_insert_workers
        else:
            pcores = 1

        dict_futures = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=pcores) as executor:

            for i, b in enumerate(range(0, len(data), batch_size)):
                _batch = data.iloc[b : b + batch_size]
                data_json = data2json(
                    _batch, dtype=db_type, has_filter=has_filter, predict=predict
                )
                task = executor.submit(self._insert_json, name, data_json)
                dict_futures[task] = i

            with tqdm(total=len(dict_futures), desc="Insert Data") as pbar:
                insert_responses = {}
                for future in concurrent.futures.as_completed(dict_futures):
                    arg = dict_futures[future]
                    insert_res = future.result()
                    if self.safe_mode:
                        insert_res = check_response(InsertDataResponse, insert_res)
                    insert_responses[arg] = insert_res
                    pbar.update(1)

        # check if we inserted everything we were supposed to
        self._check_ids_consistency(name=name, data=data)

        return insert_responses

    def _check_ids_consistency(self, name, data, handle_error="raise"):
        """
        Check if inserted data is consistent with what we expect.
        This is mainly to assert that all data was properly inserted.

        Args
        ----
        name : str
            Database name.
        data : pandas.DataFrame or pandas.Series
            Inserted data.
        handle_error : 'raise' or 'bool'
            If data is inconsistent:
            - `raise`: delete data and raise an error.
            - `bool`: returns False.

        Return
        ------
        bool or Exception
            If an inconsistency is found, an error is raised. If no inconsistency is found, returns True.
        """
        handle_error = handle_error.lower()
        if handle_error not in ["raise", "bool"]:
            warnings.warn(
                f"handle_error must be `raise` or `bool`, found: `{handle_error}`. Using `raise`."
            )
            handle_error = "raise"

        # using mode='simple' to reduce the volume of data transit.
        try:
            inserted_ids = self._temp_ids(name, "simple")
            if self.safe_mode:
                inserted_ids = check_response(List[str], inserted_ids)
        except ValueError as error:
            if handle_error == "raise":
                raise error
            return False

        if len(data) != int(inserted_ids[0].split()[0]):
            if handle_error == "raise":
                print(f"Found invalid ids: {inserted_ids[0]}")
                raise Exception(
                    "Something went wrong on data insertion. Please try again."
                )
            return False
        return True
