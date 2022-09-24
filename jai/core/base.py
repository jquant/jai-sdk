import concurrent
import json
import warnings
from copy import copy
from io import BytesIO
from typing import Any, Dict, List, Optional

import numpy as np
import psutil
import requests
from decouple import config
from pydantic import HttpUrl
from tqdm.auto import tqdm

from ..core.utils_funcs import data2json, get_pcores
from ..core.validations import check_response
from ..types.generic import Mode
from ..types.responses import (
    AddDataResponse,
    DescribeResponse,
    EnvironmentsResponse,
    FieldsResponse,
    FlatResponse,
    InfoResponse,
    InfoSizeResponse,
    InsertDataResponse,
    InsertVectorResponse,
    LinearFitResponse,
    LinearLearnResponse,
    LinearPredictResponse,
    PredictResponse,
    RecNestedResponse,
    Report1Response,
    Report2Response,
    SetupResponse,
    SimilarNestedResponse,
    StatusResponse,
    UserResponse,
    ValidResponse,
)
from .authentication import get_authentication
from .exceptions import DeprecatedError, ParamError, ValidationError

__all__ = ["BaseJai", "RequestJai"]


class RequestJai(object):
    """
    Base class for requests with the Mycelia API.
    An authorization key is needed to use the Mycelia API.

    Parameters
    ----------
    environment : str
        Jai environment id or name to use. Defaults to "default"
    env_var : str
        Name of the Environment Variable to get the value of your auth key.
        Defaults to "JAI_AUTH".
    url_var : str
        Name of the Environment Variable to get the value of API's URL.
        Used to help development only. Defaults to "JAI_URL".

    Returns
    -------
        None
    """

    def __init__(
        self,
        auth_key: str = None,
        environment: str = "default",
        env_var: str = "JAI_AUTH",
        url_var: str = "JAI_URL",
    ):
        if auth_key is None:
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

    def _get__user(self):
        """
        Get name and type of each database in your environment.
        """
        return requests.get(url=self.url + f"/user", headers=self.headers)

    def _get__environments(self):
        """
        Get name of environments available.
        """
        return requests.get(url=self.url + f"/environments", headers=self.headers)

    def _get__info(self, mode="complete", get_size=True):
        """
        Get name and type of each database in your environment.
        """
        get_size = json.dumps(get_size)
        return requests.get(
            url=self.url + f"/info?mode={mode}&get_size={get_size}",
            headers=self.headers,
        )

    def _get__status(self):
        """
        Get the status of your JAI environment when training.
        """
        return requests.get(self.url + "/status", headers=self.headers)

    def _delete__status(self, name):
        """
        Remove database from status. Used when processing ended.
        """
        return requests.delete(
            self.url + f"/status?db_name={name}", headers=self.headers
        )

    def _get__download_vectors(self, name: str):
        """
        Download vectors from a particular database.

        Args
        ----
        name : str
            String with the name of a database in your JAI environment.
        """
        return requests.get(self.url + f"/key/{name}", headers=self.headers)

    def _get__filters(self, name):
        """
        Gets the valid values of filters.

        Args
        ----
        name : str
            String with the name of a database in your JAI environment.
        """
        return requests.get(self.url + f"/filters/{name}", headers=self.headers)

    def _put__similar_id(
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

    def _put__similar_json(
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

    def _put__recommendation_id(
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

    def _put__recommendation_json(
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

    def _put__predict(self, name: str, data_json, predict_proba: bool = False):
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

    def _get__ids(self, name: str, mode: Mode = "simple"):
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

    def _get__validation(self, name: str):
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

    def _patch__rename(self, original_name: str, new_name: str):
        """
        Change name of a database in your environment.
        """
        body = {"original_name": original_name, "new_name": new_name}
        return requests.patch(
            url=self.url + f"/rename", headers=self.headers, json=body
        )

    def _post__update_database(
        self, name: str, display_name: str = None, project: str = None
    ):
        if display_name is None and project is None:
            raise ValueError("must pass one of `displayName` or `project`")
        elif display_name is None:
            path = f"/database/{name}?project={project}"
        elif project is None:
            path = f"/database/{name}?displayName={display_name}"
        else:
            path = f"/database/{name}?displayName={display_name}&project={project}"
        return requests.post(self.url + path, headers=self.headers)

    def _post__transfer(
        self,
        original_name: str,
        to_environment: str,
        new_name: str = None,
        from_environment: str = "default",
    ):
        """
        Transfer a database between environments.
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

    def _post__import_database(
        self,
        database_name: str,
        owner_id: str,
        owner_email: str,
        import_name: str = None,
    ):
        """
        Import a database from a públic environment.
        """
        body = {"database_name": database_name, "import_name": import_name}
        return requests.post(
            url=self.url + f"/import?userId={owner_id}&email={owner_email}",
            headers=self.headers,
            json=body,
        )

    def _patch__append(self, name: str):
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

    def _post__insert_json(self, name: str, data_json):
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

    def _put__check_parameters(
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

    def _post__setup(self, name: str, body, overwrite=False):
        """
        Call the API method for database setup.
        This is a protected method.

        Args
        ----
        name : str
            String with the name of a database in your JAI environment.
        body : dict
            Any parameters the user wants to (or needs to) set for the given datase. Please
            refer to the API methods to see the possible arguments.
        overwrite : bool
            [Optional] Whether of not to overwrite the given database. `Default is False`.


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

    def _get__report(self, name, verbose: int = 2):
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

    def _get__temp_ids(self, name: str, mode: Mode = "complete"):
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

    def _get__fields(self, name: str):
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

    def _get__describe(self, name: str):
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

    def _post__cancel_setup(self, name: str):
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

    def _delete__ids(self, name, ids):
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
            self.url + f"/entity/{name}?{'&'.join([f'id={i}'for i in ids])}",
            headers=self.headers,
        )

    def _delete__raw_data(self, name: str):
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

    def _delete__database(self, name: str):
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

    def _post__insert_vectors_json(self, name: str, data_json, overwrite: bool = False):
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

    def _post__linear_train(
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

    def _post__linear_learn(self, name: str, data_dict, y: list):
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

    def _put__linear_predict(
        self, name: str, data_dict: list, predict_proba: bool = False
    ):
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

    def _get__linear_model_weights(self, name: str):
        """
        Get model weights from the specified model. This is a protected method.
        Args
        ----
        name : str
            String with the name of a database in your JAI environment.

        Return
        ------
        response : dict
            Dictionary with the API response.
        """

        return requests.get(self.url + f"/linear/weights/{name}", headers=self.headers)


class BaseJai(RequestJai):
    """
    Base class for requests with the Mycelia API.
    An authorization key is needed to use the Mycelia API.
    Performs basic validations to the requests made, a layer of complexity above RequestJai.

    Parameters
    ----------
    environment : str
        Jai environment id or name to use. Defaults to "default"
    env_var : str
        Name of the Environment Variable to get the value of your auth key.
        Defaults to "JAI_AUTH".
    url_var : str
        Name of the Environment Variable to get the value of API's URL.
        Used to help development only. Defaults to "JAI_URL".
    safe_mode : bool
        When safe_mode is True, responses from Jai API are validated.
        If the validation fails, the current version you are using is probably incompatible with the current API version.
        We advise updating it to a newer version. If the problem persists and you are on the latest SDK version, please open an issue so we can work on a fix.
        If safe_mode is True also checks if code is exact, otherwise checks only if code is 2xx. Defaults to False.

    Returns
    -------
        None
    """

    def __init__(
        self,
        auth_key: str = None,
        environment: str = "default",
        env_var: str = "JAI_AUTH",
        url_var: str = "JAI_URL",
        safe_mode: bool = False,
    ):
        super(BaseJai, self).__init__(
            auth_key=auth_key, environment=environment, env_var=env_var, url_var=url_var
        )
        self.safe_mode = safe_mode

    def _check_status_code(self, response, code: int = 200):
        """
        Decorator to process responses with unexpected response codes.

        If safe_mode is True, then checks if code is exact,
        otherwise checks only if code is 2xx. Defaults to False.

        Args
        ----
        code: int
            Expected Code.
        """

        if self.safe_mode and response.status_code == code:
            return response.json()
        elif response.status_code >= 200 and response.status_code <= 299:
            return response.json()
        # find a way to process this
        # what errors to raise, etc.
        message = f"Something went wrong.\n\nSTATUS: {response.status_code}\n"
        try:
            res_json = response.json()
            if isinstance(res_json, dict):
                detail = res_json.get("message", res_json.get("detail", response.text))
            else:
                detail = response.text
        except:
            detail = response.text

        detail = str(detail)

        if "Error: " not in detail:
            raise ValueError(message + detail)

        error, msg = detail.split(": ", 1)
        try:
            raise eval(error)(message + msg)
        except:
            if error == "DeprecatedError":
                raise DeprecatedError(message + msg)
            elif error == "ValidationError":
                raise ValidationError(message + msg)
            elif error == "ParamError":
                raise ParamError(message + msg)
            raise BaseException(message + response.text)

    def _user(self):
        """
        Get name and type of each database in your environment.
        """
        response = self._get__user()
        user = self._check_status_code(response)
        if self.safe_mode:
            return check_response(UserResponse, user).dict()
        return user

    def _environments(self):
        """
        Get name of environments available.
        """
        response = self._get__environments()
        envs = self._check_status_code(response)
        if self.safe_mode:
            environments = []
            for v in check_response(EnvironmentsResponse, envs, list_of=True):
                if v["key"] is None:
                    v.pop("key")
                environments.append(v)
            return environments
        return envs

    def _info(self, mode="complete", get_size=True):
        """
        Get name and type of each database in your environment.
        """
        response = self._get__info(mode=mode, get_size=get_size)
        info = self._check_status_code(response)
        if mode == "names":
            names = info
            if self.safe_mode:
                names = check_response(List[str], names)
            return sorted(names)
        elif self.safe_mode:
            if get_size:
                info = check_response(InfoSizeResponse, info, list_of=True)
            else:
                info = check_response(InfoResponse, info, list_of=True)
        return info

    def _status(self):
        """
        Get the status of your JAI environment when training.
        """
        response = self._get__status()
        status = self._check_status_code(response)
        if self.safe_mode:
            return check_response(Dict[str, StatusResponse], status, as_dict=True)
        return status

    def _delete_status(self, name: str):
        """
        Remove database from status. Used when processing ended.
        """
        response = self._delete__status(name)
        response = self._check_status_code(response)
        if self.safe_mode:
            response = check_response(str, response)
        return response

    def _download_vectors(self, name: str):
        """
        Download vectors from a particular database.

        Args
        ----
        name : str
            String with the name of a database in your JAI environment.
        """
        response = self._get__download_vectors(name)
        url = self._check_status_code(response)
        if self.safe_mode:
            url = check_response(HttpUrl, url)
        r = requests.get(url)
        return np.load(BytesIO(r.content))

    def _filters(self, name):
        """
        Gets the valid values of filters.

        Args
        ----
        name : str
            String with the name of a database in your JAI environment.
        """
        response = self._get__filters(name)
        filters = self._check_status_code(response)
        if self.safe_mode:
            return check_response(List[str], filters)
        return filters

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
        response = self._put__similar_id(
            name=name,
            id_item=id_item,
            top_k=top_k,
            orient=orient,
            filters=filters,
        )
        res = self._check_status_code(response)
        if orient == "flat":
            if self.safe_mode:
                res = check_response(FlatResponse, res, list_of=True)
            return res
        if self.safe_mode:
            res = check_response(SimilarNestedResponse, res).dict()
        return res["similarity"]

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
        response = self._put__similar_json(
            name=name,
            data_json=data_json,
            top_k=top_k,
            orient=orient,
            filters=filters,
        )
        sim = self._check_status_code(response)
        if orient == "flat":
            if self.safe_mode:
                sim = check_response(FlatResponse, sim, list_of=True)
            return sim
        if self.safe_mode:
            sim = check_response(SimilarNestedResponse, sim).dict()
        return sim["similarity"]

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
        response = self._put__recommendation_id(
            name=name,
            id_item=id_item,
            top_k=top_k,
            orient=orient,
            filters=filters,
        )
        rec = self._check_status_code(response)
        if orient == "flat":
            if self.safe_mode:
                rec = check_response(FlatResponse, rec, list_of=True)
            return rec
        if self.safe_mode:
            rec = check_response(RecNestedResponse, rec).dict()
        return rec["recommendation"]

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
        response = self._put__recommendation_json(
            name=name,
            data_json=data_json,
            top_k=top_k,
            orient=orient,
            filters=filters,
        )
        rec = self._check_status_code(response)
        if orient == "flat":
            if self.safe_mode:
                rec = check_response(FlatResponse, rec, list_of=True)
            return rec
        if self.safe_mode:
            rec = check_response(RecNestedResponse, rec).dict()
        return rec["recommendation"]

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
        response = self._put__predict(
            name=name,
            data_json=data_json,
            predict_proba=predict_proba,
        )
        pred = self._check_status_code(response)
        if self.safe_mode:
            pred = check_response(PredictResponse, pred, list_of=True)
        return pred

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
        response = self._get__ids(name=name, mode=mode)
        ids = self._check_status_code(response)

        if self.safe_mode:
            ids = check_response(List[Any], ids)

        return ids

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
        response = self._get__validation(name=name)
        valid = self._check_status_code(response)
        if self.safe_mode:
            valid = check_response(ValidResponse, valid).dict()
        return valid["value"]

    def _rename(self, original_name: str, new_name: str):
        """
        Change name of a database in your environment.
        """
        response = self._patch__rename(original_name=original_name, new_name=new_name)
        response = self._check_status_code(response, code=201)
        if self.safe_mode:
            return check_response(str, response)
        return response

    def _transfer(
        self,
        original_name: str,
        to_environment: str,
        new_name: str = None,
        from_environment: str = "default",
    ):
        """
        Transfer a database between environments.
        """
        response = self._post__transfer(
            original_name=original_name,
            to_environment=to_environment,
            new_name=new_name,
            from_environment=from_environment,
        )
        response = self._check_status_code(response)
        if self.safe_mode:
            return check_response(str, response)
        return response

    def _import_database(
        self,
        database_name: str,
        owner_id: str,
        owner_email: str,
        import_name: str = None,
    ):
        """
        Import a database from a públic environment.
        """
        response = self._post__import_database(
            database_name=database_name,
            owner_id=owner_id,
            owner_email=owner_email,
            import_name=import_name,
        )
        response = self._check_status_code(response)
        if self.safe_mode:
            return check_response(str, response)
        return response

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
        response = self._patch__append(name=name)
        add_data_response = self._check_status_code(
            response,
            code=202,
        )
        if self.safe_mode:
            add_data_response = check_response(AddDataResponse, add_data_response)
        return add_data_response

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
        response = self._post__insert_json(name=name, data_json=data_json)
        insert_res = self._check_status_code(
            response,
            code=202,
        )
        if self.safe_mode:
            insert_res = check_response(InsertDataResponse, insert_res)
        return insert_res

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
        response = self._put__check_parameters(
            db_type=db_type,
            hyperparams=hyperparams,
            features=features,
            num_process=num_process,
            cat_process=cat_process,
            datetime_process=datetime_process,
            pretrained_bases=pretrained_bases,
            label=label,
            split=split,
        )
        return self._check_status_code(response)

    def _setup(self, name: str, body, overwrite=False):
        """
        Call the API method for database setup.
        This is a protected method.

        Args
        ----
        name : str
            String with the name of a database in your JAI environment.
        body : dict
            Any parameters the user wants to (or needs to) set for the given datase. Please
            refer to the API methods to see the possible arguments.
        overwrite : bool
            [Optional] Whether of not to overwrite the given database. `Default is False`.


        Return
        -------
        response : dict
            Dictionary with the API response.
        """
        response = self._post__setup(name=name, body=body, overwrite=overwrite)
        setup_response = self._check_status_code(
            response,
            code=202,
        )
        if self.safe_mode:
            setup_response = check_response(SetupResponse, setup_response).dict()
        return setup_response

    def _update_database(
        self, name: str, display_name: str = None, project: str = None
    ):
        response = self._post__update_database(
            name=name, display_name=display_name, project=project
        )
        setup_response = self._check_status_code(response)
        if self.safe_mode:
            setup_response = check_response(str, setup_response)
        return setup_response

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
        response = self._get__report(name=name, verbose=verbose)
        report = self._check_status_code(
            response,
        )
        if self.safe_mode:
            if verbose >= 2:
                report = check_response(Report2Response, report).dict(by_alias=True)
            elif verbose == 1:
                report = check_response(Report1Response, report).dict(by_alias=True)
            else:
                report = check_response(Report1Response, report).dict(by_alias=True)
        return report

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
        response = self._get__temp_ids(name=name, mode=mode)
        inserted_ids = self._check_status_code(
            response,
        )
        if self.safe_mode:
            inserted_ids = check_response(List[str], inserted_ids)
        return inserted_ids

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
        response = self._get__fields(name=name)
        fields = self._check_status_code(
            response,
        )
        if self.safe_mode:
            return check_response(FieldsResponse, fields, list_of=True)
        return fields

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
        response = self._get__describe(name=name)
        description = self._check_status_code(
            response,
        )
        if self.safe_mode:
            description = check_response(DescribeResponse, description).dict()
            description = {k: v for k, v in description.items() if v is not None}
        return description

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
        response = self._post__cancel_setup(name=name)
        response = self._check_status_code(response, code=204)
        if self.safe_mode:
            response = check_response(str, response)
        return response

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
        response = self._delete__ids(name=name, ids=ids)
        response = self._check_status_code(response)
        if self.safe_mode:
            response = check_response(str, response)
        return response

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
        response = self._delete__raw_data(name=name)
        response = self._check_status_code(response)
        if self.safe_mode:
            response = check_response(str, response)
        return response

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
        response = self._delete__database(name=name)
        response = self._check_status_code(response)
        if self.safe_mode:
            response = check_response(str, response)
        return response

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
        response = self._post__insert_vectors_json(
            name=name, data_json=data_json, overwrite=overwrite
        )
        response = self._check_status_code(
            response,
            code=201,
        )
        if self.safe_mode:
            response = check_response(InsertVectorResponse, response)
        return response

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
        Train a new linear model.

        Args
        ----
          X: pd.DataFrame)
            dataframe of features.
          y: pd.Series):
            The target variable.
          pretrained_bases: list
            mapping of ids to previously trained databases.
          overwrite: bool
            If True, will overwrite the model if it already exists. Defaults to False.

        Returns
        -------
          A dictionary with information about the training.
          - id_train: List[Any]
          - id_test: List[Any]
          - metrics: Dict[str, Union[float, str]]
        """
        response = self._post__linear_train(
            name=name,
            data_dict=data_dict,
            y=y,
            task=task,
            learning_rate=learning_rate,
            l2=l2,
            model_parameters=model_parameters,
            pretrained_bases=pretrained_bases,
            overwrite=overwrite,
        )
        response = self._check_status_code(response)
        if self.safe_mode:
            response = check_response(LinearFitResponse, response).dict(by_alias=True)

        return response

    def _linear_learn(self, name: str, data_dict, y: list):
        """
        Improves an existing model with informantion from a new data.

        Args
        ----
        X: pd.DataFrame
            dataframe of features.
        y: pd.Series
            The target variable.

        Returns
        -------
        response: dict
            A dictionary with the learning report.
            - before: Dict[str, Union[float, str]]
            - after: Dict[str, Union[float, str]]
            - change: bool
        """
        response = self._post__linear_learn(
            name=name,
            data_dict=data_dict,
            y=y,
        )
        response = self._check_status_code(response)
        if self.safe_mode:
            response = check_response(LinearLearnResponse, response).dict()

        return response

    def _linear_predict(self, name: str, data_dict: list, predict_proba: bool = False):
        """
        Makes the prediction using the linear models.

        Args
        ----
        X: pd.DataFrame
            Raw data to be predicted.
        predict_proba:bool):
            If True, the model will return the probability of each class. Defaults to False.
        as_frame: bool
            If True, the result will be returned as a pandas DataFrame. If False, it will be
        returned as a list of dictionaries. Defaults to True.

        Returns
        -------
            A list of dictionaries.
        """
        response = self._put__linear_predict(
            name=name,
            data_dict=data_dict,
            predict_proba=predict_proba,
        )
        result = self._check_status_code(response)
        if self.safe_mode:
            if predict_proba:
                result = check_response(List[Dict[Any, Any]], result)
            else:
                result = check_response(LinearPredictResponse, result, list_of=True)
        return result

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
