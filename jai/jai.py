import secrets
import json
import pandas as pd
import numpy as np
import requests
import time

from .functions.utils_funcs import data2json, pbar_steps
from .functions.classes import PossibleDtypes, Mode
from pandas.api.types import is_integer_dtype
from tqdm import trange, tqdm

__all__ = ["Jai"]


class Jai:
    """
    Base class for communication with the Mycelia API.

    Used as foundation for more complex applications for data validation such
    as matching tables, resolution of duplicated values, filling missing values
    and more.

    """
    def __init__(self, auth_key: str, url: str = None):
        """
        Inicialize the Jai class.

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
        if url is None:
            self.__url = "https://mycelia.azure-api.net"
            self.header = {"Auth": auth_key}
        else:
            if url.endswith("/"):
                url = url[:-1]
            self.__url = url
            self.header = {"company-key": auth_key}

    @property
    def url(self):
        """
        API Url that the class uses for requests made.

        """
        return self.__url

    @property
    def names(self):
        """
        Retrieves databases already created for the provided Auth Key.

        Args
        ----
            None

        Return
        ------
            List with the databases created so far.

        Example
        -------
        >>> j.names
        ['jai_database', 'jai_unsupervised', 'jai_supervised']

        """
        response = requests.get(url=self.url + "/info?mode=names",
                                headers=self.header)

        if response.status_code == 200:
            return response.json()
        else:
            return self.assert_status_code(response)

    @property
    def info(self):
        """
        Get name and type of each database in your environment.

        Args
        ----
            None

        Return
        ------
        pandas.DataFrame
            Pandas dataframe with name and type of each database in your environment.

        Example
        -------
        >>> j.info
                                db_name       db_type
        0                  jai_database          Text
        1              jai_unsupervised  Unsupervised
        2                jai_supervised    Supervised
        """
        response = requests.get(url=self.url + "/info?mode=complete",
                                headers=self.header)

        if response.status_code == 200:
            df = pd.DataFrame(response.json()).rename({
                "db_name": "name",
                "db_type": "type"
            })
            return df
        else:
            return self.assert_status_code(response)

    @property
    def status(self):
        """
        Get the status of your JAI environment when training.

        Args
        ----
            None

        Return
        ------
        response : dict
            A `JSON` file with the current status of the training tasks.

        Example
        -------
        >>> j.status
        {
            "Task": "Training",
            "Status": "Completed",
            "Description": "Training of database YOUR_DATABASE has ended."
        }
        """
        response = requests.get(self.url + "/status", headers=self.header)

        max_trials = 5
        patience = 25  # time in seconds that we'll wait
        trials = 0

        while trials < max_trials:
            if response.status_code == 200:
                return response.json()
            time.sleep(patience // max_trials)
            trials += 1
            response = requests.get(self.url + "/status", headers=self.header)
        return self.assert_status_code(response)

    @staticmethod
    def get_auth_key(email: str, firstName: str, lastName: str):
        """
        Request an auth key to use JAI-SDK with.

        Args
        ----------
        `email`: str
            A valid email address where the auth key will be sent to.
        `firstName`: str
            User's first name.
        `lastName`: str
            User's last name.

        Return
        ----------
        `response`: dict
            A Response object with whether or not the auth key was created.
        """
        url = "https://mycelia.azure-api.net/clone"
        body = {"email": email, "firstName": firstName, "lastName": lastName}
        response = requests.put(url + "/auth", data=json.dumps(body))
        return response

    def generate_name(self,
                      length: int = 8,
                      prefix: str = "",
                      suffix: str = ""):
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
                f"length {length} is should be larger than {len_prefix+len_suffix} for prefix and suffix inputed."
            )

        length -= len_prefix + len_suffix
        code = secrets.token_hex(length)[:length].lower()
        name = str(prefix) + str(code) + str(suffix)
        names = self.names

        while name in names:
            code = secrets.token_hex(length)[:length].lower()
            name = str(prefix) + str(code) + str(suffix)

        return name

    def assert_status_code(self, response):
        # find a way to process this
        # what errors to raise, etc.
        print(response.json())
        print(f"\n\nSTATUS: {response.status_code}\n\n")
        raise ValueError(f"Something went wrong.\n{response.content}")

    def similar(self,
                name: str,
                data,
                top_k: int = 5,
                batch_size: int = 16384):
        """
        Query a database in search for the `top_k` most similar entries for each
        input data passed as argument.

        Args
        ----
        name : str
            String with the name of a database in your JAI environment.
        data : list, pd.Series or pd.DataFrame
            Data to be queried for similar inputs in your database.
        top_k : int
            Number of k similar items that we want to return. `Default is 5`.
        batch_size : int
            Size of batches to send the data. `Default is 16384`.

        Return
        ------
        results : dict
            Dictionary with the index and distance of the k most similar items.

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
        dtype = self._get_dtype(name)

        if isinstance(data, list):
            data = np.array(data)

        is_id = is_integer_dtype(data)

        results = []
        for i in trange(0, len(data), batch_size, desc="Similar"):
            if is_id:
                if isinstance(data, pd.Series):
                    _batch = data.iloc[i:i + batch_size].tolist()
                elif isinstance(data, pd.Index):
                    _batch = data[i:i + batch_size].tolist()
                else:
                    _batch = data[i:i + batch_size].tolist()
                res = self._similar_id(name, _batch, top_k=top_k)
            else:
                if isinstance(data, (pd.Series, pd.DataFrame)):
                    _batch = data.iloc[i:i + batch_size]
                else:
                    _batch = data[i:i + batch_size]
                res = self._similar_json(name,
                                         data2json(_batch, dtype=dtype),
                                         top_k=top_k)
            results.extend(res["similarity"])
        return results

    def _similar_id(self, name: str, id_item: int, top_k: int = 5):
        """
        Creates a list of dicts, with the index and distance of the k items most similars given an id.
        This is a protected method.

        Args
        ----
        name : str
            String with the name of a database in your JAI environment.

        idx_tem : int
            Index of the item the user is looking for.

        top_k : int
            Number of k similar items we want to return. `Default is 5`.

        Return
        ------
        response : dict
            Dictionary with the index and distance of `the k most similar items`.
        """

        if isinstance(id_item, list):
            pass
        elif isinstance(id_item, int):
            id_item = [id_item]
        else:
            raise TypeError(
                f"id_item param must be int or list, {type(id_item)} found.")

        response = requests.put(
            self.url + f"/similar/id/{name}?top_k={top_k}",
            headers=self.header,
            data=json.dumps(id_item),
        )

        if response.status_code == 200:
            return response.json()
        else:
            return self.assert_status_code(response)

    def _get_dtype(self, name):
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
        dtypes = self.info
        if self.is_valid(name):
            return dtypes.loc[dtypes["db_name"] == name, "db_type"].values[0]
        else:
            raise ValueError(f"{name} is not a valid name.")

    def _similar_json(self, name: str, data_json, top_k: int = 5):
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

        Return
        ------
        response : dict
            Dictionary with the index and distance of `the k most similar items`.
        """
        url = self.url + f"/similar/data/{name}?top_k={top_k}"

        response = requests.put(url, headers=self.header, data=data_json)
        if response.status_code == 200:
            return response.json()
        else:
            return self.assert_status_code(response)

    def _check_dtype_and_clean(self, data, db_type):
        """
        Check data type and remove NAs from the data.
        This is a protected method.

        Args
        ----
        data : pandas.DataFrame or pandas.Series
            Data to be checked and cleaned.

        db_type : str
            Database type (Supervised, Unsupervised, Text...)

        Return
        ------
        data : pandas.DataFrame or pandas.Series
            Data without NAs
        """
        if isinstance(data, (list, np.ndarray)):
            data = pd.Series(data)
        elif not isinstance(data, (pd.Series, pd.DataFrame)):
            raise TypeError(f"Inserted data is of type {type(data)},\
 but supported types are list, np.ndarray, pandas.Series or pandas.DataFrame")
        if db_type in [
                PossibleDtypes.text,
                PossibleDtypes.fasttext,
                PossibleDtypes.edit,
        ]:
            data = data.dropna()
        else:
            cols_to_drop = []
            for col in data.select_dtypes(include="category").columns:
                if data[col].nunique() > 1024:
                    cols_to_drop.append(col)
            data = data.dropna(subset=cols_to_drop)
        return data

    def predict(self,
                name: str,
                data,
                predict_proba: bool = False,
                batch_size: int = 16384):
        """
        Predict the output of new data for a given database.

        Args
        ----
        name : str
            String with the name of a database in your JAI environment.
        data : list, pd.Series or pd.DataFrame
            Data to be queried for similar inputs in your database.
        predict_proba : bool
            Whether or not to return the probabilities of each prediction is
            it's a classification. `Default is False`.
        batch_size : int
            Size of batches to send the data. `Default is 16384`.

        Return
        ------
        results : list of dicts
            List of predictions for the data passed as parameter.

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
        dtype = self._get_dtype(name)
        if dtype != "Supervised":
            raise ValueError("predict is only available to dtype Supervised.")

        results = []
        for i in trange(0, len(data), batch_size, desc="Predict"):
            if isinstance(data, (pd.Series, pd.DataFrame)):
                _batch = data.iloc[i:i + batch_size]
            else:
                _batch = data[i:i + batch_size]
            res = self._predict(name,
                                data2json(_batch, dtype=dtype),
                                predict_proba=predict_proba)
            results.extend(res)
        return results

    def _predict(self, name: str, data_json, predict_proba: bool = False):
        """
        Predict the output of new data for a given database by calling its
        respecive API method. This is a protected method.

        Args
        ----
        name : str
            String with the name of a database in your JAI environment.
        data_json : JSON file (dict)
            Data to be queried for similar inputs in your database.
        predict_proba : bool
            Whether or not to return the probabilities of each prediction. `Default is False`.

        Return
        -------
        results : dict
            Dictionary of predctions for the data passed as parameter.
        """
        url = self.url + \
            f"/predict/{name}?predict_proba={predict_proba}"

        response = requests.put(url, headers=self.header, data=data_json)
        if response.status_code == 200:
            return response.json()
        else:
            return self.assert_status_code(response)

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
        >>> j = Jai(AUTH_KEY)
        >>> ids = j.ids(name)
        >>> print(ids)
        ['891 items from 0 to 890']
        """
        response = requests.get(self.url + f"/id/{name}?mode={mode}",
                                headers=self.header)
        if response.status_code == 200:
            return response.json()
        else:
            return self.assert_status_code(response)

    def is_valid(self, name: str):
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

        Example
        -------
        >>> name = 'chosen_name'
        >>> j = Jai(AUTH_KEY)
        >>> check_valid = j.is_valid(name)
        >>> print(check_valid)
        True
        """
        response = requests.get(self.url + f"/validation/{name}",
                                headers=self.header)
        if response.status_code == 200:
            return response.json()["value"]
        else:
            return self.assert_status_code(response)

    def _temp_ids(self, name: str, mode: Mode = "simple"):
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
        response = requests.get(self.url + f"/setup/ids/{name}?mode={mode}",
                                headers=self.header)
        if response.status_code == 200:
            return response.json()
        else:
            return self.assert_status_code(response)

    def _insert_data(self, data, name, db_type, batch_size):
        """
        Insert raw data for training. This is a protected method.

        Args
        ----------
        name : str
            String with the name of a database in your JAI environment.
        db_type : str
            Database type (Supervised, Unsupervised, Text...)
        batch_size : int
            Size of batch to send the data.

        Return
        ------
        insert_responses : dict
            Dictionary of responses for each batch. Each response contains
            information of whether or not that particular batch was successfully inserted.
        """
        insert_responses = {}
        for i, b in enumerate(
                trange(0, len(data), batch_size, desc="Insert Data")):
            _batch = data.iloc[b:b + batch_size]
            insert_responses[i] = self._insert_json(
                name, data2json(_batch, dtype=db_type))
        return insert_responses

    def _check_ids_consistency(self, name, data):
        """
        Check if inserted data is consistent with what we expect.
        This is mainly to assert that all data was properly inserted.

        Args
        ----
        name : str
            Database name.
        data : pandas.DataFrame or pandas.Series
            Inserted data.

        Return
        ------
        None or Exception
            If an inconsistency is found, an error is raised.
        """
        inserted_ids = self._temp_ids(name)
        if len(data) != int(inserted_ids[0].split()[0]):
            print(f"Found invalid ids: {inserted_ids[0]}")
            print(self.delete_raw_data(name))
            raise Exception(
                "Something went wrong on data insertion. Please try again.")

    def setup(self,
              name: str,
              data,
              db_type: str,
              batch_size: int = 16384,
              frequency_seconds: int = 10,
              **kwargs):
        """
        Insert data and train model. This is JAI's crème de la crème.

        Args
        ----
        name : str
            Database name.
        data : pandas.DataFrame or pandas.Series
            Data to be inserted and used for training.
        db_type : str
            Database type {Supervised, Unsupervised, Text, FastText, TextEdit, Image}
        batch_size : int
            Size of batch to insert the data.`Default is 16384 (2**14)`.
        frequency_seconds : int
            Time in between each check of status. `Default is 10`.
        **kwargs
            Parameters that should be passed as a dictionary in compliance with the
            API methods. In other words, every kwarg argument should be passed as if
            it were in the body of a POST method. **To check all possible kwargs in
            Jai.setup method, you can check the** `Setup kwargs`_ **section**.

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
        >>> j = Jai(AUTH_KEY)
        >>> _, setup_response = j.setup(name=name, data=data, db_type="Supervised", label={"task": "metric_classification", "label_name": "my_label"})
        >>> print(setup_response)
        {
            "Task": "Training",
            "Status": "Started",
            "Description": "Training of database chosen_name has started."
        }
        """

        # delete data reamains
        self.delete_raw_data(name)

        # make sure our data has the correct type and is free of NAs
        data = self._check_dtype_and_clean(data=data, db_type=db_type)

        # insert data
        insert_responses = self._insert_data(data=data,
                                             name=name,
                                             batch_size=batch_size,
                                             db_type=db_type)

        # check if we inserted everything we were supposed to
        self._check_ids_consistency(name=name, data=data)

        # train model
        setup_response = self._setup_database(name, db_type, **kwargs)

        if frequency_seconds >= 1:
            self.wait_setup(name=name, frequency_seconds=frequency_seconds)

        return insert_responses, setup_response

    def add_data(self,
                 name: str,
                 data,
                 batch_size: int = 16384,
                 frequency_seconds: int = 10):
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
        db_type = self._get_dtype(name)

        # make sure our data has the correct type and is free of NAs
        data = self._check_dtype_and_clean(data=data, db_type=db_type)

        # insert data
        insert_responses = self._insert_data(data=data,
                                             name=name,
                                             batch_size=batch_size,
                                             db_type=db_type)

        # check if we inserted everything we were supposed to
        self._check_ids_consistency(name=name, data=data)

        # add data per se
        add_data_response = self._append(name=name)

        if frequency_seconds >= 1:
            self.wait_setup(name=name, frequency_seconds=frequency_seconds)

        return insert_responses, add_data_response

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
        response = requests.patch(self.url + f"/data/{name}",
                                  headers=self.header)
        if response.status_code == 202:
            return response.json()
        else:
            return self.assert_status_code(response)

    def _insert_json(self, name: str, df_json):
        """
        Insert data in JSON format. This is a protected method.

        Args
        ----
        name : str
            String with the name of a database in your JAI environment.
        df_json : dict
            Data in JSON format.

        Return
        ------
        response : dict
            Dictionary with the API response.
        """
        response = requests.post(self.url + f"/data/{name}",
                                 headers=self.header,
                                 data=df_json)
        if response.status_code == 200:
            return response.json()
        else:
            return self.assert_status_code(response)

    def _check_kwargs(self, db_type, **kwargs):
        """
        Sanity checks in the keyword arguments.
        This is a protected method.

        Args
        ----
        db_type : str
            Database type (Supervised, Unsupervised, Text...)

        Return
        ------
        body: dict
            Body to be sent in the POST request to the API.
        """
        possible = ["hyperparams", "callback_url"]
        must = []
        if db_type == "Unsupervised":
            possible.extend([
                'num_process', 'cat_process', 'high_process', 'mycelia_bases'
            ])
        elif db_type == "Supervised":
            possible.extend([
                'num_process', 'cat_process', 'high_process', 'mycelia_bases',
                'label', 'split'
            ])
            must.extend(['label'])

        missing = [key for key in must if kwargs.get(key, None) is None]
        if len(missing) > 0:
            raise ValueError(f"missing arguments {missing}")

        body = {}
        flag = True
        for key in possible:
            val = kwargs.get(key, None)
            if val is not None:
                if flag:
                    print("Recognized setup args:")
                    flag = False
                print(f"{key}: {val}")
                body[key] = val

        body["db_type"] = db_type
        return body

    def _setup_database(self, name: str, db_type, overwrite=False, **kwargs):
        """
        Call the API method for database setup.
        This is a protected method.

        Args
        ----
        name : str
            String with the name of a database in your JAI environment.
        db_type : str
            Database type (Supervised, Unsupervised, Text...)
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
        body = self._check_kwargs(db_type=db_type, **kwargs)
        response = requests.post(
            self.url + f"/setup/{name}?overwrite={overwrite}",
            headers=self.header,
            data=json.dumps(body),
        )

        if response.status_code == 201:
            return response.json()
        else:
            return self.assert_status_code(response)

    def fields(self, name: str):
        """
        Get the table fields for a Supervised/Unsupervised database.

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
        dtype = self._get_dtype(name)
        if dtype != "Unsupervised" and dtype != "Supervised":
            raise ValueError(
                "'fields' method is only available to dtype Unsupervised and Supervised."
            )

        response = requests.get(self.url + f"/table/fields/{name}",
                                headers=self.header)
        if response.status_code == 200:
            return response.json()
        else:
            return self.assert_status_code(response)

    def wait_setup(self, name: str, frequency_seconds: int = 5):
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
        max_steps = None
        while max_steps is None:
            status = self.status[name]
            starts_at, max_steps = pbar_steps(status=status)
            time.sleep(1)

        step = starts_at
        aux = 0
        with tqdm(total=max_steps, desc="JAI is working") as pbar:
            while status['Status'] != 'Task ended successfully.':
                if status['Status'] == 'Something went wrong.':
                    raise BaseException(status['Description'])
                if (step == starts_at) and (aux == 0):
                    pbar.update(starts_at)
                else:
                    diff = step - starts_at
                    pbar.update(diff)
                    starts_at = step
                step, _ = pbar_steps(status=status, step=step)
                time.sleep(frequency_seconds)
                status = self.status[name]
                aux += 1
            if (starts_at != max_steps) and aux != 0:
                diff = max_steps - starts_at
                pbar.update(diff)
            elif (starts_at != max_steps) and aux == 0:
                pbar.update(max_steps)
        return status

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
        >>> j = Jai(AUTH_KEY)
        >>> j.delete_raw_data(name=name)
        'All raw data from database 'chosen_name' was deleted!'
        """
        response = requests.delete(self.url + f"/data/{name}",
                                   headers=self.header)
        if response.status_code == 200:
            return response.json()
        else:
            return self.assert_status_code(response)

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
        >>> j = Jai(AUTH_KEY)
        >>> j.delete_database(name=name)
        'Bombs away! We nuked database chosen_name!'
        """
        response = requests.delete(self.url + f"/database/{name}",
                                   headers=self.header)
        if response.status_code == 200:
            return response.json()
        else:
            return self.assert_status_code(response)

    def match(self,
              name: str,
              data_left,
              data_right,
              top_k: int = 20,
              overwrite=False):
        """
        Experimental

        Match two datasets with their possible equal values.

        Queries the data right to get the similar results in data left.

        Parameters
        ----------
        name: str
            String with the name of a database in your JAI environment.
        data_left, data_right : pd.Series
            data to be matched.

        Returns
        -------
        dict
            each key is the id from data_right and the value is a list of ids from data_left
            that match.

        Example
        -------
        >>> import pandas as pd
        >>> from jai.processing import process_similar
        >>>
        >>> j = Jai(AUTH_KEY)
        >>> results = j.match(name, data1, data2)
        >>> processed = process_similar(results, return_self=True)
        >>> pd.DataFrame(processed).sort_values('query_id')
        >>> # query_id is from data_right and id is from data_left
                 query_id           id     distance
           0            1            2         0.11
           1            2            1         0.11
           2            3          NaN          NaN
           3            4          NaN          NaN
           4            5            5         0.15
        """
        if name not in self.names or overwrite:
            nt = np.clip(np.round(len(data_left) / 10, -3), 1000, 10000)
            self.setup(
                name,
                data_left,
                db_type="TextEdit",
                overwrite=overwrite,
                hyperparams={"nt": nt},
            )
        return self.similar(name, data_right, top_k=top_k)

    def resolution(self, name: str, data, top_k: int = 20, overwrite=False):
        """
        Experimental

        Find possible duplicated values within the data.

        Parameters
        ----------
        name: str
            String with the name of a database in your JAI environment.
        data : pd.Series
            data to find duplicates.

        Returns
        -------
        dict
            each key is the id and the value is a list of ids that are duplicates.

        Example
        -------
        >>> import pandas as pd
        >>> from jai.processing import process_similar
        >>>
        >>> j = Jai(AUTH_KEY)
        >>> results = j.resolution(name, data)
        >>> processed = process_similar(results, return_self=True)
        >>> pd.DataFrame(processed).sort_values('query_id')
                 query_id           id     distance
           0            1            2         0.11
           1            2            1         0.11
           2            3          NaN          NaN
           3            4            5         0.15
        """
        if name not in self.names or overwrite:
            nt = np.clip(np.round(len(data) / 10, -3), 1000, 10000)
            self.setup(
                name,
                data,
                db_type="TextEdit",
                overwrite=overwrite,
                hyperparams={"nt": nt},
            )
        return self.similar(name, data.index, top_k=top_k)

    def fill(self, name: str, data, column: str, **kwargs):
        """
        Experimental

        Fills the column in data with the most likely value given the other columns.

        Parameters
        ----------
        name: str
            String with the name of a database in your JAI environment.
        data : pd.DataFrame
            data to fill NaN.
        column : str
            name of the column to be filled.
        **kwargs : TYPE
            Extra args for supervised model. See setup method.

        Returns
        -------
        list of dicts
            List of dicts with possible filling values for each id with column NaN.

        Example
        -------
        >>> import pandas as pd
        >>> from jai.processing import process_predict
        >>>
        >>> j = Jai(AUTH_KEY)
        >>> results = j.fill(name, data, COL_TO_FILL)
        >>> processed = process_predict(results)
        >>> pd.DataFrame(processed).sort_values('id')
                  id   sanity_prediction    confidence_level (%)
           0       1             value_1                    70.9
           1       4             value_1                    67.3
           2       7             value_1                    80.2
        """
        cat_threshold = kwargs.get("cat_threshold", 512)
        data = data.copy()
        vals = data[column].value_counts() < 2
        if vals.sum() > 0:
            eliminate = vals[vals].index.tolist()
            print(
                f"values {eliminate} from column {column} were removed for having less than 2 examples."
            )
            data.loc[data[column].isin(eliminate), column] = None

        mask = data[column].isna()
        train = data.loc[~mask].copy()
        test = data.loc[mask].drop(columns=[column])

        cat = train.select_dtypes(exclude="number")
        pre = cat.columns[cat.nunique() > cat_threshold].tolist()
        prep_bases = []
        for col in pre:
            id_col = "id_" + col
            origin = name + "_" + col
            origin = origin.lower().replace("-", "_").replace(" ", "_")[:35]
            train[id_col], test[id_col] = self.embedding(
                origin, train[col], test[col])
            prep_bases.append({"id_name": id_col, "db_parent": origin})
        train = train.drop(columns=pre)
        test = test.drop(columns=pre)

        if name not in self.names:
            label = {"task": "metric_classification", "label_name": column}
            split = {
                "type": "stratified",
                "split_column": column,
                "test_size": 0.2
            }
            mycelia_bases = kwargs.get("mycelia_bases", [])
            mycelia_bases.extend(prep_bases)
            self.setup(
                name,
                train,
                db_type="Supervised",
                hyperparams={"learning_rate": 0.001},
                label=label,
                split=split,
                **kwargs,
            )

        return self.predict(name, test, predict_proba=True)

    def sanity(
        self,
        name: str,
        data,
        data_validate=None,
        columns_ref: list = None,
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
        data_validate : TYPE, optional
            Data to be checked if is valid or not. The default is None.
        columns_ref : list, optional
            Columns that can have inconsistencies. As default we use all non numeric columns.
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
        >>> from jai.processing import process_predict
        >>>
        >>> j = Jai(AUTH_KEY)
        >>> results = j.sanity(name, data)
        >>> processed = process_predict(results)
        >>> pd.DataFrame(processed).sort_values('id')
                  id   sanity_prediction    confidence_level (%)
           0       1               Valid                    70.9
           1       4             Invalid                    67.3
           2       7             Invalid                    80.6
           3      13               Valid                    74.2
        """
        frac = kwargs.get("frac", 0.1)
        random_seed = kwargs.get("random_seed", 42)
        cat_threshold = kwargs.get("cat_threshold", 512)
        target = kwargs.get("target", "is_valid")

        SKIP_SHUFFLING = target in data.columns

        np.random.seed(random_seed)

        data = data.copy()
        if data_validate is not None:
            data_validate = data_validate.copy()

        cat = data.select_dtypes(exclude="number")
        pre = cat.columns[cat.nunique() > cat_threshold].tolist()
        if columns_ref is None:
            columns_ref = cat.columns.tolist()

        prep_bases = []
        for col in pre:
            id_col = "id_" + col
            origin = name + "_" + col
            origin = origin.lower().replace("-", "_").replace(" ", "_")[:35]
            if data_validate is not None:
                data[id_col], data_validate[id_col] = self.embedding(
                    origin, data[col], data_validate[col])
            else:
                data[id_col] = self.embedding(origin, data[col])

            prep_bases.append({"id_name": id_col, "db_parent": origin})

            if col in columns_ref:
                columns_ref.remove(col)
                columns_ref.append(id_col)

        data = data.drop(columns=pre)
        if data_validate is not None:
            data_validate = data_validate.drop(columns=pre)
            test = data_validate.copy()
        else:
            test = data.copy()

        if name not in self.names:
            if not SKIP_SHUFFLING:

                def change(options, original):
                    return np.random.choice(options[options != original])

                # get a sample of the data and shuffle it
                sample = []
                for c in columns_ref:
                    s = data.sample(frac=frac)
                    uniques = s[c].unique()
                    s[c] = [change(uniques, v) for v in s[c]]
                    sample.append(s)
                sample = pd.concat(sample)

                # set target column values
                sample[target] = "Invalid"

                # set index of samples with different values as data
                idx = np.arange(len(data) + len(sample))
                mask_idx = np.logical_not(np.isin(idx, data.index))
                sample.index = idx[mask_idx][:len(sample)]

                data[target] = "Valid"
                train = pd.concat([data, sample])

            label = {"task": "metric_classification", "label_name": target}
            split = {
                "type": "stratified",
                "split_column": target,
                "test_size": 0.2
            }
            mycelia_bases = kwargs.get("mycelia_bases", [])
            mycelia_bases.extend(prep_bases)

            self.setup(
                name,
                train,
                db_type="Supervised",
                hyperparams={"learning_rate": 0.001},
                label=label,
                split=split,
                **kwargs,
            )

        return self.predict(name, test, predict_proba=True)

    def embedding(
        self,
        name: str,
        train,
        test=None,
        db_type="FastText",
        hyperparams=None,
    ):
        """
        Experimental

        Quick embedding for high numbers of categories in columns.

        Parameters
        ----------
        name: str
            String with the name of a database in your JAI environment.
        train : pd.Series
            Data to train your text based model.
        test : pd.Series, optional
            Extra data do be added to the database.
        db_type : str, optional
            type of model to be trained. The default is 'FastText'.
        hyperparams: optional
            See setup documentation for the db_type used.

        Returns
        -------
        name : str
            name of the base where the data was embedded.

        """
        if isinstance(train, pd.Series):
            train = train.copy()
        else:
            raise ValueError("train must be a Series")
        n = len(train)
        if test is None:
            values, inverse = np.unique(train, return_inverse=True)
        else:
            if isinstance(test, pd.Series):
                train = train.copy()
            else:
                raise ValueError("test must be a Series")
            test = test.copy()
            values, inverse = np.unique(train.tolist() + test.tolist(),
                                        return_inverse=True)

        train.loc[:] = inverse[:n]
        i_train = np.unique(inverse[:n])
        settrain = pd.Series(values[i_train], index=i_train)

        if name not in self.names:
            self.setup(name,
                       settrain,
                       db_type=db_type,
                       hyperparams=hyperparams)
        else:
            missing = i_train[~np.isin(i_train, self.ids(name, "complete"))]
            if len(missing) > 0:
                self.add_data(name, settrain.loc[missing])

        if test is not None:
            test.loc[:] = inverse[n:]
            i_test = np.unique(inverse[n:])
            settest = pd.Series(values[i_test], index=i_test)
            missing = i_test[~np.isin(i_test, self.ids(name, "complete"))]
            if len(missing) > 0:
                self.add_data(name, settest.loc[missing])

            return train, test
        else:
            return train
