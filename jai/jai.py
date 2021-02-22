"""
--- jai_core.py ---

created by @dionisio
"""
import secrets
import json
import pandas as pd
import numpy as np
import requests
import time

from .auxiliar_funcs.utils_funcs import data2json
from .auxiliar_funcs.classes import PossibleDtypes, Mode
from pandas.api.types import is_integer_dtype
from tqdm import trange


class Jai():
    def __init__(self, auth_key: str, url=None):
        if url is None:
            self.base_api_url = 'https://mycelia.azure-api.net'
            self.header = {'Auth': auth_key}
        else:
            if url.endswith('/'):
                url = url[:-1]
            self.base_api_url = url
            self.header = {'company-key': auth_key}

    @property
    def names(self):
        """
        Retrieves collections already created for the provided Auth Key.

        Args
        ----------
        None.

        Return
        ----------
        List with the collections created so far.

        Examples
        ----------
        ```python
        >>> j.names
        ['jai_database', 'jai_unsupervised', 'jai_supervised']

        ```
        """
        response = requests.get(url=self.base_api_url +
                                '/info?mode=names', headers=self.header)
        if response.status_code == 200:
            return response.json()
        else:
            return self.assert_status_code(response)

    @property
    def info(self):
        """
        Get name and type of each database in your environment.

        Args
        ----------
        None.

        Return
        ----------
        `df`: pandas.DataFrame
            Pandas dataframe with name and type of each database in your environment.

        Examples
        ----------
        ```python
        >>> j.info
                                db_name       db_type
        0                  jai_database          Text
        1              jai_unsupervised  Unsupervised
        2                jai_supervised    Supervised
        ```
        """
        response = requests.get(url=self.base_api_url +
                                '/info?mode=complete', headers=self.header)
        if response.status_code == 200:
            df = pd.DataFrame(response.json()).rename({'db_name': 'name',
                                                       'db_type': 'type'})
            return df
        else:
            return self.assert_status_code(response)

    @property
    def status(self):
        """
        Get the status of your JAI environment when training.

        Args
        ----------
        None.

        Return
        ----------
        `response`: dict
            A `JSON` file with the current status of the training tasks.

        Examples
        ----------
        ```python
        >>> j.status
        {
            "Task": "Training",
            "Status": "Completed",
            "Description": "Training of database YOUR_DATABASE has ended."
        }
        ```
        """
        response = requests.get(
            self.base_api_url + '/status', headers=self.header)
        if response.status_code == 200:
            return response.json()
        else:
            return self.assert_status_code(response)

    def generate_name(self, length: int=8, prefix: str='', suffix: str=''):
        """
        Generate a random string. You can pass a prefix and/or suffix. In this case,
        the generated string will be a concatenation of `prefix + random + suffix`.

        Args
        ----------
        `length`: int [Optional]
            Length for the desired string. Default is 8.
        `prefix`: string [Optional]
            Prefix of your string. Default is empty.
        `suffix`: string [Optional]
            Suffix of your string. Default is empty.

        Return
        ----------
        `str`: a random string.

        Examples
        ----------
        ```python
        >>> j.generate_name()
        13636a8b
        >>> j.generate_name(length=16, prefix="company")
        companyb8bbd445d
        ```
        """
        len_prefix = len(prefix)
        len_suffix = len(suffix)

        if length <= len_prefix + len_suffix:
            raise ValueError(
                f"length {length} is should be larger than {len_prefix+len_suffix} for prefix and suffix inputed.")

        length -= (len_prefix + len_suffix)
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
        # raise ValueError(response.content)
        print(response.json())
        return response

    def similar(self, name: str, data, top_k: int=5, batch_size: int=16384):
        """
        Query a database in search for the `top_k` most similar entries for each
        input data passed as argument.

        Args
        ----------
        `name`: str
            String with the name of a database in your JAI environment.
        `data`: list, pd.Series or pd.DataFrame
            Data to be queried for similar inputs in your database.
        `top_k`: int [Optional]
            Number of k similar items that we want to return. Default is 5.
        `batch_size`: int [Optional]
            Size of batches to send the data. Default is 16384.

        Return
        -------
        results : dict
            Dictionary with the index and distance of the K most similar items.

        Examples
        ----------
        >>> name = 'chosen_name'
        >>> DATA_ITEM = # data in the format of the database
        >>> TOP_K = 3
        >>> jai = jAI(AUTH_KEY)
        >>> df_index_distance = jai.similar(name, DATA_ITEM, TOP_K)
        >>> print(pd.DataFrame(df_index_distance['similarity']))
        index  distance
        10007  0.0
        45568  6995.6
        8382   7293.2
        """
        dtypes = self.info
        if any(dtypes['db_name'] == name):
            dtype = dtypes.loc[dtypes['db_name'] == name, 'db_type'].values[0]
        else:
            raise ValueError()

        if isinstance(data, list):
            data = np.array(data)

        is_id = is_integer_dtype(data)

        results = []
        for i in trange(0, len(data), batch_size, desc="Similar"):
            if is_id:
                if isinstance(data, pd.Series):
                    _batch = data.iloc[i:i+batch_size].tolist()
                if isinstance(data, pd.Index):
                    _batch = data[i:i+batch_size].tolist()
                else:
                    _batch = data[i:i+batch_size]
                res = self._similar_id(name, _batch, top_k=top_k)
            else:
                if isinstance(data, (pd.Series, pd.DataFrame)):
                    _batch = data.iloc[i:i+batch_size]
                else:
                    _batch = data[i:i+batch_size]
                res = self._similar_json(name, data2json(_batch, dtype=dtype),
                                        top_k=top_k)
            results.extend(res['similarity'])
        return results


    def _similar_id(self, name: str, id_item: int, top_k: int=5, method="PUT"):
        """
        Creates a list of dicts, with the index and distance of the k items most similars given an id.
        This is a protected method.

        Args
        ----------
        `name`: str 
            String with the name of a database in your JAI environment.

        `idx_tem`: int 
            Index of the item the user is looking for.

        `top_k`: int
            Number of k similar items we want to return.

        Return
        ----------
        `response`: dict
            Dictionary with the index and distance of the k most similar items.
        """
        if method == "GET":
            if isinstance(id_item, list):
                id_req = '&'.join(['id=' + str(i) for i in set(id_item)])
                url = self.base_api_url + \
                    f"/similar/id/{name}?{id_req}&top_k={top_k}"
            elif isinstance(id_item, int):
                url = self.base_api_url + \
                    f"/similar/id/{name}?id={id_item}&top_k={top_k}"
            else:
                raise TypeError(
                    f"id_item param must be int or list, {type(id_item)} found.")

            response = requests.get(url, headers=self.header)
        elif method == "PUT":
            if isinstance(id_item, list):
                pass
            elif isinstance(id_item, int):
                id_item = [id_item]
            else:
                raise TypeError(
                    f"id_item param must be int or list, {type(id_item)} found.")

            response = requests.put(self.base_api_url + \
                    f"/similar/id/{name}?top_k={top_k}", headers=self.header, data=json.dumps(id_item))
        else:
            raise ValueError("method must be GET or PUT.")
        
        if response.status_code == 200:
            return response.json()
        else:
            return self.assert_status_code(response)


    def _similar_json(self, name: str, data_json, top_k: int = 5):
        """
        Creates a list of dicts, with the index and distance of the k items most similars given a JSON data entry.
        This is a protected method

        Args
        ----------
        `name`: str 
            String with the name of a database in your JAI environment.

        `data_json`: dict (JSON) 
            Data in JSON format. Each input in the dictionary will be used to search for the `top_k` most
            similar entries in the database.

        `top_k`: int
            Number of k similar items we want to return.

        Return
        ----------
        `response`: dict
            Dictionary with the index and distance of the k most similar items.
        """
        url = self.base_api_url + f"/similar/data/{name}?top_k={top_k}"

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
        ----------
        `data`: pandas.DataFrame or pandas.Series
            Data to be checked and cleaned.

        `db_type`: str
            Database type (Supervised, Unsupervised, Text...)

        Return
        ----------
        `data`: pandas.DataFrame or pandas.Series
            Data without NAs
        """
        if isinstance(data, (list, np.ndarray)):
            data = pd.Series(data)
        elif not isinstance(data, (pd.Series, pd.DataFrame)):
            raise TypeError(f"Inserted data is of type {type(data)},\
 but supported types are list, np.ndarray, pandas.Series or pandas.DataFrame")
        if db_type in [PossibleDtypes.text, PossibleDtypes.fasttext, PossibleDtypes.edit]:
            data = data.dropna()
        else:
            cols_to_drop = []
            for col in data.select_dtypes(include='category').columns:
                if data[col].nunique() > 1024:
                    cols_to_drop.append(col)
            data = data.dropna(subset=cols_to_drop)
        return data

    def predict(self, name: str, data, predict_proba:bool=False, batch_size: int=16384):
        """
        Query a database in search for the `top_k` most similar entries for each
        input data passed as argument.

        Args
        ----------
        `name`: str
            String with the name of a database in your JAI environment.
        `data`: list, pd.Series or pd.DataFrame
            Data to be queried for similar inputs in your database.
        `predict_proba`: bool [Optional]
            Whether or not to return the probabilities of each prediction. Default is False.
        `batch_size`: int [Optional]
            Size of batches to send the data. Default is 16384.

        Return
        -------
        results : list
            List of predctions for the data passed as parameter.

        Examples
        ----------
        >>> name = 'chosen_name'
        >>> DATA_ITEM = # data in the format of the database
        >>> TOP_K = 3
        >>> jai = jAI(AUTH_KEY)
        >>> df_index_distance = jai.similar(name, DATA_ITEM, TOP_K)
        >>> print(pd.DataFrame(df_index_distance['similarity']))
        index  distance
        10007  0.0
        45568  6995.6
        8382   7293.2
        """
        dtypes = self.info
        if any(dtypes['db_name'] == name):
            dtype = dtypes.loc[dtypes['db_name'] == name, 'db_type'].values[0]
            if dtype != "Supervised":
                raise ValueError("predict is only available to dtype Supervised.")
        else:
            raise ValueError(f"{name} is not a valid name.")

        results = []
        for i in trange(0, len(data), batch_size, desc="Similar"):
            if isinstance(data, (pd.Series, pd.DataFrame)):
                _batch = data.iloc[i:i+batch_size]
            else:
                _batch = data[i:i+batch_size]
            res = self._predict(name, data2json(_batch, dtype=dtype),
                                predict_proba=predict_proba)
            results.extend(res)
        return results


    def _predict(self, name: str, data_json, predict_proba:bool=False):
        url = self.base_api_url + f"/predict/{name}?predict_proba={predict_proba}"

        response = requests.put(url, headers=self.header, data=data_json)
        if response.status_code == 200:
            return response.json()
        else:
            return self.assert_status_code(response)


    def ids(self, name: str, mode: Mode = 'simple'):
        response = requests.get(
            self.base_api_url + f'/id/{name}?mode={mode}', headers=self.header)
        if response.status_code == 200:
            return response.json()
        else:
            return self.assert_status_code(response)

    def is_valid(self, name: str):
        response = requests.get(
            self.base_api_url + f'/validation/{name}', headers=self.header)
        if response.status_code == 200:
            return response.json()['value']
        else:
            return self.assert_status_code(response)

    def _temp_ids(self, name: str, mode: Mode='simple'):
        response = requests.get(
            self.base_api_url + f'/setup/ids/{name}?mode={mode}', headers=self.header)
        if response.status_code == 200:
            return response.json()
        else:
            return self.assert_status_code(response)

    def _insert_data(self, data, name, db_type, batch_size):
        insert_responses = {}
        for i, b in enumerate(trange(0, len(data), batch_size, desc="Insert Data")):
            _batch = data.iloc[b:b+batch_size]
            insert_responses[i] = self._insert_json(name,
                                                    data2json(_batch, dtype=db_type))
        return insert_responses

    def _check_ids_consistency(self, data, name):
        inserted_ids = self._temp_ids(name)
        if len(data) != int(inserted_ids[0].split()[0]):
            print(f"Found invalid ids: {inserted_ids[0]}")
            print(self.delete_raw_data(name))
            raise Exception("Something went wrong on data insertion. Please try again.")

    def setup(self, name: str, data, db_type: str, batch_size: int = 16384, **kwargs):
        # make sure our data has the correct type and is free of NAs
        data = self._check_dtype_and_clean(data=data, db_type=db_type)

        # insert data
        insert_responses = self._insert_data(data=data, name=name, batch_size=batch_size, db_type=db_type)

        # check if we inserted everything we were supposed to
        self._check_ids_consistency(data=data, name=name)

        # train model
        setup_response = self._setup_database(name, db_type, **kwargs)
        return insert_responses, setup_response

    def add_data(self, name: str, data, db_type: str, batch_size: int = 16384):
        insert_responses = {}
        for i, b in enumerate(trange(0, len(data), batch_size, desc="Insert Data")):
            if isinstance(data, (pd.Series, pd.DataFrame)):
                _batch = data.iloc[b:b+batch_size]
            else:
                _batch = data[b:b+batch_size]
            insert_responses[i] = self._insert_json(name,
                                                    data2json(_batch, dtype=db_type))

        inserted_ids = self._temp_ids(name, 'simple')
        if len(data) != int(inserted_ids[0].split()[0]):
            print(f"Found invalid ids: {inserted_ids[0]}")
            print(self.delete_raw_data(name))
            raise Exception("Something went wrong on data insertion. Please try again.")

        setup_response = self._append(name)
        return insert_responses, setup_response

    def _insert_json(self, name: str, df_json):
        response = requests.post(self.base_api_url + f'/data/{name}',
                                 headers=self.header, data=df_json)
        if response.status_code == 200:
            return response.json()
        else:
            return self.assert_status_code(response)

    def _check_kwargs(self, db_type, **kwargs):
        possible = ['hyperparams', 'callback_url']
        must = []
        if db_type == "Unsupervised":
            possible.extend(['num_process', 'cat_process',  'high_process',
                             'mycelia_bases'])
        elif db_type == "Supervised":
            possible.extend(['num_process', 'cat_process',  'high_process',
                             'mycelia_bases', 'label', 'split'])
            must.extend(['label', 'split'])

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

        body['db_type'] = db_type
        return body


    def _setup_database(self, name: str, db_type, overwrite=False, **kwargs):
        body = self._check_kwargs(db_type=db_type, **kwargs)
        response = requests.post(self.base_api_url + f'/setup/{name}?overwrite={overwrite}',
                                 headers=self.header, data=json.dumps(body))

        if response.status_code == 201:
            return response.json()
        else:
            return self.assert_status_code(response)

    def fields(self, name: str):
        dtypes = self.info
        if any(dtypes['db_name'] == name):
            dtype = dtypes.loc[dtypes['db_name'] == name, 'db_type'].values[0]
            if dtype != "Unsupervised" and dtype != "Supervised":
                raise ValueError("predict is only available to dtype Unsupervised and Supervised.")
        else:
            raise ValueError(f"{name} is not a valid name.")
        response = requests.get(self.base_api_url + f'/table/fields/{name}',
                                headers=self.header)
        if response.status_code == 200:
            return response.json()
        else:
            return self.assert_status_code(response)

    def wait_setup(self, name: str, frequency_seconds:int=5):
        status = self.status
        if name in status.keys():
            status = status[name]
            while status['Status'] != 'Task ended successfully.':
                if status['Status'] == 'Something went wrong.':
                    raise BaseException(status['Description'])

                for x in range(int(frequency_seconds)*5):
                    for frame in r'-\|/-\|/':
                        print('\b', frame, sep='', end='', flush=True)
                        time.sleep(0.2)

                status = self.status
                if name in status.keys():
                    status = status[name]
                else:
                    break

    def _append(self, name: str):
        response = requests.patch(
            self.base_api_url + f'/data/{name}', headers=self.header)
        if response.status_code == 202:
            return response.json()
        else:
            return self.assert_status_code(response)

    def delete_raw_data(self, name: str):
        response = requests.delete(
            self.base_api_url + f'/data/{name}', headers=self.header)
        if response.status_code == 200:
            return response.json()
        else:
            return self.assert_status_code(response)

    def delete_database(self, name: str):
        response = requests.delete(
            self.base_api_url + f'/database/{name}', headers=self.header)
        if response.status_code == 200:
            return response.json()
        else:
            return self.assert_status_code(response)