"""
--- mycelia_core.py ---

created by @dionisio
"""
import secrets
import json
import pandas as pd
import requests
import time
from tqdm import trange

from auxiliar_funcs.utils_funcs import data2json


class Mycelia():
    """
    """

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
        """Retrieves collections already created for the provided Auth Key.

        Args
        ----------
        header (dict): dict with the authentication key from mycelia platform. Example {'Auth': 'auth_key_mycelia'}.

        Return
        ----------
        collections_json (list): list with the collections created so far.

        Examples
        ----------

        """
        response = requests.get(url=self.base_api_url +
                                '/info?mode=names', headers=self.header)
        if response.status_code == 200:
            return response.json()
        else:
            return self.assert_status_code(response)

    @property
    def info(self):
        """Retrieves collections already created for the provided Auth Key.

        Args
        ----------
        header (dict): dict with the authentication key from mycelia platform. Example {'Auth': 'auth_key_mycelia'}.

        Return
        ----------
        collections_json (list): list with the collections created so far.

        Examples
        ----------

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
        response = requests.get(
            self.base_api_url + '/status', headers=self.header)
        if response.status_code == 200:
            return response.json()
        else:
            return self.assert_status_code(response)

    def generate_name(self, length=8, prefix='', suffix=''):
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

    def similar_list(self, name, list_id, top_k=5, batch_size=1024):
        results = []
        for i in trange(0, len(list_id), batch_size, desc="Similar List Id"):
            _list = list_id[i:i+batch_size].tolist()
            res = self.similar_id(name, _list, top_k=top_k)
            results.extend(res['similarity'])
        return results

    def similar_id(self, name: str, id_item: int, top_k: int = 5):
        """Creates a list of dicts, with the index and distance of the k itens most similars.

        Args
        ----------
        name (str): string with the name of the database you created on the mycelia platform.

        idx_tem (int): index of the item the customer is looking for at the moment.

        top_k (int): number of k similar items that we want to return.

        Return
        ----------
        df_index_distance (dict): dataframe with the index and distance of the k most similar items.

        Examples
        ----------
        >>> name = 'chosen_name'
        >>> ID_ITEM = 10007
        >>> TOP_K = 3
        >>> mycelia = Mycelia(AUTH_KEY)
        >>> df_index_distance = mycelia.similar_id(name, ID_ITEM, TOP_K)
        >>> print(pd.DataFrame(df_index_distance['similarity']))
        index  distance
        10007  0.0
        45568  6995.6
        8382   7293.2
        """
        if isinstance(id_item, list):
            id_req = '&'.join(['id=' + str(i) for i in id_item])
            url = self.base_api_url + \
                f"/similar/id/{name}?{id_req}&top_k={top_k}"
        elif isinstance(id_item, int):
            url = self.base_api_url + \
                f"/similar/id/{name}?id={id_item}&top_k={top_k}"
        else:
            raise TypeError(
                f"id_item param must be int or list, {type(id_item)} found.")

        response = requests.get(url, headers=self.header)
        if response.status_code == 200:
            return response.json()
        else:
            return self.assert_status_code(response)

    def similar_data(self, name, data, top_k=5, batch_size=1024):
        results = []
        for i in trange(0, len(data), batch_size, desc="Similar Data"):
            if isinstance(data, (pd.Series, pd.DataFrame)):
                _batch = data.iloc[i:i+batch_size]
            else:
                _batch = data[i:i+batch_size]
            res = self.similar_json(name, data2json(_batch), top_k=top_k)
            results.extend(res['similarity'])
        return results

    def similar_json(self, name: str, data_json, top_k: int = 5):
        url = self.base_api_url + f"/similar/data/{name}?top_k={top_k}"

        response = requests.put(url, headers=self.header, data=data_json)
        if response.status_code == 200:
            return response.json()
        else:
            return self.assert_status_code(response)

    def ids(self, name, mode='summarized'):
        response = requests.get(
            self.base_api_url + f'/id/{name}?mode={mode}', headers=self.header)
        if response.status_code == 200:
            return response.json()
        else:
            return self.assert_status_code(response)

    def is_valid(self, name):
        response = requests.get(
            self.base_api_url + f'/validation/{name}', headers=self.header)
        if response.status_code == 200:
            return response.json()
        else:
            return self.assert_status_code(response)

    def inserted_ids(self, name, mode='summarized'):
        response = requests.get(
            self.base_api_url + f'/setup/ids/{name}?mode={mode}', headers=self.header)
        if response.status_code == 200:
            return response.json()
        else:
            return self.assert_status_code(response)

    def insert_data(self, name, data, batch_size=1024):
        for i in trange(0, len(data), batch_size, desc="Insert Data"):
            if isinstance(data, (pd.Series, pd.DataFrame)):
                _batch = data.iloc[i:i+batch_size]
            else:
                _batch = data[i:i+batch_size]
            self.insert_json(name, data2json(_batch))

    def insert_json(self, name, df_json):
        response = requests.post(self.base_api_url + f'/data/{name}',
                                 headers=self.header, data=df_json)
        if response.status_code == 200:
            return response.json()
        else:
            return self.assert_status_code(response)

    def setup_database(self, name, **kwargs):
        response = requests.post(self.base_api_url + f'/setup/{name}',
                                 headers=self.header, data=json.dumps(kwargs))
        if response.status_code == 201:
            return response.json()
        else:
            return self.assert_status_code(response)

    def wait_setup(self, frequency_seconds=5):
        status = self.status
        while status['Status'] != 'Task ended successfully.':
            if status['Status'] == 'Something went wrong.':
                raise BaseException(status['Description'])
            time.sleep(frequency_seconds)
            status = self.status

    def append_data(self, name):
        response = requests.patch(
            self.base_api_url + f'/data/{name}', headers=self.header)
        if response.status_code == 202:
            return response.json()
        else:
            return self.assert_status_code(response)

    def delete_raw_data(self, name):
        response = requests.delete(
            self.base_api_url + f'/data/{name}', headers=self.header)
        if response.status_code == 200:
            return response.json()
        else:
            return self.assert_status_code(response)

    def delete_database(self, name):
        response = requests.delete(
            self.base_api_url + f'/database/{name}', headers=self.header)
        if response.status_code == 200:
            return response.json()
        else:
            return self.assert_status_code(response)
