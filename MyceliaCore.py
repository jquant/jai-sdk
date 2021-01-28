import json
import pandas as pd
import requests
from brain_plasma import Brain

class Mycelia():
    def __init__(self, auth_key):
        self.header = {'Auth': auth_key}
        self.base_api_url = 'https://mycelia.azure-api.net'
        self.brain = Brain(path='/mnt/socket/plasma')

    def get_databases(self):
        """Retrieves collections already created for the provided Auth Key

        Args
        ----------
        header (dict): dict with the authentication key from mycelia platform. Example {'Auth': 'auth_key_mycelia'}.

        Return
        ----------
        collections_json (json): dict with the collections created so far

        Examples
        ----------

        """
        #r = requests.get(url= base_api_url+f'info?mode=names', headers=header)
        r = requests.get(url= self.base_api_url+f'/names', headers=self.header)
        collections_json = r.json()
        return sorted(collections_json)


    def list_k_similar_distance(self, db_name: str, idx_item: int, header: dict, top_k: int=5) -> pd.DataFrame:
        """Creates a list of dicts, with the index and distance of the k itens most similars.

        Args
        ----------
        db_name (str): string with the name of the database you created on the mycelia platform.

        idx_tem (int): index of the item the customer is looking for at the moment.

        header (dict): dict with the authentication key from mycelia platform. Example {'Auth': 'auth_key_mycelia'}.

        top_k (int): number of k similar items that we want to return.

        Return
        ----------
        df_index_distance (pd.DataFrame): dataframe with the index and distance of the k most similar items.

        Examples
        ----------
        >>> DB_NAME = 'chosen_name'
        >>> IDX_ITEM = 10007
        >>> HEADER = {'Auth': 'auth_key_mycelia'}
        >>> TOP_K = 3
        >>> df_index_distance = list_k_similar_distance(DB_NAME, IDX_ITEM, HEADER, TOP_K)
        >>> print(df_index_distance)
        index  distance
        10007  0.0
        45568  6995.6
        8382   7293.2
        """
        url = self.base_api_url + f"/similar/id/{db_name}?id={idx_item}&top_k={top_k}"
        dict_result_query = requests.get(url, headers=self.header).json()
        df_index_distance = pd.DataFrame(dict_result_query['similarity'][0]['results'])
        return df_index_distance, dict_result_query 
