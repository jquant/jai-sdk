"""
--- mycelia_core.py ---

created by @dionisio
"""
from auxiliar_funcs import auxiliar
from azure.storage.blob import BlobServiceClient
from brain_plasma import Brain
import io
import json
import numpy as np
import pandas as pd
import requests


class Mycelia():
    """
    """
    base_api_url = 'https://mycelia.azure-api.net'
    brain = Brain(path='/mnt/socket/plasma')

    def __init__(self, auth_key: str, company_id: str, conn_str: str):
        self.header = {'Auth': auth_key}
        self.company_id = company_id
        self.conn_str = conn_str 

        
    def get_databases(self):
        """Retrieves collections already created for the provided Auth Key.

        Args
        ----------
        header (dict): dict with the authentication key from mycelia platform. Example {'Auth': 'auth_key_mycelia'}.

        Return
        ----------
        collections_json (json): dict with the collections created so far.

        Examples
        ----------

        """
        r = requests.get(url=self.base_api_url+f'/info', headers=self.header)
        databases_json = sorted(r.json())
        return databases_json
    

    def get_collection_to_memory(self, db_name: str) -> np.ndarray: 
        """Downloads mycelia collections into memory

        Args
        ----------
        db_name (str): string with the name of the database you created on the mycelia platform.
        
        Return
        ----------
        collections_json (json): dict with the collections created so far

        Examples
        ----------

        """
        blob_client = auxiliar.connect_azure_blob_storage(db_name=db_name, company_id=self.company_id, conn_str=self.conn_str)

        try:
            stream = io.BytesIO()
            downloader = blob_client.download_blob()
            info = downloader.download_to_stream(stream)
            
        except ResourceNotFoundError:
            print("No blob found.")
        
        arr = auxiliar.load_npy_from_stream(stream)
        brain[db_name] = arr
        
        return True


    def list_k_similar_distance(self, db_name: str, id_item: int, top_k: int=5) -> pd.DataFrame:
        """Creates a list of dicts, with the index and distance of the k itens most similars.

        Args
        ----------
        db_name (str): string with the name of the database you created on the mycelia platform.

        idx_tem (int): index of the item the customer is looking for at the moment.

        top_k (int): number of k similar items that we want to return.

        Return
        ----------
        df_index_distance (pd.DataFrame): dataframe with the index and distance of the k most similar items.

        Examples
        ----------
        >>> DB_NAME = 'chosen_name'
        >>> ID_ITEM = 10007
        >>> TOP_K = 3
        >>> df_index_distance = list_k_similar_distance(DB_NAME, ID_ITEM, TOP_K)
        >>> print(df_index_distance)
        index  distance
        10007  0.0
        45568  6995.6
        8382   7293.2
        """
        url = self.base_api_url + f"/similar/id/{db_name}?id={id_item}&top_k={top_k}"
        dict_result_query = requests.get(url, headers=self.header).json()
        # df_index_distance = pd.DataFrame(dict_result_query['similarity'])
        return dict_result_query# df_index_distance, dict_result_query 
