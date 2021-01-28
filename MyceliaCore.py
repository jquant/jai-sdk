import io
import json
import pandas as pd
import requests
from brain_plasma import Brain
from azure.storage.blob import BlobServiceClient
import re
import numpy as np


def _load_npy_from_stream(_stream):
    """Experimental, may not work!
    :param stream_: io.BytesIO() object obtained by e.g. calling BlockBlobService().get_blob_to_stream() containing
        the binary stream of a standard format .npy file.
    :return: numpy.ndarray
    """
    _stream.seek(0)
    prefix_ = _stream.read(128)  # first 128 bytes seem to be the metadata
    dict_string = re.search('\{(.*?)\}', prefix_[1:].decode())[0]
    metadata_dict = eval(dict_string)
    array = np.frombuffer(_stream.read(), dtype=metadata_dict['descr']).reshape(metadata_dict['shape'])
    return array

class Mycelia():
    def __init__(self, auth_key: str):
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
    
    def get_collection_to_memory(self, db_name: str):
            
        """Downloads mycelia collections into memory

        Args
        ----------
        db_name (str): string with the name of the database you created on the mycelia platform.
        
        header (dict): dict with the authentication key from mycelia platform. Example {'Auth': 'auth_key_mycelia'}.

        Return
        ----------
        collections_json (json): dict with the collections created so far

        Examples
        ----------

        """
        #TODO - get company id from Azure AD Login
        company_id = 'z5fb6c8fe89881ad5840ec145'
        conn_string = "DefaultEndpointsProtocol=https;AccountName=jquantappstorage;AccountKey=7rEO7wy16hwgbdtMPNoobchlk/BHeF4lyeIZTAO8jmvZLyDwp/rEUmGKuhH7cHlQHXQr6uWyTOsKsmRpWPRQJA==;EndpointSuffix=core.windows.net"
        blob_service_client = BlobServiceClient.from_connection_string(conn_str=conn_string)
        container_name = company_id+'-'+db_name
        container_client = blob_service_client.get_container_client(container_name)
       #generator = blob_service.list_blobs(container_client, prefix="high_dimensional_vectors/")
        blob_list = container_client.list_blobs(name_starts_with="high_dimensional_vectors/")
        blobs = []
        for blob in blob_list:
            blobs.append(blob)    
        blob = blobs[-1]
        blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob)
        try:
            stream = io.BytesIO()
            downloader = blob_client.download_blob()
            info = downloader.download_to_stream(stream)
            
        except ResourceNotFoundError:
            print("No blob found.")
        
        arr = _load_npy_from_stream(stream)
        self.brain[db_name] = arr
        
        return True


    def list_k_similar_distance(self, db_name: str, idx_item: int, top_k: int=5) -> pd.DataFrame:
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
