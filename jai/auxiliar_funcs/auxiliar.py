"""
--- auxiliar.py ---

created by @rogerioguicampos
"""
from azure.storage.blob import BlobServiceClient
import numpy as np
import re


def load_npy_from_stream(stream_) -> np.ndarray:
    """Experimental, may not work!

    Args
    ----------
    stream_: io.BytesIO() object obtained by e.g. calling BlockBlobService().get_blob_to_stream()
    containingthe binary stream of a standard format .npy file.

    Return
    ----------
    array (np.ndarray):

    Examples
    ----------
    """
    stream_.seek(0)
    prefix_ = stream_.read(128)  # first 128 bytes seem to be the metadata
    dict_string = re.search('\{(.*?)\}', prefix_[1:].decode())[0]
    metadata_dict = eval(dict_string)
    array = np.frombuffer(stream_.read(), dtype=metadata_dict['descr']).reshape(
        metadata_dict['shape'])
    return array


def connect_azure_blob_storage(db_name: str, company_id: str, conn_str: str):
    """

    Args
    ----------

    Return
    ----------

    Examples
    ----------

    """
    blob_service_client = BlobServiceClient.from_connection_string(
        conn_str=conn_str)
    container_name = company_id + '-' + db_name
    container_client = blob_service_client.get_container_client(container_name)

    blob_list = container_client.list_blobs(
        name_starts_with="high_dimensional_vectors/")
    blobs = []

    for blob in blob_list:
        blobs.append(blob)
    blob = blobs[-1]
    blob_client = blob_service_client.get_blob_client(
        container=container_name, blob=blob)

    return blob_client
