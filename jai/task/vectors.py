import time
from fnmatch import fnmatch

import matplotlib.pyplot as plt
from tqdm import trange

from .base import TaskBase
from ..core.utils_funcs import data2json
from ..core.validations import check_response, check_dtype_and_clean
from pandas.api.types import is_numeric_dtype

from ..types.generic import PossibleDtypes
from ..types.hyperparams import InsertParams

from ..types.responses import InsertVectorResponse
from typing import Dict

__all__ = ["Trainer"]


def get_numbers(status):
    if fnmatch(status["Description"], "*Iteration:*"):
        curr_step, max_iterations = status["Description"].split(
            "Iteration: ")[1].strip().split(" / ")
        return int(curr_step), int(max_iterations)
    return False, 0, 0


class Vectors(TaskBase):
    """
    Base class for communication with the Mycelia API.

    Used as foundation for more complex applications for data validation such
    as matching tables, resolution of duplicated values, filling missing values
    and more.

    """

    def __init__(self,
                 name: str,
                 environment: str = "default",
                 env_var: str = "JAI_AUTH",
                 verbose: int = 1,
                 safe_mode: bool = False):
        """
        Initialize the Jai class.

        An authorization key is needed to use the Mycelia API.

        Parameters
        ----------

        Returns
        -------
            None

        """
        super(Vectors, self).__init__(name=name,
                                      environment=environment,
                                      env_var=env_var,
                                      verbose=verbose,
                                      safe_mode=safe_mode)

    def delete_raw_data(self):
        """
        Delete raw data. It is good practice to do this after training a model.


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
        response = self._delete_raw_data(self.name)
        if self.safe_mode:
            return check_response(str, response)
        return response

    def delete_database(self):
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
        response = self._delete_database(self.name)
        if self.safe_mode:
            return check_response(str, response)
        return response

    def insert_vectors(self,
                       data,
                       batch_size: int = 10000,
                       overwrite: bool = False,
                       append: bool = False):
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

        if self.is_valid:
            if overwrite:
                create_new_collection = True
                self.delete_database()
            elif not overwrite and append:
                create_new_collection = False
            else:
                raise KeyError(
                    f"Database '{self.name}' already exists in your environment."
                    f"Set overwrite=True to overwrite it or append=True to add new data to your database."
                )
        else:
            # delete data remains
            create_new_collection = True
            self.delete_raw_data()

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
        for i, b in enumerate(
                trange(0, len(data), batch_size, desc="Insert Vectors")):
            _batch = data.iloc[b:b + batch_size]
            data_json = data2json(_batch,
                                  dtype=PossibleDtypes.vector,
                                  predict=False)
            if i == 0 and create_new_collection is True:
                response = self._insert_vectors_json(self.name,
                                                     data_json,
                                                     overwrite=True)
            else:
                response = self._insert_vectors_json(self.name,
                                                     data_json,
                                                     overwrite=False)

            if self.safe_mode:
                response = check_response(InsertVectorResponse, response)
            insert_responses[i] = response

        return insert_responses