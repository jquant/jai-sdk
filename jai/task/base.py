import numpy as np
from io import BytesIO
from ..core.base import BaseJai
from ..core.validations import check_response
import requests
from ..types.generic import Mode

from ..types.responses import (UserResponse, ValidResponse)

from typing import Dict, List, Any
from pydantic import HttpUrl
import sys
if sys.version < '3.8':
    from typing_extensions import Literal
else:
    from typing import Literal

__all__ = ["Trainer"]


class TaskBase(BaseJai):
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
        super(TaskBase, self).__init__(environment, env_var)
        self.safe_mode = safe_mode

        if self.safe_mode:
            user = self._user()
            user = check_response(UserResponse, user).dict()

            if verbose:
                user_print = '\n'.join(
                    [f"- {k}: {v}" for k, v in user.items()])
                print(f"Connection established.\n{user_print}")

        self.name = name

    @property
    def name(self):
        return self._name

    @property
    def db_type(self):
        if self.is_valid():
            return self.describe()['dtype']
        return None

    @name.setter
    def name(self, value):
        self._name = value

    def is_valid(self):
        """
        Check if a given name is a valid database name (i.e., if it is
        in your environment).

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
        valid = self._is_valid(self.name)
        if self.safe_mode:
            return check_response(ValidResponse, valid).dict()["value"]
        return valid["value"]

    def describe(self):
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
        description = self._describe(self.name)
        if self.safe_mode:
            return check_response(None, description)  # TODO Validator
        return description

    def fields(self):
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

        Example
        -------
        >>> name = 'chosen_name'
        >>> j = Jai(AUTH_KEY)
        >>> fields = j.fields(name=name)
        >>> print(fields)
        {'id': 0, 'feature1': 0.01, 'feature2': 'string', 'feature3': 0}
        """
        fields = self._fields(self.name)
        if self.safe_mode:
            return check_response(
                Dict[str, Literal["int32", "int64", "float32", "float64",
                                  "string", "embedding", "label", "datetime"]],
                fields)
        return fields

    def download_vectors(self):
        """
        Download vectors from a particular database.

        Args
        ----
        name : str
            String with the name of a database in your JAI environment.

        Return
        ------
        vector : np.array
            Numpy array with all vectors.

        Example
        -------
        >>> name = 'chosen_name'
        >>> j = Jai(AUTH_KEY)
        >>> vectors = j.download_vectors(name=name)
        >>> print(vectors)
        [[ 0.03121682  0.2101511  -0.48933393 ...  0.05550333  0.21190546  0.19986008]
        [-0.03121682 -0.21015109  0.48933393 ...  0.2267401   0.11074653  0.15064166]
        ...
        [-0.03121682 -0.2101511   0.4893339  ...  0.00758727  0.15916921  0.1226602 ]]
        """
        url = self._download_vectors(self.name)
        if self.safe_mode:
            url = check_response(HttpUrl, url)
        r = requests.get(url)
        return np.load(BytesIO(r.content))

    def filters(self):
        """
        Gets the valid values of filters.

        Return
        ------
        response : list of strings
            List of valid filter values.
        """
        filters = self._filters(self.name)
        if self.safe_mode:
            return check_response(List[str], filters)
        return filters

    def ids(self, mode: Mode = "complete"):
        """
        Get id information of a given database.

        Args
        mode : str, optional

        Return
        -------
        response: list
            List with the actual ids (mode: 'complete') or a summary of ids
            ('simple') of the given database.

        Example
        ----------
        >>> name = 'chosen_name'
        >>> j = Jai(AUTH_KEY)
        >>> ids = j.ids(name)
        >>> print(ids)
        ['891 items from 0 to 890']
        """
        ids = self._ids(self.name, mode)
        if self.safe_mode:
            return check_response(List[Any], ids)
        return ids