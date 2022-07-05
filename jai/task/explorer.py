from typing import List

import pandas as pd

from ..core.base import BaseJai
from ..core.validations import check_response
from ..types.responses import (
    DescribeResponse,
    EnvironmentsResponse,
    InfoResponse,
    InfoSizeResponse,
    UserResponse,
)

__all__ = ["Explorer"]


class Explorer(BaseJai):
    """
    Explorer task class.

    This is a general class for managing databases on Jai API.

    An authorization key is needed to use the Jai API.

    Parameters
    ----------
    environment : str
        Jai environment id or name to use. Defaults to "default"
    env_var : str
        Name of the Environment Variable to get the value of your auth key.
        Defaults to "JAI_AUTH".
    safe_mode : bool
        When safe_mode is True, responses from Jai API are validated.
        If the validation fails, the current version you are using is probably incompatible with the current API version.
        We advise updating it to a newer version. If the problem persists and you are on the latest SDK version, please open an issue so we can work on a fix.
        Defaults to False.

    """

    def __init__(
        self,
        environment: str = "default",
        env_var: str = "JAI_AUTH",
        safe_mode: bool = False,
    ):
        super(Explorer, self).__init__(environment, env_var)
        self.safe_mode = safe_mode

    @property
    def names(self):
        """
        Retrieves databases already created for the provided Auth Key.

        Return
        ------
            List with the sorted names of the databases created so far.

        Example
        -------
        >>> from jai import Explorer
        ...
        >>> explorer = Explorer()
        >>> explorer.names
        """
        names = self._info(mode="names")
        if self.safe_mode:
            names = check_response(List[str], names)
        return sorted(names)

    def info(self, get_size=True):
        """
        Get name and type of each database in your environment.

        Return
        ------
        pandas.DataFrame
            Pandas dataframe with name, type, creation date and parent
            databases of each database in your environment.

        Example
        -------
        >>> from jai import Explorer
        ...
        >>> explorer = Explorer()
        >>> explorer.info()
        """
        info = self._info(mode="complete", get_size=get_size)
        if self.safe_mode:
            if get_size:
                info = check_response(InfoSizeResponse, info, list_of=True)
            else:
                info = check_response(InfoResponse, info, list_of=True)

        df_info = pd.DataFrame(info).rename(
            columns={
                "db_name": "name",
                "db_type": "type",
                "db_version": "last modified",
                "db_parents": "dependencies",
            }
        )
        if len(df_info) == 0:
            return df_info
        return df_info.sort_values(by="name")

    def user(self):
        """
        User information.

        Returns:
            dict:
            - userId: string
            - email: string
            - firstName: string
            - lastName: string
            - memberRole: string
            - namespace: string

        Example
        -------
        >>> from jai import Explorer
        ...
        >>> explorer = Explorer()
        >>> explorer.user()
        """
        user = self._user()
        if self.safe_mode:
            return check_response(UserResponse, user).dict()
        return user

    def environments(self):
        """
        Return names of available environments.

        Example
        -------
        >>> from jai import Explorer
        ...
        >>> explorer = Explorer()
        >>> explorer.environments()
        """
        envs = self._environments()
        if self.safe_mode:
            environments = []
            for v in check_response(EnvironmentsResponse, envs, list_of=True):
                if v["key"] is None:
                    v.pop("key")
                environments.append(v)
            return environments
        return envs

    def describe(self, name: str):
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

        Example
        -------
        >>> from jai import Explorer
        ...
        >>> explorer = Explorer()
        >>> explorer.describe(name)
        """
        description = self._describe(name)
        if self.safe_mode:
            description = check_response(DescribeResponse, description).dict()
            description = {k: v for k, v in description.items() if v is not None}
        return description

    def rename(self, original_name: str, new_name: str):
        """
        Renames a database.

        Args:
        original_name (str): The name of the database you want to rename.
        new_name (str): The new name of the database.

        Returns:
        String with message of successful renaming.

        Example
        -------
        >>> from jai import Explorer
        ...
        >>> explorer = Explorer()
        >>> explorer.rename(original_name, new_name)
        """

        response = self._rename(original_name=original_name, new_name=new_name)
        if self.safe_mode:
            return check_response(str, response)
        return response

    def transfer(
        self,
        original_name: str,
        to_environment: str,
        new_name: str = None,
        from_environment: str = "default",
    ):
        """
        Transfers a collection from one environment to another.

        Args
        ----
        original_name: str
            The name of the database you want to transfer.
        to_environment: str
            The environment to transfer the variable to.
        new_nam: str
            The name of the new environment. If not specified, the original name is used.
        from_environment: str
            The environment to transfer the variable from. Defaults to `default`.

        Returns
        -------
        A string confirming the tranfer.

        Example
        -------
        >>> from jai import Explorer
        ...
        >>> explorer = Explorer()
        >>> explorer.transfer(original_name, to_environment)
        """
        response = self._transfer(
            original_name=original_name,
            to_environment=to_environment,
            new_name=new_name,
            from_environment=from_environment,
        )
        if self.safe_mode:
            return check_response(str, response)
        return response

    def import_database(
        self,
        database_name: str,
        owner_id: str,
        owner_email: str,
        import_name: str = None,
    ):
        """
        It imports a database from another user/environment.

        Args
        ----
          database_name: str
            The name of the database to import.
          owner_id: str
            The ID of the user who own the database.
          owner_email: str
            The email address of the user who own the database. This is an alternative to owner_id.
          import_name: str
            The name of the database to be imported, in case of renaming. Defaults to None (the database is not renamed).

        Returns
        -------
          A string confirming the import process.

        Example
        -------
        >>> from jai import Explorer
        ...
        >>> explorer = Explorer()
        >>> explorer.import_database(
        ...     database_name,
        ...     owner_id,
        ...     import_name
        ... )
        """
        response = self._import_database(
            database_name=database_name,
            owner_id=owner_id,
            owner_email=owner_email,
            import_name=import_name,
        )
        if self.safe_mode:
            return check_response(str, response)
        return response
