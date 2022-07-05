from ..core.base import BaseJai
from ..core.validations import check_response
from ..types.responses import DescribeResponse, UserResponse, ValidResponse

__all__ = ["TaskBase"]


class TaskBase(BaseJai):
    """
    Base class for communication with the Mycelia API.

    Used as foundation for more complex applications for data validation such
    as matching tables, resolution of duplicated values, filling missing values
    and more.

    """

    def __init__(
        self,
        name: str,
        environment: str = "default",
        env_var: str = "JAI_AUTH",
        verbose: int = 1,
        safe_mode: bool = False,
    ):
        self._init_values = {
            "environment": environment,
            "env_var": env_var,
            "verbose": verbose,
            "safe_mode": safe_mode,
        }
        super(TaskBase, self).__init__(environment, env_var)
        self.safe_mode = safe_mode

        if self.safe_mode:
            user = self._user()
            user = check_response(UserResponse, user).dict()

            if verbose:
                print(
                    "Connection established.\n"
                    f"Welcome {user['firstName']} {user['lastName']}"
                )

        self.name = name

    @property
    def name(self):
        return self._name

    @property
    def db_type(self):
        if self.is_valid():
            return self.describe()["dtype"]
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
            description = check_response(DescribeResponse, description).dict()
            description = {k: v for k, v in description.items() if v is not None}
        return description
