from ..core.base import BaseJai

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
        auth_key: str = None,
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
        super(TaskBase, self).__init__(
            auth_key, environment=environment, env_var=env_var
        )
        self.safe_mode = safe_mode

        if self.safe_mode:
            user = self._user()

            if verbose:
                print(
                    "Connection established.\n"
                    f"Welcome {user['firstName']} {user['lastName']}"
                )

        self.name = name

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        self._name = value

    @property
    def db_type(self):
        if self.is_valid():
            return self.describe()["dtype"]
        return None

    def is_valid(self):
        """
        Check if a given name is a valid database name (i.e., if it is
        in your environment).

        Return
        ------
        response: bool
            True if name is in your environment. False, otherwise.

        """
        return self._is_valid(self.name)

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
        return self._describe(self.name)
