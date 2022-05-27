import os
import warnings
from decouple import config


def set_authentication(auth_key: str, env_var: str = "JAI_AUTH"):

    if env_var in os.environ:
        warnings.warn("Overwriting environment variable `{env_var}`.")

    os.environ[env_var] = auth_key


def get_authentication(env_var: str = "JAI_AUTH"):
    try:
        return config('HEADER_TEST')
    except KeyError:
        raise ValueError(
            f"No authentication key found on environment variable `{env_var}`.\n"
            "Check if `env_var` is the correct name of the environment variable.\n"
            "Set the authentication key to the correct environment variable or use `set_authentication` function."
        )
