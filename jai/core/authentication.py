import os
import warnings
from decouple import config, UndefinedValueError
import requests


def get_auth_key(email: str, firstName: str, lastName: str, company: str = ""):
    """
    Request an auth key to use JAI-SDK with.

    Args
    ----------
    `email`: str
        A valid email address where the auth key will be sent to.
    `firstName`: str
        User's first name.
    `lastName`: str
        User's last name.
    `company`: str
        User's company.

    Return
    ----------
    `response`: dict
        A Response object with whether or not the auth key was created.
    """
    url = "https://mycelia.azure-api.net/clone"
    body = {
        "email": email,
        "firstName": firstName,
        "lastName": lastName,
        "company": company
    }
    response = requests.put(url + "/auth", json=body)
    return response


def set_authentication(auth_key: str, env_var: str = "JAI_AUTH"):

    if env_var in os.environ:
        warnings.warn("Overwriting environment variable `{env_var}`.")

    os.environ[env_var] = auth_key


def get_authentication(env_var: str = "JAI_AUTH"):
    try:
        return config(env_var)
    except (KeyError, UndefinedValueError):
        raise ValueError(
            f"No authentication key found on environment variable `{env_var}`.\n"
            "Check if `env_var` is the correct name of the environment variable.\n"
            "Set the authentication key to the correct environment variable or use `set_authentication` function."
        )
