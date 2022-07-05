import os
import warnings

import requests
from decouple import UndefinedValueError, config


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

    Example
    ----------
    >>> from jai import get_auth_key
    >>> r = get_auth_key(email='email@mail.com', firstName='Jai', lastName='Z')
    >>> print(r.status_code) # This should be 201
    >>> print(r.json())
    """
    url = "https://mycelia.azure-api.net/clone"
    body = {
        "email": email,
        "firstName": firstName,
        "lastName": lastName,
        "company": company,
    }
    response = requests.put(url + "/auth", json=body)
    print(response.json())
    return response


def set_authentication(auth_key: str, env_var: str = "JAI_AUTH"):
    """
    Sets the environment variable with the authentication key.

    Args:
        auth_key (str): Authentication key value.
        env_var (str, optional): Environment variable name. Defaults to "JAI_AUTH".

    Example
    ----------
    >>> from jai import set_authentication
    >>> set_authentication("xXxxxXXxXXxXXxXXxXXxXXxXXxxx")
    """
    if env_var in os.environ:
        warnings.warn(f"Overwriting environment variable `{env_var}`.", stacklevel=2)

    os.environ[env_var] = auth_key


def get_authentication(env_var: str = "JAI_AUTH"):
    """
    Returns the authentication key if defined.
    See :ref:`Configuring your auth key <source/overview/set_authentication:how to configure your auth key>`.

    Args:
        env_var (str, optional): Environment variable name. Defaults to "JAI_AUTH".

    Raises:
        ValueError: If no authentication key was found.

    Returns:
        str: Authentication Key.

    Example
    ----------
    >>> from jai import get_authentication
    >>> get_authentication()
    """
    try:
        return config(env_var)
    except (KeyError, UndefinedValueError):
        raise ValueError(
            f"No authentication key found on environment variable `{env_var}`.\n"
            "Check if `env_var` is the correct name of the environment variable.\n"
            "Set the authentication key to the correct environment variable or use `set_authentication` function."
        )
