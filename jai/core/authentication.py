import os
import warnings

import requests
from decouple import UndefinedValueError, config


def get_auth_key(
    firstName: str,
    lastName: str = "",
    email: str = "",
    work_email: str = "",
    phone: str = "",
    company: str = "",
    company_size: str = "",
    jobtitle: str = "",
    code_skills: str = "",
):
    """
    Request an auth key to use JAI-SDK with.

    Args
    ----------
    `firstName`: str
        User's first name.
    `lastName`: str
        User's last name.
    `email`: str
        A valid email address where the auth key will be sent to.
    `work_email`: str
        A valid business email address where the auth key will be sent to.
    `phone`: str
        Phone number.
    `company`: str
        User's company.
    `company_size`: str
        User's company size.
    `jobtitle`: str
        User's job title.
    `coding`: str
        If the user has experience coding.

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
        "firstName": firstName,
        "lastName": lastName,
        "email": email,
        "workEmail": work_email,
        "phone": phone,
        "company": company,
        "companySize": str(company_size),
        "jobtitle": jobtitle,
        "codeSkills": code_skills,
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


def get_authentication(env_var: str = "JAI_AUTH") -> str:
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
        return config(env_var)  # type: ignore
    except (KeyError, UndefinedValueError):
        raise ValueError(
            f"No authentication key found on environment variable `{env_var}`.\n"
            "Check if `env_var` is the correct name of the environment variable.\n"
            "Set the authentication key to the correct environment variable or use `set_authentication` function."
        )
