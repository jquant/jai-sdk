import os
import warnings


def set_authentication(auth_key: str, env_var: str = "JAI_AUTH"):

    if env_var in os.environ:
        warnings.warn("Overwriting environment variable `{env_var}`.")

    os.environ[env_var] = auth_key


def get_authentication(env_var: str = "JAI_AUTH"):
    if env_var not in os.environ:
        raise ValueError(
            f"No authentication key found on environment variable `{env_var}`."
            "Check if `env_var` is the correct name of the environment variable."
            "Set the authentication key to the correct environment variable or use `set_authentication` function."
        )

    return os.environ[env_var]