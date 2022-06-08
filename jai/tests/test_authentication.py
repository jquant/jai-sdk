from jai.core.authentication import get_authentication, set_authentication
import pytest
import os


def test_get_authentication():
    auth_key = get_authentication(env_var="EMPTY_AUTH")
    assert auth_key == ""


def test_set_authentication():
    initial_key = get_authentication()

    set_authentication("auth")

    auth_key = get_authentication()
    assert auth_key == "auth"

    set_authentication("other_value")

    auth_key = get_authentication()
    assert auth_key == "other_value"

    os.environ.pop("JAI_AUTH", None)

    auth_key = get_authentication()
    assert auth_key == initial_key


def test_error_authentication():
    with pytest.raises(ValueError):
        get_authentication("not_a_auth_key")
