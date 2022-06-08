from jai.core.authentication import get_authentication, set_authentication
import pytest
import os


def test_set_authentication():
    set_authentication("auth", "TEST_AUTH")

    auth_key = get_authentication(env_var="TEST_AUTH")
    assert auth_key == "auth"

    set_authentication("other_value", "TEST_AUTH")

    auth_key = get_authentication(env_var="TEST_AUTH")
    assert auth_key == "other_value"

    os.environ.pop("TEST_AUTH", None)

    with pytest.raises(ValueError):
        auth_key = get_authentication("TEST_AUTH")


def test_error_authentication():
    with pytest.raises(ValueError):
        get_authentication("not_a_auth_key")
