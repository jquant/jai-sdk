from jai.core.authentication import get_authentication, set_authentication
import pytest


def test_get_authentication():
    auth_key = get_authentication()
    assert auth_key == ""


def test_set_authentication():
    set_authentication("auth")

    auth_key = get_authentication()
    assert auth_key == "auth"


def test_error_authentication():
    with pytest.raises(ValueError):
        get_authentication("not_a_auth_key")
