from aiohttp_cors import custom_cors
from jai.core.authentication import get_auth_key, get_authentication, set_authentication
import pytest
import os
import requests

#! TODO: CHECK THE CORRECT RESPONSE OF /AUTH METHOD


class MockResponse():
    @staticmethod
    def json():
        return {
            "content":
            "Registration successful. Check your email for the auth key."
        }


def test_authentication(monkeypatch):
    new_user = {
        "email": "user@test.com",
        "firstName": "User",
        "lastName": "Name",
        "company": "JAI"
    }

    custom_res = {
        "content":
        "Registration successful. Check your email for the auth key."
    }

    def mockreturn(*args, **kwargs):
        return MockResponse()

    monkeypatch.setattr(requests, "put", mockreturn)

    x = get_auth_key(**new_user)
    assert x.json() == custom_res


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
