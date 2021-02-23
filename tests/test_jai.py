from jai import Jai

URL = 'http://localhost:8001'

def test_mock_demais():
    j = Jai(url=URL, auth_key="qualquer_coisa")
    assert j