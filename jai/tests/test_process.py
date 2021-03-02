import pytest
from jai.functions.utils_funcs import process_similar, process_predict


# =============================================================================
# Tests for process similar
# =============================================================================
def test_process_similar_threshold():
    similar = [{
        "query_id":
        0,
        "results": [{
            'id': 0,
            'distance': 0
        }, {
            'id': 1,
            'distance': 1
        }, {
            'id': 2,
            'distance': 2
        }]
    }]
    gab = [{'id': 0, 'distance': 0, 'query_id': 0}]
    assert process_similar(
        similar, 0, True) == gab, "process similar results failed. (threshold)"
    assert process_similar(
        similar, 1, True) == gab, "process similar results failed. (threshold)"


def test_process_similar_self():
    similar = [{
        "query_id":
        0,
        "results": [{
            'id': 0,
            'distance': 0
        }, {
            'id': 1,
            'distance': 1
        }, {
            'id': 2,
            'distance': 2
        }]
    }]
    assert process_similar(
        similar, 0,
        False) == [], "process similar results failed. (self param)"
    assert process_similar(similar, 1, False) == [{
        'id': 1,
        'distance': 1,
        'query_id': 0
    }], "process similar results failed. (self param)"


def test_process_similar_null():
    similar = [{
        "query_id":
        0,
        "results": [{
            'id': 0,
            'distance': 0
        }, {
            'id': 1,
            'distance': 1
        }, {
            'id': 2,
            'distance': 2
        }]
    }]
    assert process_similar(similar, 0, False, False) == [{
        'query_id': 0,
        'distance': None,
        'id': None
    }], "process similar results failed. (null param)"
    assert process_similar(similar, 1, False, False) == [{
        'query_id': 0,
        'id': 1,
        'distance': 1
    }], "process similar results failed. (null param)"


# =============================================================================
# Tests for process predict
# =============================================================================
@pytest.mark.parametrize('predict', [[{"id": 0, "predict": 'class1'}]])
def test_process_predict(predict):
    assert process_predict(predict) == [{
        'id': 0,
        'predict': 'class1'
    }], "process predict results failed."


@pytest.mark.parametrize('predict', [[{
    "id": 0,
    "predict": {
        'class0': 0.1,
        'class1': .5,
        'class2': .4
    }
}]])
def test_process_predict_proba(predict):
    assert process_predict(predict) == [{
        'id': 0,
        'predict': 'class1',
        'probability(%)': 50.0
    }], "process predict results failed. (proba)"


@pytest.mark.parametrize('predict', [[{"id": 0, "predict": ['class1']}]])
def test_process_predict_error(predict):
    with pytest.raises(ValueError):
        process_predict(predict)