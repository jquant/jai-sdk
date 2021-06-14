import pytest
import numpy as np
import pandas as pd
from jai.processing import (find_threshold, process_similar, process_predict,
                            process_resolution)


# =============================================================================
# Tests for find threshold
# =============================================================================
@pytest.mark.parametrize(
    'similar, threshold',
    [([{
        "query_id": i,
        "results": [{
            'id': i + j,
            'distance': i // 50 + j
        } for j in range(20)]
    } for i in range(1000)], 6.),
     ([{
         "query_id": i,
         "results": [{
             'id': i + j,
             'distance': i // 50 + j
         } for j in range(20)]
     } for i in range(20)], 1.85),
     ([{
         "query_id": 0,
         "results": [{
             'id': j,
             'distance': j
         } for j in range(65)]
     }], 4.15)])
def test_find_threshold(similar, threshold):
    np.random.seed(42)
    assert find_threshold(similar) == threshold, "find threshold failed."


# =============================================================================
# Tests for process similar
# =============================================================================
def test_process_similar_threshold():
    similar = [{
        "query_id": 0,
        "results": [{
            'id': i,
            'distance': i
        } for i in range(20)]
    }]
    gab = [{'id': 0, 'distance': 0, 'query_id': 0}]
    assert process_similar(
        similar, 0, True) == gab, "process similar results failed. (threshold)"
    assert process_similar(
        similar, 1, True) == gab, "process similar results failed. (threshold)"


def test_process_similar_self():
    similar = [{
        "query_id": 0,
        "results": [{
            'id': i,
            'distance': i
        } for i in range(20)]
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
        "query_id": 0,
        "results": [{
            'id': i,
            'distance': i
        } for i in range(20)]
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
    res = pd.DataFrame({'predict': 'class1'}, index=pd.Index([0], name="id"))
    assert (process_predict(predict) == res
            ).all(None), "process predict results failed."


@pytest.mark.parametrize('predict', [[{
    "id": 0,
    "predict": {
        'class0': 0.1,
        'class1': .5,
        'class2': .4
    }
}]])
def test_process_predict_proba(predict):
    res = pd.DataFrame(
        {
            'class0': 0.1,
            'class1': .5,
            'class2': .4,
            'predict': 'class1',
            'probability(%)': 50.0
        },
        index=pd.Index([0], name="id"))
    assert (process_predict(predict) == res
            ).all(None), "process predict results failed. (proba)"


@pytest.mark.parametrize('predict', [[{"id": 0, "predict": ['class1']}]])
def test_process_predict_error(predict):
    with pytest.raises(ValueError):
        process_predict(predict)


# =============================================================================
# Tests for process resolution
# =============================================================================
def test_process_resolution():
    similar = [{
        "query_id":
        0,
        "results": [{
            'id': 0,
            'distance': 0
        }, {
            'id': 1,
            'distance': 0.1
        }]
    }, {
        "query_id":
        1,
        "results": [{
            'id': 1,
            'distance': 0
        }, {
            'id': 0,
            'distance': 0.1
        }, {
            'id': 2,
            'distance': 0.2
        }]
    }, {
        "query_id":
        2,
        "results": [{
            'id': 2,
            'distance': 0
        }, {
            'id': 1,
            'distance': 0.2
        }]
    }, {
        "query_id":
        3,
        "results": [{
            'id': 3,
            'distance': 0
        }, {
            'id': 4,
            'distance': 0.15
        }]
    }, {
        "query_id":
        4,
        "results": [{
            'id': 4,
            'distance': 0
        }, {
            'id': 3,
            'distance': 0.15
        }]
    }, {
        "query_id": 5,
        "results": [{
            'id': 5,
            'distance': 0
        }]
    }]
    expect = [{
        'id': 0,
        'resolution_id': 0
    }, {
        'id': 1,
        'resolution_id': 0
    }, {
        'id': 2,
        'resolution_id': 0
    }, {
        'id': 3,
        'resolution_id': 3
    }, {
        'id': 4,
        'resolution_id': 3
    }, {
        'id': 5,
        'resolution_id': 5
    }]
    assert process_resolution(
        similar, .2) == expect, "process resolution results failed."
