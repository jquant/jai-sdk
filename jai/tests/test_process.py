import numpy as np
import pandas as pd
import pytest

from jai.utilities.processing import (filter_resolution, filter_similar,
                                       find_threshold, predict2df)


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
# Tests for filter similar
# =============================================================================
def test_filter_similar_threshold():
    similar = [{
        "query_id": 0,
        "results": [{
            'id': i,
            'distance': i
        } for i in range(20)]
    }]
    gab = [{'id': 0, 'distance': 0, 'query_id': 0}]
    assert filter_similar(
        similar, threshold=0,
        return_self=True) == gab, "filter similar results failed"
    assert filter_similar(
        similar, threshold=1,
        return_self=True) == gab, "filter similar results failed"

    assert filter_similar(
        similar, threshold=None,
        return_self=True) == gab, "filter similar results failed"


def test_filter_similar_self():
    similar = [{
        "query_id": 0,
        "results": [{
            'id': i,
            'distance': i
        } for i in range(20)]
    }]
    assert filter_similar(
        similar, threshold=0,
        return_self=False) == [], "filter similar results failed. (self param)"
    assert filter_similar(similar, threshold=0, return_self=True) == [{
        'distance':
        0,
        'id':
        0,
        'query_id':
        0
    }], "filter similar results failed. (self param)"


def test_filter_similar_null():
    similar = [{
        "query_id": 0,
        "results": [{
            'id': i,
            'distance': i
        } for i in range(20)]
    }]
    assert filter_similar(similar,
                          threshold=0,
                          return_self=False,
                          skip_null=False) == [{
                              'query_id': 0,
                              'distance': None,
                              'id': None
                          }], "filter similar results failed. (null param)"
    assert filter_similar(
        similar, threshold=0, return_self=False,
        skip_null=True) == [], "filter similar results failed. (null param)"


# =============================================================================
# Tests for filter predict
# =============================================================================
@pytest.mark.parametrize('predict, res', [([{
    "id": 0,
    "predict": 0.1
}], pd.DataFrame({'predict': 0.1}, index=pd.Index([0], name="id")))])
def test_process_predict_regression(predict, res):
    assert (
        predict2df(predict) == res).all(None), "filter predict results failed."


@pytest.mark.parametrize(
    'predict, res',
    [([{
        "id": 0,
        "predict": {
            '0.1': 1,
            '0.5': 5,
            '0.9': 9
        }
    }],
      pd.DataFrame({
          'predict_0.1': 1,
          'predict_0.5': 5,
          'predict_0.9': 9
      },
                   index=pd.Index([0], name="id")))])
def test_process_predict_quantiles(predict, res):
    assert (
        predict2df(predict) == res).all(None), "filter predict results failed."


@pytest.mark.parametrize('predict, res', [([{
    "id": 0,
    "predict": 'class1'
}], pd.DataFrame({'predict': 'class1'}, index=pd.Index([0], name="id")))])
def test_process_predict_classification(predict, res):
    assert (
        predict2df(predict) == res).all(None), "filter predict results failed."


@pytest.mark.parametrize('predict, res',
                         [([{
                             "id": 0,
                             "predict": {
                                 'class0': 0.1,
                                 'class1': .5,
                                 'class2': .4
                             }
                         }],
                           pd.DataFrame(
                               {
                                   'class0': 0.1,
                                   'class1': .5,
                                   'class2': .4,
                                   'predict': 'class1',
                                   'probability(%)': 50.0
                               },
                               index=pd.Index([0], name="id")))])
def test_process_predict_proba(predict, res):
    assert (predict2df(predict) == res
            ).all(None), "filter predict results failed. (proba)"


# =============================================================================
# Tests for filter resolution
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
    assert filter_resolution(similar,
                             .2) == expect, "filter resolution results failed."
