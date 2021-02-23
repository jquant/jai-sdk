# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 15:04:55 2021

@author: Kazu
"""

import numpy as np
import pandas as pd
import pytest
from jai.auxiliar_funcs.utils_funcs import (list2json, series2json, df2json,
                                            data2json, process_similar, process_predict)

@pytest.mark.parametrize('data', [list('ab'), np.array(['abc', 'def'])])
@pytest.mark.parametrize('name', ['text', 'image_base64'])
def test_list2json(data, name):
    index = pd.Index(range(len(data)), name='id')
    gab = pd.Series(data, index=index, name=name).reset_index().to_json(orient='records')
    assert list2json(data, name) == gab, 'list2json failed.'

@pytest.mark.parametrize('data', [list('ab'), np.array(['abc', 'def'])])
@pytest.mark.parametrize('name', ['text', 'image_base64'])
@pytest.mark.parametrize('ids', [None, [10, 12]])
def test_series2json(data, name, ids):
    ids = ids if ids is not None else range(len(data))
    s = pd.Series(data, index=pd.Index(ids, name='id'), name=name)
    gab = s.reset_index().to_json(orient='records')
    assert series2json(s, name) == gab, 'series2json failed.'


def t_df2json():
    pass

def t_data2json():
    pass

def test_process_similar_threshold():
    similar = [{"query_id": 0, "results": [{'id':0, 'distance':0}, {'id':1, 'distance':1}, {'id':2, 'distance':2}]}]
    assert process_similar(similar, 0, True) == [{'id': 0, 'distance': 0, 'query_id': 0}], "process similar results failed. (threshold)"
    assert process_similar(similar, 1, True) == [{'id': 0, 'distance': 0, 'query_id': 0}], "process similar results failed. (threshold)"

def test_process_similar_self():
    similar = [{"query_id": 0, "results": [{'id':0, 'distance':0}, {'id':1, 'distance':1}, {'id':2, 'distance':2}]}]
    assert process_similar(similar, 0, False) == [], "process similar results failed. (self param)"
    assert process_similar(similar, 1, False) == [{'id': 1, 'distance': 1, 'query_id': 0}], "process similar results failed. (self param)"

def test_process_similar_null():
    similar = [{"query_id": 0, "results": [{'id':0, 'distance':0}, {'id':1, 'distance':1}, {'id':2, 'distance':2}]}]
    assert process_similar(similar, 0, False, False) == [{'query_id': 0, 'distance': None, 'id': None}], "process similar results failed. (null param)"


@pytest.mark.parametrize('predict', [[{"id": 0, "predict": 'class1'}]])
def test_process_predict(predict):
    assert process_predict(predict) == [{'id': 0, 'predict': 'class1'}], "process predict results failed."


@pytest.mark.parametrize('predict',
                         [[{"id": 0, "predict": {'class0': 0.1, 'class1':.5, 'class2':.4}}]])
def test_process_predict_proba(predict):
     assert process_predict(predict) == [{'id': 0, 'predict': 'class1', 'probability(%)': 50.0}], "process predict results failed. (proba)"


@pytest.mark.parametrize('predict',
                         [[{"id": 0, "predict": ['class1']}]])
def test_process_predict_error(predict):
    with pytest.raises(ValueError):
        process_predict(predict)