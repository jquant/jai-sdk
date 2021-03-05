import numpy as np
import pandas as pd
import pytest
from jai.functions.utils_funcs import (list2json, series2json, df2json,
                                       data2json)



@pytest.fixture(scope="session")
def setup_dataframe():
    TITANIC_TRAIN = "https://raw.githubusercontent.com/rebeccabilbro/titanic/master/data/train.csv"
    TITANIC_TEST = "https://raw.githubusercontent.com/rebeccabilbro/titanic/master/data/test.csv"
    train = pd.read_csv(TITANIC_TRAIN)
    test = pd.read_csv(TITANIC_TEST)
    return train, test

# =============================================================================
# Tests for data2json
# =============================================================================
@pytest.mark.parametrize('data', [list('ab'), np.array(['abc', 'def'])])
@pytest.mark.parametrize('name', ['text', 'image_base64'])
def test_list2json(data, name):
    index = pd.Index(range(len(data)), name='id')
    gab = pd.Series(data, index=index,
                    name=name).reset_index().to_json(orient='records')
    assert list2json(data, name) == gab, 'list2json failed.'


@pytest.mark.parametrize('data', [list('ab'), np.array(['abc', 'def'])])
@pytest.mark.parametrize('name', ['text', 'image_base64'])
@pytest.mark.parametrize('ids', [None, [10, 12]])
def test_series2json(data, name, ids):
    ids = ids if ids is not None else range(len(data))
    s = pd.Series(data, index=pd.Index(ids, name='id'), name=name)
    gab = s.reset_index().to_json(orient='records')
    assert series2json(s, name) == gab, 'series2json failed.'


@pytest.mark.parametrize('col1, col2, ids',
                         [([42, 123], ['abc', 'def'], None),
                          ([69, 420], ['ghi', 'jkl'], [10, 64])])
def test_df2json(col1, col2, ids):
    df = pd.DataFrame({"col1": col1, "col2": col2}, index=ids)
    if ids is None:
        ids = range(len(col1))
    out = ','.join([
        f'{{"id":{i},"col1":{a},"col2":"{b}"}}'
        for i, a, b in zip(ids, col1, col2)
    ])
    assert df2json(df) == '[' + out + ']', 'df2json failed.'




@pytest.mark.parametrize("dtype", ["list", "array", "series", "df", "df_id"])
def test_data2json(setup_dataframe, dtype):
    db_type = "Text"
    train, _ = setup_dataframe
    data = train.rename(columns={"PassengerId": "id"}).set_index("id")['Name']

    if dtype == 'list':
        data = data.tolist()
        ids = range(len(data))
        s = pd.Series(data, index=pd.Index(ids, name='id'), name="text")
        gab = s.reset_index().to_json(orient='records')
    elif dtype == 'array':
        data = data.values
        ids = range(len(data))
        s = pd.Series(data, index=pd.Index(ids, name='id'), name="text")
        gab = s.reset_index().to_json(orient='records')
    elif dtype == 'df':
        data = data.to_frame()
        ids = data.index
        s = pd.Series(data['Name'], index=pd.Index(ids, name='id'), name="text")
        gab = s.reset_index().to_json(orient='records')
    elif dtype == 'df_id':
        data = data.reset_index().rename(columns={"index": "id"})
        gab = data.rename(columns={'Name': "text"}).to_json(orient='records')
    else:
        data = data.rename("text")
        gab = data.reset_index().to_json(orient='records')

    assert data2json(data, db_type) == gab, 'df2json failed.'



@pytest.mark.parametrize('data', [list('ab'), np.array(['abc', 'def'])])
@pytest.mark.parametrize('name', ['text', 'image_base64'])
@pytest.mark.parametrize('ids', [[1, 1], [10, 10]])
def test_series_error(data, name, ids):
    with pytest.raises(ValueError):
        ids = ids if ids is not None else range(len(data))
        s = pd.Series(data, index=pd.Index(ids, name='id'), name=name)
        series2json(s, name)


@pytest.mark.parametrize('col1, col2, ids',
                         [([42, 123], ['abc', 'def'], [2, 2]),
                          ([69, 420], ['ghi', 'jkl'], [42, 42])])
def test_df_error(col1, col2, ids):
    with pytest.raises(ValueError):
        df = pd.DataFrame({"col1": col1, "col2": col2}, index=ids)
        df2json(df)
