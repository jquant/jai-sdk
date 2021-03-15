import numpy as np
import pandas as pd
import pytest
from jai.functions.utils_funcs import (list2json, series2json, df2json,
                                       data2json)
from jai.image import read_image_folder, resize_image_folder

from pandas._testing import assert_series_equal
from pathlib import Path


@pytest.fixture(scope="session")
def setup_dataframe():
    TITANIC_TRAIN = "https://raw.githubusercontent.com/rebeccabilbro/titanic/master/data/train.csv"
    TITANIC_TEST = "https://raw.githubusercontent.com/rebeccabilbro/titanic/master/data/test.csv"
    train = pd.read_csv(TITANIC_TRAIN)
    test = pd.read_csv(TITANIC_TEST)
    return train, test


@pytest.fixture(scope="session")
def setup_img_data():
    IMG_FILE = Path("jai/test_data/test_imgs/dataframe_img.pkl")
    img_file = pd.read_pickle(IMG_FILE)
    return img_file


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
def test_data2json(setup_dataframe, setup_img_data, dtype):
    dict_dbtype = {"Text": "text", "Image": "image_base64"}
    db_types = ["Text", "Image"]
    train, _ = setup_dataframe
    img_data = setup_img_data

    for db_type in db_types:
        col_name = dict_dbtype[db_type]

        if db_type == "Text":
            data = train.rename(columns={
                "PassengerId": "id",
                "Name": col_name
            }).set_index("id")[col_name]
        else:
            data = img_data

        if dtype == 'list':
            data = data.tolist()
            ids = range(len(data))
            s = pd.Series(data, index=pd.Index(ids, name='id'), name=col_name)
            gab = s.reset_index().to_json(orient='records')
        elif dtype == 'array':
            data = data.values
            ids = range(len(data))
            s = pd.Series(data, index=pd.Index(ids, name='id'), name=col_name)
            gab = s.reset_index().to_json(orient='records')
        elif dtype == 'df':
            data = data.to_frame()
            ids = data.index
            s = pd.Series(data[col_name],
                          index=pd.Index(ids, name='id'),
                          name=col_name)
            gab = s.reset_index().to_json(orient='records')
        elif dtype == 'df_id':
            data = data.reset_index().rename(columns={"index": "id"})
            gab = data.rename(columns={
                'Name': col_name
            }).to_json(orient='records')
        else:
            data = data.rename(col_name)
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


def test_read_image_folder(setup_img_data,
                           img_folder=Path("jai/test_data/test_imgs")):
    img_data = setup_img_data
    data = read_image_folder(image_folder=img_folder)
    assert_series_equal(img_data, data)


def test_resize_image_folder(img_folder=Path("jai/test_data/test_imgs"),
                             extensions=[".png", ".jpg", ".jpeg"]):
    # paths
    previously_generated_imgs = img_folder / "generate_resize"
    test_generated_imgs = img_folder / "resized"

    # if things go well
    resize_image_folder(img_folder=img_folder)
    set_previous = set(
        [item.name for item in list(previously_generated_imgs.iterdir())])
    set_current = set(
        [item.name for item in list(test_generated_imgs.iterdir())])
    assert set_previous == set_current

    # if things go south
    with pytest.raises(Exception):
        resize_image_folder(img_folder="not_found")
