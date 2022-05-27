from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandas._testing import assert_frame_equal

from jai.utilities import read_image
from jai.core.validations import check_dtype_and_clean
from jai.core.utils_funcs import data2json, df2json, series2json, resolve_db_type


@pytest.fixture(scope="session")
def setup_dataframe():
    TITANIC_TRAIN = "https://raw.githubusercontent.com/rebeccabilbro/titanic/master/data/train.csv"
    TITANIC_TEST = "https://raw.githubusercontent.com/rebeccabilbro/titanic/master/data/test.csv"

    train = pd.read_csv(TITANIC_TRAIN)
    test = pd.read_csv(TITANIC_TEST)
    return train, test


@pytest.fixture(scope="session")
def setup_img_data():
    IMG_FILE = Path("jai/test_data/test_imgs/dataframe_img.csv")
    return pd.read_csv(IMG_FILE).set_index("id").sort_index()


# =============================================================================
# Tests for data2json
# =============================================================================
@pytest.mark.parametrize('data', [list('ab'), np.array(['abc', 'def'])])
@pytest.mark.parametrize('name', ['text', 'image_base64'])
@pytest.mark.parametrize('ids', [None, [10, 12]])
def test_series2json(data, name, ids):
    ids = ids if ids is not None else range(len(data))
    s = pd.Series(data, index=pd.Index(ids, name='id'), name=name)
    gab = s.reset_index().to_json(orient='records')
    assert series2json(s) == gab, 'series2json failed.'


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


@pytest.mark.parametrize("col_name, db_type", [("text", "Text"),
                                               ("image_base64", "Image")])
@pytest.mark.parametrize("dtype", ["series", "df", "df_id"])
def test_data2json(setup_dataframe, setup_img_data, dtype, col_name, db_type):

    if db_type == "Text":
        train, _ = setup_dataframe
        data = train.rename(columns={
            "PassengerId": "id",
            "Name": col_name
        }).set_index("id")[col_name]
    else:
        data = setup_img_data.rename(columns={"test_imgs": "image_base64"})
        data = data['image_base64']

    if dtype == 'df':
        data = data.to_frame()
        ids = data.index
        s = pd.Series(data[col_name],
                      index=pd.Index(ids, name='id'),
                      name=col_name)
        gab = s.reset_index().to_json(orient='records')
    elif dtype == 'df_id':
        data = data.reset_index().rename(columns={"index": "id"})
        gab = data.rename(columns={'Name': col_name}).to_json(orient='records')
    else:
        data = data.rename(col_name)
        gab = data.reset_index().to_json(orient='records')

    assert data2json(data, db_type) == gab, 'df2json failed.'


@pytest.mark.parametrize("col_name, db_type", [("text", "Text"),
                                               ("image_base64", "Image")])
@pytest.mark.parametrize("filter_name", ["Pclass"])
def test_data2json_filters(setup_dataframe, col_name, filter_name, db_type):

    train, _ = setup_dataframe
    train = train.rename(columns={
        "PassengerId": "id",
        "Name": col_name,
    })

    data = train.set_index("id").loc[:, [col_name, filter_name]]

    gab = data.reset_index().to_json(orient='records')

    assert data2json(data, db_type,
                     filter_name=filter_name) == gab, 'df2json failed.'

    data = train.loc[:, ['id', col_name, filter_name]]
    gab = data.to_json(orient='records')

    assert data2json(data, db_type,
                     filter_name=filter_name) == gab, 'df2json failed.'


def test_data2json_exceptions(setup_dataframe):
    train, _ = setup_dataframe
    train = train.rename(columns={"PassengerId": "id"})

    with pytest.raises(TypeError):
        data2json(data=list(), dtype="Text")

    with pytest.raises(TypeError):
        data2json(data=tuple(), dtype="Text")

    with pytest.raises(TypeError):
        data2json(data=np.array([]), dtype="Text")

    with pytest.raises(NotImplementedError):
        data2json(data=dict(), dtype="Text")

    with pytest.raises(ValueError):
        data2json(data=train[["Name", "Sex"]], dtype="Text")

    with pytest.raises(ValueError):
        data2json(data=train[["Name", "Sex", "Pclass"]], dtype="Text")

    with pytest.raises(ValueError):
        data2json(data=train, dtype="Text")

    with pytest.raises(ValueError):
        data2json(data=train[["Name"]], dtype="SelfSupervised")

    with pytest.raises(NotImplementedError):
        data2json(data=dict(), dtype="SelfSupervised")

    with pytest.raises(ValueError):
        data2json(data=train[["Name"]], dtype="Supervised")

    with pytest.raises(NotImplementedError):
        data2json(data=dict(), dtype="Supervised")

    with pytest.raises(ValueError):
        data2json(data=train[["Name"]], dtype="Unsupervised")

    with pytest.raises(ValueError):
        data2json(data=train[["Name"]], dtype="Invalid")


@pytest.mark.parametrize('data', [list('ab'), np.array(['abc', 'def'])])
@pytest.mark.parametrize('name', ['text', 'image_base64'])
@pytest.mark.parametrize('ids', [[1, 1], [10, 10]])
def test_series_error(data, name, ids):
    with pytest.raises(ValueError):
        ids = ids if ids is not None else range(len(data))
        s = pd.Series(data, index=pd.Index(ids, name='id'), name=name)
        series2json(s)


@pytest.mark.parametrize('col1, col2, ids',
                         [([42, 123], ['abc', 'def'], [2, 2]),
                          ([69, 420], ['ghi', 'jkl'], [42, 42])])
def test_df_error(col1, col2, ids):
    with pytest.raises(ValueError):
        df = pd.DataFrame({"col1": col1, "col2": col2}, index=ids)
        df2json(df)


@pytest.mark.parametrize('folder', [Path("jai/test_data/test_imgs")])
def test_read_image(setup_img_data, folder):
    img_data = setup_img_data
    data = read_image(folder=folder, id_pattern="img(\d+)")
    assert_frame_equal(img_data, data.sort_index())


@pytest.mark.parametrize('folder', [Path("jai/test_data/test_imgs_corrupted")])
@pytest.mark.parametrize('handle_errors', ["ignore", "warn"])
def test_read_image_corrupted_ignore(folder, handle_errors):
    # create empty Series
    empty_df = pd.DataFrame([])
    data = read_image(folder=folder,
                      id_pattern="img(\d+)_corrupted",
                      handle_errors=handle_errors)
    assert_frame_equal(empty_df, data)


@pytest.mark.parametrize('folder', [Path("jai/test_data/test_imgs_corrupted")])
def test_read_image_corrupted(folder):
    with pytest.raises(ValueError):
        read_image(folder=folder, handle_errors="raise")


def test_read_image_no_parameters():
    # just call function with no parameters
    with pytest.raises(TypeError):
        read_image()


@pytest.mark.parametrize('images', [[Path("jai/test_data/test_imgs/")]])
def test_read_image_list(setup_img_data, images):
    # the idea for this particular test is to simply make use of the
    # previously generated dataframe for the read_image test; since
    # we are passing the paths to each image file DIRECTLY, the indexes will
    # differ. That is why we reset it and rename it to "id" again
    img_data = setup_img_data
    data = read_image(folder=images, id_pattern="img(\d+)")
    assert_frame_equal(img_data, data.sort_index())


def test_check_dtype_and_clean():
    # mock data
    r = 1100
    data = pd.DataFrame({
        "category": [str(i) for i in range(r)],
        "number": [i for i in range(r)]
    })

    # make a few lines on 'category' column NaN
    data.loc[1050:, "category"] = np.nan
    assert_frame_equal(check_dtype_and_clean(data, "Supervised"), data)


@pytest.mark.parametrize("db_type, col, ans", [({
    "col1": "FastText"
}, "col1", "FastText"), ({
    "col1": "FastText"
}, "col2", "TextEdit")])
def test_resolve_db_type(db_type, col, ans):
    assert resolve_db_type(db_type, col) == ans
