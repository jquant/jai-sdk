from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandas._testing import assert_frame_equal, assert_series_equal

from jai.core.utils_funcs import data2json, df2json, series2json
from jai.utilities import read_image_folder


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
        data = setup_img_data

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


def test_read_image_folder(setup_img_data,
                           image_folder=Path("jai/test_data/test_imgs")):
    img_data = setup_img_data
    data = read_image_folder(image_folder=image_folder)
    data = data.rename(columns={"test_imgs": "image_base64"})
    assert_series_equal(img_data.to_frame(), data['image_base64'])


def test_read_image_folder_corrupted_ignore(
        image_folder=Path("jai/test_data/test_imgs_corrupted")):
    # create empty Series
    index = pd.Index([], name='id')
    empty_series = pd.Series([], index=index, name='image_base64')
    data = read_image_folder(image_folder=image_folder)
    assert_series_equal(empty_series, data)


def test_read_image_folder_corrupted(
        image_folder=Path("jai/test_data/test_imgs_corrupted")):
    with pytest.raises(ValueError):
        read_image_folder(image_folder=image_folder, handle_errors="raise")


def test_read_image_folder_no_parameters():
    # just call function with no parameters
    with pytest.raises(TypeError):
        read_image_folder()


def test_read_image_folder_list(setup_img_data,
                                images=[
                                    Path("jai/test_data/test_imgs/"),
                                ]):
    # the idea for this particular test is to simply make use of the
    # previously generated dataframe for the read_image_folder test; since
    # we are passing the paths to each image file DIRECTLY, the indexes will
    # differ. That is why we reset it and rename it to "id" again
    img_data = setup_img_data
    img_data = img_data.reset_index(drop=True).rename_axis(index="id")
    data = read_image_folder(image_folder=images)
    data = data.rename(columns={"test_imgs": "image_base64"})
    assert_series_equal(img_data.to_frame(), data['image_base64'])
