import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from jai.utilities.splits import split, split_recommendation


# =============================================================================
# Tests for split
# =============================================================================
@pytest.mark.parametrize(
    "param, gab_bases, gab",
    [
        (
            "col2",
            {"col2": pd.DataFrame({"col2": ["a", "b", "c", "d", "e"]})},
            pd.DataFrame(
                {
                    "col1": [0, 1, 2, 3, 4],
                    "col3": ["a", "a", "b", "b", "a"],
                    "col4": ["a, b", "a", "c", "a, c", "b"],
                    "id_col2": [0, 1, 2, 3, 4],
                }
            ),
        ),
        (
            "col3",
            {"col3": pd.DataFrame({"col3": ["a", "b"]})},
            pd.DataFrame(
                {
                    "col1": [0, 1, 2, 3, 4],
                    "col2": ["a", "b", "c", "d", "e"],
                    "col4": ["a, b", "a", "c", "a, c", "b"],
                    "id_col3": [0, 0, 1, 1, 0],
                }
            ),
        ),
        (
            "col4",
            {"col4": pd.DataFrame({"col4": ["a, b", "a", "c", "a, c", "b"]})},
            pd.DataFrame(
                {
                    "col1": [0, 1, 2, 3, 4],
                    "col2": ["a", "b", "c", "d", "e"],
                    "col3": ["a", "a", "b", "b", "a"],
                    "id_col4": [0, 1, 2, 3, 4],
                }
            ),
        ),
        (
            ["col3", "col4"],
            {
                "col3": pd.DataFrame({"col3": ["a", "b"]}),
                "col4": pd.DataFrame({"col4": ["a, b", "a", "c", "a, c", "b"]}),
            },
            pd.DataFrame(
                {
                    "col1": [0, 1, 2, 3, 4],
                    "col2": ["a", "b", "c", "d", "e"],
                    "id_col3": [0, 0, 1, 1, 0],
                    "id_col4": [0, 1, 2, 3, 4],
                }
            ),
        ),
        (
            {"col4": ","},
            {"col4": pd.DataFrame({"col4": ["a", "b", "c"]})},
            pd.DataFrame(
                {
                    "col1": [0, 1, 2, 3, 4],
                    "col2": ["a", "b", "c", "d", "e"],
                    "col3": ["a", "a", "b", "b", "a"],
                    "id_col4": [[0, 1], [0], [2], [0, 2], [1]],
                }
            ),
        ),
        (
            {"col3": None, "col4": ","},
            {
                "col3": pd.DataFrame({"col3": ["a", "b"]}),
                "col4": pd.DataFrame({"col4": ["a", "b", "c"]}),
            },
            pd.DataFrame(
                {
                    "col1": [0, 1, 2, 3, 4],
                    "col2": ["a", "b", "c", "d", "e"],
                    "id_col3": [0, 0, 1, 1, 0],
                    "id_col4": [[0, 1], [0], [2], [0, 2], [1]],
                }
            ),
        ),
    ],
)
def test_split(param, gab_bases, gab):
    df = pd.DataFrame(
        {
            "col1": [0, 1, 2, 3, 4],
            "col2": ["a", "b", "c", "d", "e"],
            "col3": ["a", "a", "b", "b", "a"],
            "col4": ["a, b", "a", "c", "a, c", "b"],
        }
    )

    bases, out = split(df, param)

    for col in gab_bases.keys():
        gab_base = gab_bases[col]
        gab_base.index.name = "id"
        assert_frame_equal(bases[col], gab_base)
    assert_frame_equal(out, gab)


def test_split_recommendation():
    mock_db = pd.DataFrame(
        {
            "User": [1, 2, 3, 1, 2, 3, 2, 2, 1, 3],
            "Item": [2, 3, 1, 1, 1, 2, 3, 3, 2, 1],
            "Colour": ["b", "w", "g", "y", "y", "b", "p", "g", "w", "o"],
        }
    )
    gab_user = pd.DataFrame(data={"User": [1, 2, 3]}, index=[0, 1, 2])
    gab_user.index.name = "test_Users"

    gab_item = pd.DataFrame(
        data={
            "test_Item": [0, 2, 0, 1, 1, 1, 2, 2],
            "Colour": ["b", "y", "w", "w", "p", "g", "g", "o"],
        },
        index=[0, 1, 2, 3, 5, 6, 7, 9],
    )
    gab_item.index.name = "test_Items"

    gab_main = pd.DataFrame(
        {
            "test_Users": [0, 2, 0, 1, 0, 1, 1, 1, 2, 2],
            "test_Items": [0, 0, 1, 1, 2, 3, 5, 6, 7, 9],
        }
    )
    gab_pre = pd.DataFrame({"Item": [2, 3, 1]})
    gab_pre.index.name = "id"

    main_bases, pretrained_bases = split_recommendation(
        dataframe=mock_db,
        split_config={"Users": ["User"], "Items": ["Item", "Colour"]},
        columns=["Item"],
        prefix="test_",
    )

    assert list(main_bases.keys()) == ["Users", "Items", "main"]
    assert list(pretrained_bases.keys()) == ["Item"]
    assert_frame_equal(main_bases["Users"], gab_user)
    assert_frame_equal(main_bases["Items"], gab_item)
    assert_frame_equal(main_bases["main"], gab_main)
    assert_frame_equal(pretrained_bases["Item"], gab_pre)
