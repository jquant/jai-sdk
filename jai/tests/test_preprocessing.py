import pytest
import pandas as pd

from pandas.testing import assert_frame_equal, assert_series_equal
from jai.core.preprocessing import split


# =============================================================================
# Tests for split
# =============================================================================
@pytest.mark.parametrize('param, gab_bases, gab', [
    ('col2', {
        "col2": pd.DataFrame({"col2": ["a", "b", "c", "d", "e"]})
    },
     pd.DataFrame({
         "col1": [0, 1, 2, 3, 4],
         "col3": ["a", "a", "b", "b", "a"],
         "col4": ["a, b", "a", "c", "a, c", "b"],
         "id_col2": [0, 1, 2, 3, 4]
     })),
    ('col3', {
        "col3": pd.DataFrame({"col3": ["a", "b"]})
    },
     pd.DataFrame({
         "col1": [0, 1, 2, 3, 4],
         "col2": ["a", "b", "c", "d", "e"],
         "col4": ["a, b", "a", "c", "a, c", "b"],
         "id_col3": [0, 0, 1, 1, 0]
     })),
    ('col4', {
        "col4": pd.DataFrame({"col4": ["a, b", "a", "c", "a, c", "b"]})
    },
     pd.DataFrame({
         "col1": [0, 1, 2, 3, 4],
         "col2": ["a", "b", "c", "d", "e"],
         "col3": ["a", "a", "b", "b", "a"],
         "id_col4": [0, 1, 2, 3, 4]
     })),
    (['col3', 'col4'], {
        "col3": pd.DataFrame({"col3": ["a", "b"]}),
        "col4": pd.DataFrame({"col4": ["a, b", "a", "c", "a, c", "b"]})
    },
     pd.DataFrame({
         "col1": [0, 1, 2, 3, 4],
         "col2": ["a", "b", "c", "d", "e"],
         "id_col3": [0, 0, 1, 1, 0],
         "id_col4": [0, 1, 2, 3, 4]
     })),
    ({
        'col4': ','
    }, {
        "col4": pd.DataFrame({"col4": ["a", "b", "c"]})
    },
     pd.DataFrame({
         "col1": [0, 1, 2, 3, 4],
         "col2": ["a", "b", "c", "d", "e"],
         "col3": ["a", "a", "b", "b", "a"],
         "id_col4": [[0, 1], [0], [2], [0, 2], [1]]
     })),
    ({
        'col3': None,
        'col4': ','
    }, {
        "col3": pd.DataFrame({"col3": ["a", "b"]}),
        "col4": pd.DataFrame({"col4": ["a", "b", "c"]})
    },
     pd.DataFrame({
         "col1": [0, 1, 2, 3, 4],
         "col2": ["a", "b", "c", "d", "e"],
         "id_col3": [0, 0, 1, 1, 0],
         "id_col4": [[0, 1], [0], [2], [0, 2], [1]]
     })),
])
def test_split(param, gab_bases, gab):

    df = pd.DataFrame({
        "col1": [0, 1, 2, 3, 4],
        "col2": ["a", "b", "c", "d", "e"],
        "col3": ["a", "a", "b", "b", "a"],
        "col4": ["a, b", "a", "c", "a, c", "b"]
    })

    bases, out = split(df, param)

    for col in gab_bases.keys():
        gab_base = gab_bases[col]
        gab_base.index.name = "id"
        assert_frame_equal(bases[col], gab_base)
    assert_frame_equal(out, gab)
