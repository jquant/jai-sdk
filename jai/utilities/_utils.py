from functools import cmp_to_key
from operator import itemgetter


def cmp(x, y):
    """
    Replacement for built-in function cmp that was removed in Python 3

    Compare the two objects x and y and return an integer according to
    the outcome. The return value is negative if x < y, zero if x == y
    and strictly positive if x > y.

    https://portingguide.readthedocs.io/en/latest/comparisons.html#the-cmp-function
    """

    return (x > y) - (x < y)


def multikeysort(items, columns):
    """
    Sort a list of dictionaries.

    https://stackoverflow.com/a/1144405
    https://stackoverflow.com/a/73050

    Parameters
    ----------
    items : list of dictionaries
        list of dictionaries to be sorted.
    columns : list of strings
        list of key names to be sorted on the order of the sorting. add '-' at
        the start of the name if it should be sorted from high to low.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    comparers = [
        (
            (itemgetter(col[1:].strip()), -1)
            if col.startswith("-")
            else (itemgetter(col.strip()), 1)
        )
        for col in columns
    ]

    def comparer(left, right):
        comparer_iter = (cmp(fn(left), fn(right)) * mult for fn, mult in comparers)
        return next((result for result in comparer_iter if result), 0)

    return sorted(items, key=cmp_to_key(comparer))
