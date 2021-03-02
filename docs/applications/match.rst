#################################
Matching values from two datasets
#################################

Match two datasets with their possible equal values.

Queries the `data right` to get the similar results in `data left`.

.. code-block:: python

    >>> import pandas as pd
    >>> from jai.functions.utils_funcs import process_similar
    >>>
    >>> j = Jai(AUTH_KEY)
    >>> results = j.match(name, data1, data2)
    >>> processed = process_similar(results, return_self=True)
    >>> pd.DataFrame(processed).sort_values('query_id')
    >>> # query_id is from data_right and id is from data_left
             query_id           id     distance
       0            1            2         0.11
       1            2            1         0.11
       2            3          NaN          NaN
       3            4          NaN          NaN
       4            5            5         0.15