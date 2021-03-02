###############################
Resolution of duplicated values
###############################

Find possible duplicated values within the data.

.. code-block:: python

    >>> import pandas as pd
    >>> from jai.functions.utils_funcs import process_similar
    >>>
    >>> j = Jai(AUTH_KEY)
    >>> results = j.resolution(name, data)
    >>> processed = process_similar(results, return_self=True)
    >>> pd.DataFrame(processed).sort_values('query_id')
             query_id           id     distance
       0            1            2         0.11
       1            2            1         0.11
       2            3          NaN          NaN
       3            4            5         0.15