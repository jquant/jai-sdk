#################
Check data sanity
#################

Validates consistency in the columns (columns_ref).

.. code-block:: python

    >>> import pandas as pd
    >>> from jai.functions.utils_funcs import process_predict
    >>>
    >>> j = Jai(AUTH_KEY)
    >>> results = j.sanity(name, data)
    >>> processed = process_predict(results)
    >>> pd.DataFrame(processed).sort_values('id')
              id   sanity_prediction    confidence_level (%)
       0       1               Valid                    70.9
       1       4             Invalid                    67.3
       2       7             Invalid                    80.6
       3      13               Valid                    74.2
