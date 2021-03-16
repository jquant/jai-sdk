######################
Filling missing values
######################

Fills the column in data with the most likely value given the other columns.

.. code-block:: python

    >>> import pandas as pd
    >>> from jai.processing import process_predict
    >>>
    >>> j = Jai(AUTH_KEY)
    >>> results = j.fill(name, data, COL_TO_FILL)
    >>> processed = process_predict(results)
    >>> pd.DataFrame(processed).sort_values('id')
              id     fill_prediction    confidence_level (%)
       0       1             value_1                    70.9
       1       4             value_1                    67.3
       2       7             value_1                    80.2
