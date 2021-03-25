
############
Applications
############

Here are some applications developed using the basic operations, to allow a more direct approach to the specific goals listed bellow. They should work as standalone without the need of any of the basic operations

*********************************
Matching values from two datasets
*********************************

Match two datasets with their possible equal values. This method matches similar values in between text columns of two databases. It queries *data right* to get the similar results in *data left*.

You can find a complete `match example here <https://github.com/jquant/jai-sdk/blob/main/examples/match_example.ipynb>`_.



.. code-block:: python

    >>> data1, data2 = dataframe1['name'], dataframe2['name']
    >>>
    >>> j = Jai(AUTH_KEY)
    >>> match = j.match(name, data1, data2)
    >>> match
              id_left     id_right     distance
       0            1            2         0.11
       1            2            1         0.11
       2            3          NaN          NaN
       3            4          NaN          NaN
       4            5            5         0.15

*******************************
Resolution of duplicated values
*******************************

Find possible duplicated values within the data. This method finds similar values in text columns of your database.

You can find a complete `resoultion example here <https://github.com/jquant/jai-sdk/blob/main/examples/resolution_example.ipynb>`_.

.. code-block:: python

    >>> data = dataframe['name']
    >>> j = Jai(AUTH_KEY)
    >>> results = j.resolution(name, data)
    >>> results
      id  resolution_id
       0              0
       1              0
       2              0
       3              3
       4              3
       5              5

**********************
Filling missing values
**********************

Fills the column in data with the most likely value given the other columns.

You can find a complete `fill example here <https://github.com/jquant/jai-sdk/blob/main/examples/fill_example.ipynb>`_.

.. code-block:: python

    >>> import pandas as pd
    >>> from jai.processing import process_predict
    >>>
    >>> j = Jai(AUTH_KEY)
    >>> results = j.fill(name, data, COL_TO_FILL)
    >>> processed = process_predict(results)
    >>> pd.DataFrame(processed).sort_values('id')
              id   sanity_prediction    confidence_level (%)
       0       1             value_1                    70.9
       1       4             value_1                    67.3
       2       7             value_1                    80.2
       
*****************
Check data sanity
*****************

Validates consistency in the columns (columns_ref).

You can find a complete `sanity example here <https://github.com/jquant/jai-sdk/blob/main/examples/sanity_example.ipynb>`_.

.. code-block:: python

    >>> import pandas as pd
    >>> from jai.processing import process_predict
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
