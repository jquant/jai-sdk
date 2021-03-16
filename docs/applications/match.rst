#################################
Matching values from two datasets
#################################

Match two datasets with their possible equal values.

Queries the `data right` to get the similar results in `data left`.

This method matches similar values in between text columns of two databases.

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