###############################
Resolution of duplicated values
###############################

Find possible duplicated values within the data.

This method finds similar values in text columns of your database.

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