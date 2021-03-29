#######################
Checking your databases
#######################

Here are some methods to check your databases.

The name of your database should appear in:

.. code-block:: python

    >>> j.names
    ['jai_database', 'jai_selfsupervised', 'jai_supervised']

You can also check the types for each of your databases with:

.. code-block:: python

    >>> j.info
                            db_name           db_type
    0                  jai_database              Text
    1            jai_selfsupervised    SelfSupervised
    2                jai_supervised        Supervised

Or you can check if a given database name is valid:

.. code-block:: python

    >>> j.is_valid(name)
    True

Here you can check which ids are inserted in your database:

.. code-block:: python

    >>> j.ids(name)
    ['1000 items from 0 to 999']

