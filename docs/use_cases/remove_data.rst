#############
Removing data
#############

After you're done with the model setup, you can delete your raw data

.. code-block:: python

    # Delete the raw data inputed as it won't be needed anymore
    >>> j.delete_raw_data(name)

If you no longer need the model or anything else related to your database:

.. code-block:: python 
    
    >>> j.delete_database(name)

