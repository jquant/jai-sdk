#############
Removing data
#############

After you're done with the model setup, you can instantly delete your raw data using 
:code:`j.delete_raw_data`.

.. code-block:: python

    # Delete the raw data inputed as it won't be needed anymore
    j.delete_raw_data(name='your_collection_name')

If you no longer need the model or anything else related to your database, you can use 
:code:`j.delete_database` to remove your collection from your JAI environment.

.. code-block:: python 
    
    j.delete_database(name='your_collection_name')

On the other hand, if you just need to remove some ids from your collection, you can use 
:code:`j.delete_ids` to perform this action.

.. code-block:: python

    j.delete_ids(name='your_collection_name', ids=[1, 2, 3, 4, 5])

See the :ref:`package reference <source/reference/jai:jai module>` for more information.