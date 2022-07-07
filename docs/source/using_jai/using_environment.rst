
#############################
Working with your environment
#############################

Sometimes, you may need to have more information about your environment data in JAI. 
For this, JAI owns some managing methods that are shown in this section.

:code:`j.names`
---------------

If you need to know which databases are in your environment, use :code:`j.names`. 
It'll show you the names of all your databases in JAI.

.. code-block:: python

    >>> # Example response supposing these collections are in the environment:
    >>> j.names

    ['jai_database', 'jai_selfsupervised', 'jai_supervised']

:code:`j.info`
--------------

To get more information about your collections, use :code:`j.info`. 
This method will present all your collection names, types,  last modifications, 
dependencies and sizes.

.. code-block:: python

    >>> j.info

                  name|           type|    last modified|          dependencies|   size|
    ------------------|---------------|-----------------|----------------------|-------|
          jai_database|           Text| 2021-10-26-20h22|                    []|  20640|
    jai_selfsupervised| SelfSupervised| 2021-11-03-17h50|                    []|   8500|
        jai_supervised|     Supervised| 2021-11-03-17h50|  [jai_selfsupervised]|   8500|
    ------------------------------------------------------------------------------------

The drawback of :code:`j.info` is that it can run slowly depending on the number of collections in your environment.

:code:`j.fields`
----------------

If you forgot what columns your database has, this information could be accessed by :code:`j.fields` method.

.. code-block:: python

    >>> # Use your collection name
    >>> j.fields(name='jai_database')

    {'id': 'int64',
     'Column1': 'float64',
     'Column2': 'float64',
     'Column3': 'float64',
     'Column4': 'float64'}

:code:`j.get_dtype`
-------------------

To get what collection type your collection is, use :code:`j.get_dtype`.

.. code-block:: python

    >>> # Use your collection name
    >>> j.get_dtype(name='jai_selfsupervised')

    'SelfSupervised'

:code:`j.describe`
------------------

However, if you need details of what parameters you choose to fit your collection, :code:`j.describe` can bring it for you.

.. code-block:: python

    >>> # Use your collection name
    >>> j.describe(name='jai_database')

    {'name': 'california',
     'dtype': 'SelfSupervised',
     'state': 'active',
     'version': '2021-10-26-20h22',
     'has_filter': False,
     'model_hyperparams': {'batch_size': 512,
     'learning_rate': 0.01,
     'encoder_layer': '2LM',
     'decoder_layer': '2LM',
     'hidden_latent_dim': 64,
     'dropout_rate': 0.1,
     'momentum': 0.1,
     'pretraining_ratio': 0.1,
     'noise_level': 0.0,
     'training_type': 'contrastive'}
     ...

:code:`j.report`
----------------

To recover the fit report for your collection, use :code:`j.report`.

.. code-block:: python

    >>> # Use your collection name
    >>> j.report(name='jai_database')


:code:`j.ids`
-------------

If you need to remember how many ids your collection have, use :code:`j.ids`.

.. code-block:: python

    >>> # Use your collection name
    >>> j.ids(name='jai_database', mode='summarized') # default

    ['20640 items from 0 to 20639']

For more information about how to work with your environment, check :ref:`JAI Python Class <source/reference/jai_class:JAI Python Class>`