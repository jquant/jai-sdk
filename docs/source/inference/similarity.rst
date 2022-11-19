################
Similarity Seach
################

After fitting your database, you can perform similarity searches in two ways: Based on an existing 
index of your already included model or using new data.

Using existing index
====================

You can query itens that already have been inputed by their :code:`ids`. The below example shows 
how to find the five most similar values for :code:`ids` 0 and 1.

.. code-block:: python

    >>> results = j.similar(name, [0, 1], top_k=5)

To find the 20 most similar values for every id from :code:`[0, 99]`.

.. code-block:: python

    >>> ids = list(range(100))
    >>> results = j.similar(name, ids, top_k=20)

Now, finding the 100 most similar values for every input value can be done like the example below.

.. code-block:: python

    >>> results = j.similar(name, data.index, top_k=100, batch_size=1024)

Using new data
==============

*(All data should be in* :code:`pandas.DataFrame` *or* :code:`pandas.Series` *format)*

Find the 100 most similar values for every :code:`new_data`.

.. code-block:: python

   >>> results = j.similar(name, new_data, top_k=100, batch_size=1024)

The output will be a list of dictionaries with :code:`'query_id'` being the id of the value you want 
to find similars and :code:`'results'`) a list with :code:`top_k` dictionaries with the :code:`'id'` 
and the :code:`'distance'` between :code:`'query_id'` and :code:`'id'`.

.. code-block:: bash

    [
        {
            'query_id': 0,
            'results':
            [
            {'id': 0, 'distance': 0.0},
            {'id': 3836, 'distance': 2.298321008682251},
            {'id': 9193, 'distance': 2.545339584350586},
            {'id': 832, 'distance': 2.5819168090820312},
            {'id': 6162, 'distance': 2.638622283935547},
            ...
            ]
        },
        ...,
        {
            'query_id': 9,
            'results':
            [
            {'id': 9, 'distance': 0.0},
            {'id': 54, 'distance': 5.262974262237549},
            {'id': 101, 'distance': 5.634262561798096},
            ...
            ]
        },
        ...
    ]


.. note::
    
    The method :code:`similar` has a default :code:`batch_size=2**20`, which will result in 
    :code:`ceil(n_samples/batch_size) + 2` requests. We **DON'T** recommend changing the default value 
    as it could reduce the performance of the API.

Output formating
================

There are two possible output formats for the similarity search.
You can change which format you wish to use by changing the parameter :code:`orient`.

  orient: "nested" or "flat"
            Changes the output format. `Default is "nested"`.

Here are some examples for each of the possible formats bellow:

- :code:`nested`:
  
.. code-block:: bash

    [
        {
            'query_id': 0,
            'results':
            [
            {'id': 0, 'distance': 0.0},
            {'id': 3836, 'distance': 2.298321008682251},
            {'id': 9193, 'distance': 2.545339584350586},
            {'id': 832, 'distance': 2.5819168090820312},
            {'id': 6162, 'distance': 2.638622283935547},
            ...
            ]
        },
        ...,
        {
            'query_id': 9,
            'results':
            [
            {'id': 9, 'distance': 0.0},
            {'id': 54, 'distance': 5.262974262237549},
            {'id': 101, 'distance': 5.634262561798096},
            ...
            ]
        },
        ...
    ]

- :code:`flat`:
  
.. code-block:: bash

    [
        {'query_id': 0, 'id': 0, 'distance': 0.0},
        {'query_id': 0, 'id': 3836, 'distance': 2.298321008682251},
        {'query_id': 0, 'id': 9193, 'distance': 2.545339584350586},
        {'query_id': 0, 'id': 832, 'distance': 2.5819168090820312},
        {'query_id': 0, 'id': 6162, 'distance': 2.638622283935547},
        ...
        {'query_id': 9, 'id': 9, 'distance': 0.0},
        {'query_id': 9, 'id': 54, 'distance': 5.262974262237549},
        {'query_id': 9, 'id': 101, 'distance': 5.634262561798096},
        ...
    ]