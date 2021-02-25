Running similarity
==================

After you're done setting up your database, you perform similarity searches:

- Using the indexes of the input data

.. code-block:: python

    # Find the 5 most similar values for ids 0 and 1
    results = j.similar(name, [0, 1], top_k=5)

    # Find the 20 most similar values for every id from [0, 99]
    ids = list(range(100))
    results = j.similar(name, ids, top_k=20)

    # Find the 100 most similar values for every input value
    results = j.similar(name, data.index, top_k=100, batch_size=1024)

- Using new data to be processed *(All data should be in pandas.DataFrame or pandas.Series format)*

.. code-block:: python

    # Find the 100 most similar values for every new_data
    results = j.similar(name, new_data, top_k=100, batch_size=1024)

The output will be a list of dictionaries with ("query_id") being the id of the value you want to find similars and ("results") a list with :code:`top_k` dictionaries with the "id" and the "distance" between "query_id" and "id".

.. code-block:: python

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

