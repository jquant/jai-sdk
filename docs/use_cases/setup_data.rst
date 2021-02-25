Setting up your databases
=========================

All data should be in pandas.DataFrame or pandas.Series format

Setup applying NLP FastText model
---------------------------------

.. code-block:: python

    ### fasttext implementation
    # save this if you want to work in the same database later
    name = 'text_data'

    ### Insert data and train the FastText model
    # data can be a list of texts, pandas Series or DataFrame.
    # if data is a list, then the ids will be set with range(len(data_list))
    # if data is a pandas type, then the ids will be the index values.
    # heads-up: index values must not contain duplicates.
    j.setup(name, data, db_type='FastText')

    # wait for the training to finish
    j.wait_setup(name, 10)

Setup applying NLP BERT model
-----------------------------

.. code-block:: python

    ### BERT implementation
    # generate a random name for identification of the base; it can be a user input
    name = j.generate_name(20, prefix='sdk_', suffix='_text')

    # this time we choose db_type="Text", applying the pre-trained BERT model
    j.setup(name, data, db_type='Text', batch_size=1024)
    j.wait_setup(name, 10)

