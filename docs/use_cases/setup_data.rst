#########################
Setting up your databases
#########################

For every database you setup, you'll need a name identifier so you can reuse it. 

Every item in a database should have an id, so we can easily identify each item without the need of manipulating the raw data except on the setup phase.
Make sure the id values are always unique.

.. note::
	Although some data types could be inserted structured in a :code:`list` or an :code:`numpy.ndarray`, we strongly recommend the use of :code:`pandas.Series` and :code:`pandas.DataFrame`, because of the structure with :code:`.index` attributes (or optionally the use of a column named :code:`'id'`, which has priority over the :code:`.index` attribute).

The model is used with the similarity queries and predicts (methods :code:`.similar()` and :code:`.predict()`). The similiarity query will return for each input, identified with the :code:`'query_id'`, the :code:`'id'` values of similiar items, the 'distance' in between them. The number of results is controlled by the :code:`'top_k'` parameter. The predict method will return for each input, identified with :code:`'id'`, the expected value for each item :code:`'predict'`.

.. note::
	The use of a column named :code:`'id'` will overwrite the pandas index attribute with :code:`.set_index('id')`. We consider a good pratice the strict usage of '.index' to identify items: 

	* the existence of both :code:`'id'` column and :code:`'.index'` could cause ambiguity leading to misinterpretation results, 

	* it allows the usage of native pandas structures, e. g., indexing data with :code:`'.loc'`, 

	* better understanding of your data as the column :code:`'id'` will **NOT** be used for any model inferrence unlike any other columns of your data.

************************
Setup for Text type data
************************

For any uses of text-type data, data can be a list of texts, pandas Series or DataFrame.

* If data is a list, then the ids will be set with :code:`range(len(data_list))`.
* If data is a pandas type, then the ids will be the index values.

Setup applying NLP FastText model
=================================

This is an example of database with a fasttext model implementation. 

.. code-block:: python

    >>> name = 'text_data'
    >>> j.setup(name, data, db_type='FastText')

Setup applying NLP BERT model
=============================

The transformers model used by defaul is the BERT.
It's possible to generate a random name for identification of the base; it can be a user input.

.. code-block:: python

    >>> name = j.generate_name(12, prefix='sdk_', suffix='_text')

This time we choose :code:`db_type="Text"`, applying the pre-trained BERT model

.. code-block:: python

    >>> j.setup(name, data, db_type='Text')


Setup applying Edit Distance Model
==================================

It's also possible to use an model trained to reproduce the neighboring relation of the edit distance.

.. code-block:: python

    >>> j.setup(name, data, db_type='TextEdit')


*************************
Setup for Image type data
*************************

For any uses of image-type data, data should be first encoded before inserting into the Jai class.

.. code-block:: python

    >>> with open(filename, "rb") as image_file:
    >>>     encoded_string = base64.b64encode(image_file.read()).decode("utf-8")

Then the encoded string can be inserted into a list, pandas Series or DataFrame.
We provide :code:`read_image_folder` function for reading images from a local folder.

Setup applying Image Model
==========================

Images are processed using torchvision pretrained models.

.. code-block:: python

    >>> j.setup(name, data, db_type='Image')

***************************
Setup for Tabular type data
***************************

Setup applying Self-Supervised Model
====================================

.. code-block:: python

    >>> j.setup(name, data, db_type='Unsupervised')


Setup applying Supervised Model
===============================

.. code-block:: python

    >>> j.setup(name, data, db_type='Supervised', label={"task": "metric_classification", "label_name": "my_label"})

