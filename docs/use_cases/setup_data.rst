###########################
Fit: Creating Collections
###########################

JAI works by creating feature-rich representations based on latent vectors, and storing them efficiently in what is called **collections**.

.. note::
    A **Collection** is always created by the **Setup** method.

.. note::
    Since version 0.10.0, the methods :code:`fit` was added with the same purpose as :code:`setup` and the method :code:`append` as :code:`add_data`.

******
Basics
******

The Fit Method
================

* The fit method is JAI's most important and central piece - it is responsible for sending and transforming raw data into vectors and then creating and indexing them into collections.

* By changing the "db_type" argument, JAI can process Text Documents, Images, Structured (Tabular) data into vectors.

* Using the "mycelia_bases" argument, users can combine any kind of data to create rich, multi-modal and hierarchical representations.

.. code:: python

	j.fit(
		name='Collection_Name',
		data=data,
		db_type='SelfSupervised'
	)

**Main Parameters**

* name (str) – Collection name - Used to reference the collection for similarity, inference and also in the REST API path. Max len: 32 characters.

* data (pandas.DataFrame or pandas.Series) – Data to be inserted and used for training.
    
.. note::
    JAI collections are always pairs of (id, vector), where the 'id' is always an integer, either inferred from the pandas.Dataframe or pandas.Series index or from an explicitly declared "id" columns.

    * Inserting data with a columns named 'id' will overwrite the default data index and will be used instead.
 
.. note::
        It is recommended as a good JAI practice to always use the pandas native index:
        
        * It avoids the possibility of having more than one data 'index' 
        * It enables the native usage of '.loc' commands, making the use of JAI responses easier.

* db_type (str) – Database type [Supervised, SelfSupervised, Text, FastText, TextEdit, Image]


.. warning::
    After extracting the latent vectors, all the raw data sent to jai is deleted. There is no way to query raw data on JAI.

***********************
Fit for Table data
***********************

Fit applying Self-Supervised Model
====================================

.. code-block:: python

    >>> j.fit(name, data, db_type='SelfSupervised')


*************************
Fit for Text data (NLP)
*************************

For any uses of text-type data, data can be a list of texts, pandas Series or DataFrame.

* If data is a list, then the ids will be set with :code:`range(len(data_list))`.
* If data is a pandas type, then the ids will be set as described above.

Fit applying NLP FastText model
=================================

This is an example of database with a fasttext model implementation. 

.. code-block:: python

    >>> name = 'text_data'
    >>> j.fit(name, data, db_type='FastText')

Fit applying NLP BERT model
=============================

The transformers model used by defaul is the BERT.
It's possible to generate a random name for identification of the base; it can be a user input.

.. code-block:: python

    >>> name = j.generate_name(12, prefix='sdk_', suffix='_text')

This time we choose :code:`db_type="Text"`, applying the pre-trained BERT model

.. code-block:: python

    >>> j.fit(name, data, db_type='Text')


Fit applying Edit Distance Model
==================================

It's also possible to use an model trained to reproduce the neighboring relation of the edit distance.

.. code-block:: python

    >>> j.fit(name, data, db_type='TextEdit')


********************
Fit for Image data
********************

For any uses of image-type data, data should be encoded before inserting it into the Jai class.

.. code-block:: python

    >>> with open(filename, "rb") as image_file:
    >>>     encoded_string = base64.b64encode(image_file.read()).decode("utf-8")

The encoded string can then be inserted into a list, pandas Series or DataFrame.
We provide :code:`read_image_folder` and :code:`resize_image_folder` functions for reading and resizing images from a local folder.
Resizing images before inserting is recommended because it reduces writing, reading and processing time during model inference.

Fit applying Image Model
==========================

Images are processed using torchvision pretrained models.

.. code-block:: python

    >>> j.fit(name, data, db_type='Image')

.. note::
    The method :code:`fit` has a default :code:`batch_size=16384`, which will result in a total of :code:`ceil(n_samples/batch_size) + n + 5` requests, where :code:`n = ceil(training_time/frequency_seconds)` is a variable number depending on the time it takes to finish the setup.
    We do NOT recommend changing the :code:`batch_size` default value as it could reduce the performance of the API. 
    As for the :code:`frequency_seconds`, it could be changed affecting only the frequecy of the progress bar's updates. If :code:`frequency_seconds < 1`, then there will be no progress bar printed, requiring the user to interpret the response from :code:`j.status`.