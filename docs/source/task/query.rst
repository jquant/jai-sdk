############
Query Module
############

.. note::
   If you haven't yet, please check the section 
   :ref:`How to configure your auth key <source/overview/set_authentication:How to configure your auth key>` 
   before running the code snippets in this page.

The Query module is built to consume existing collections on your Jai environment.

The methods of this module are exclusively to consume existing collections in Jai.

For more information, see the full :ref:`Query class reference <source/reference/task_query:query class>`.

:code:`Query`
===============

Bellow, a simple example to instantiate the Query class:

.. code-block:: python

   >>> from jai import Query
   ...
   >>> q = Query(name)


:code:`similar`
----------------------

Performs a similarity search in the collection.

Data can be raw data or a set of ids. If it's a set of ids, the ids must exist in the collection.

.. code-block:: python

   >>> from jai import Query
   ...
   >>> q = Query(name)
   >>> q.similar(data)

:code:`recommendation`
----------------------

This is only available for Recommendation type databases.

Performs a recommendation search in the collection.

Data can be raw data or a set of ids. If it's a set of ids, the ids must exist in the collection.

Returns the ids of the database selected, therefore the data/ids input must be from the twin database.

.. code-block:: python

   >>> from jai import Query
   ...
   >>> q = Query(name)
   >>> q.recommendation(data)

:code:`predict`
---------------

This is only available for Supervised type databases.
Perform a prediction to the raw data.

.. code-block:: python

   >>> from jai import Query
   ...
   >>> q = Query(name)
   >>> q.predict(data)


   
:code:`fields`
-----------------

Return information about the fields that are expected as input.

.. code-block:: python

   >>> from jai import Query
   ...
   >>> q = Query(name)
   >>> q.fields()


      
:code:`download_vectors`
------------------------

Returns an array with the vectors from the database.

.. code-block:: python

   >>> from jai import Query
   ...
   >>> q = Query(name)
   >>> q.download_vectors()

         
:code:`filters`
-----------------

Returns the list of filters if any.

.. code-block:: python

   >>> from jai import Query
   ...
   >>> q = Query(name)
   >>> q.filters()


:code:`ids`
-----------------

Returns the list of ids in the database.

.. code-block:: python

   >>> from jai import Query
   ...
   >>> q = Query(name)
   >>> q.ids()

Inherited from :code:`TaskBase`
===============================

:code:`name`
-----------------

This attribute contains the value of the database's name.

.. code-block:: python

   >>> from jai import Query
   ...
   >>> q = Query(name)
   >>> q.name

:code:`db_type`
-----------------

This attribute returns the type of the database.

.. code-block:: python

   >>> from jai import Query
   ...
   >>> q = Query(name)
   >>> q.db_type
   
:code:`is_valid`
-----------------

This method returns a boolean indicating if the database exists or not.

.. code-block:: python

   >>> from jai import Query
   ...
   >>> q = Query(name)
   >>> q.is_valid()

:code:`describe`
-----------------

This method returns the full configuration information of the database.

.. code-block:: python

   >>> from jai import Query
   ...
   >>> q = Query(name)
   >>> q.describe()
