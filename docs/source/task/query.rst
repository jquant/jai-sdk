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

Bellow, a simple example to instanciate the Query class:

.. code-block:: python

   >>> from jai import Query
   ...
   >>> q = Query()


:code:`similar`
----------------------

.. code-block:: python

   >>> from jai import Query
   ...
   >>> q = Query()
   >>> q.similar()

:code:`recommendation`
----------------------

.. code-block:: python

   >>> from jai import Query
   ...
   >>> q = Query()
   >>> q.recommendation()

:code:`predict`
---------------

.. code-block:: python

   >>> from jai import Query
   ...
   >>> q = Query()
   >>> q.predict()


   
:code:`fields`
-----------------

.. code-block:: python

   >>> from jai import Query
   ...
   >>> q = Query()
   >>> q.fields()


      
:code:`download_vectors`
------------------------

.. code-block:: python

   >>> from jai import Query
   ...
   >>> q = Query()
   >>> q.download_vectors()

         
:code:`filters`
-----------------

.. code-block:: python

   >>> from jai import Query
   ...
   >>> q = Query()
   >>> q.filters()


:code:`ids`
-----------------

.. code-block:: python

   >>> from jai import Query
   ...
   >>> q = Query()
   >>> q.ids()

Inherited from :code:`TaskBase`
===============================

:code:`name`
-----------------

This attribute contains the value of the database's name.

.. code-block:: python

   >>> from jai import Query
   ...
   >>> q = Query()
   >>> q.name

:code:`db_type`
-----------------

This attribute returns the type of the database.

.. code-block:: python

   >>> from jai import Query
   ...
   >>> q = Query()
   >>> q.db_type
   
:code:`is_valid`
-----------------

This method returns a boolean indicating if the database exists or not.

.. code-block:: python

   >>> from jai import Query
   ...
   >>> q = Query()
   >>> q.is_valid()

:code:`describe`
-----------------

This method returns the full configuration information of the database.

.. code-block:: python

   >>> from jai import Query
   ...
   >>> q = Query()
   >>> q.describe()
