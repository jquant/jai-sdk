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


Inherited from :code:`TaskBase`
===============================

:code:`name`
-----------------

.. code-block:: python

   >>> from jai import Query
   ...
   >>> q = Query()
   >>> q.name

:code:`db_type`
-----------------

.. code-block:: python

   >>> from jai import Query
   ...
   >>> q = Query()
   >>> q.db_type
   
:code:`is_valid`
-----------------

.. code-block:: python

   >>> from jai import Query
   ...
   >>> q = Query()
   >>> q.is_valid()

:code:`describe`
-----------------

.. code-block:: python

   >>> from jai import Query
   ...
   >>> q = Query()
   >>> q.describe()

   
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
