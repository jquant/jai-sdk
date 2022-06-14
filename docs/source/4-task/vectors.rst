
##############
Vectors Module
##############

.. note::
   If you haven't yet, please check the section :ref:`How to configure your auth key <source/1-overview/set_authentication:How to configure your auth key>` 
   for more information.

The Vectors module is built to setup collections without models internally and making it possible to consume them in your Jai environment.

For more information, see the full :ref:`Vectors class reference <source/reference/vectors:vectors class>`.

:code:`Vectors`
===================

.. code-block:: python

   >>> from jai import Vectors
   ...
   >>> vectors = Vectors()


:code:`insert_vectors`
----------------------

.. code-block:: python

   >>> from jai import Vectors
   ...
   >>> vectors = Vectors()
   >>> vectors.insert_vectors()

:code:`delete_raw_data`
-----------------------

.. code-block:: python

   >>> from jai import Vectors
   ...
   >>> vectors = Vectors()
   >>> vectors.delete_raw_data()

:code:`delete_database`
-----------------------

.. code-block:: python

   >>> from jai import Vectors
   ...
   >>> vectors = Vectors()
   >>> vectors.delete_database()


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
