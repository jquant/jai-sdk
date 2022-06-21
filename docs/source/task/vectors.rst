
##############
Vectors Module
##############

.. note::
   If you haven't yet, please check the section 
   :ref:`How to configure your auth key <source/overview/set_authentication:How to configure your auth key>` 
   before running the code snippets in this page.

The Vectors module is built to setup collections without models internally and making it possible to consume them in your Jai environment.

For more information, see the full :ref:`Vectors class reference <source/reference/task_vectors:vectors class>`.

:code:`Vectors`
===================

Bellow, a simple example to instanciate the Vectors class:

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

This attribute contains the value of the database's name.

.. code-block:: python

   >>> from jai import Vectors
   ...
   >>> vectors = Vectors()
   >>> vectors.name

:code:`db_type`
-----------------

This attribute returns the type of the database.

.. code-block:: python

   >>> from jai import Vectors
   ...
   >>> vectors = Vectors()
   >>> vectors.db_type
   
:code:`is_valid`
-----------------

This method returns a boolean indicating if the database exists or not.

.. code-block:: python

   >>> from jai import Vectors
   ...
   >>> vectors = Vectors()
   >>> vectors.is_valid()

:code:`describe`
-----------------

This method returns the full configuration information of the database.

.. code-block:: python

   >>> from jai import Vectors
   ...
   >>> vectors = Vectors()
   >>> vectors.describe()
