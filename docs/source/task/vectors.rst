
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

Bellow, a simple example to instantiate the Vectors class:

.. code-block:: python

   >>> from jai import Vectors
   ...
   >>> vectors = Vectors(name)


:code:`insert_vectors`
----------------------
Insert raw vectors database directly into JAI without any need of fit.

.. code-block:: python

   >>> from jai import Vectors
   ...
   >>> vectors = Vectors(name)
   >>> vectors.insert_vectors(data)

:code:`delete_raw_data`
-----------------------

Removes any remaining raw data that might be stored.

.. code-block:: python

   >>> from jai import Vectors
   ...
   >>> vectors = Vectors(name)
   >>> vectors.delete_raw_data()

:code:`delete_database`
-----------------------

Removes the collection.

.. code-block:: python

   >>> from jai import Vectors
   ...
   >>> vectors = Vectors(name)
   >>> vectors.delete_database()


Inherited from :code:`TaskBase`
===============================

:code:`name`
-----------------

This attribute contains the value of the database's name.

.. code-block:: python

   >>> from jai import Vectors
   ...
   >>> vectors = Vectors(name)
   >>> vectors.name

:code:`db_type`
-----------------

This attribute returns the type of the database.

.. code-block:: python

   >>> from jai import Vectors
   ...
   >>> vectors = Vectors(name)
   >>> vectors.db_type
   
:code:`is_valid`
-----------------

This method returns a boolean indicating if the database exists or not.

.. code-block:: python

   >>> from jai import Vectors
   ...
   >>> vectors = Vectors(name)
   >>> vectors.is_valid()

:code:`describe`
-----------------

This method returns the full configuration information of the database.

.. code-block:: python

   >>> from jai import Vectors
   ...
   >>> vectors = Vectors(name)
   >>> vectors.describe()
