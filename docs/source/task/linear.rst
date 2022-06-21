
#############
Linear Module
#############

.. note::
   If you haven't yet, please check the section 
   :ref:`How to configure your auth key <source/overview/set_authentication:How to configure your auth key>` 
   before running the code snippets in this page.


The Linear Models module is built to train and consume linear models your Jai environment.

The methods of this module are exclusively to help you train and consume your linear models in Jai.

For more information, see the full :ref:`Linear Model class reference <source/reference/task_linear:linear model class>`.

:code:`LinearModel`
===================

Bellow, a simple example to instanciate the LinearModel class:

.. code-block:: python

   >>> from jai import LinearModel
   ...
   >>> model = LinearModel()


:code:`set_parameters`
----------------------

First step to train a new linear model is to define its parameters.

.. code-block:: python

   >>> from jai import LinearModel
   ...
   >>> model = LinearModel()
   >>> model.set_parameters()

:code:`fit`
----------------------

.. code-block:: python

   >>> from jai import LinearModel
   ...
   >>> model = LinearModel()
   >>> model.fit()

:code:`learn`
---------------

.. code-block:: python


   >>> from jai import LinearModel
   ...
   >>> model = LinearModel()
   >>> model.learn()

:code:`predict`
---------------

.. code-block:: python


   >>> from jai import LinearModel
   ...
   >>> model = LinearModel()
   >>> model.predict()

Inherited from :code:`TaskBase`
===============================

:code:`name`
-----------------

This attribute contains the value of the database's name.

.. code-block:: python

   >>> from jai import LinearModel
   ...
   >>> model = LinearModel()
   >>> model.name

:code:`db_type`
-----------------

This attribute returns the type of the database.

.. code-block:: python

   >>> from jai import LinearModel
   ...
   >>> model = LinearModel()
   >>> model.db_type
   
:code:`is_valid`
-----------------

This method returns a boolean indicating if the database exists or not.

.. code-block:: python

   >>> from jai import LinearModel
   ...
   >>> model = LinearModel()
   >>> model.is_valid()

:code:`describe`
-----------------

This method returns the full configuration information of the database.

.. code-block:: python

   >>> from jai import LinearModel
   ...
   >>> model = LinearModel()
   >>> model.describe()

   