
#############
Linear Module
#############

.. note::
   If you haven't yet, please check the section :ref:`How to configure your auth key <source/1-overview/set_authentication:How to configure your auth key>` 
   for more information.

The Linear Models module is built to train and consume linear models your Jai environment.

The methods chosen for this module are exclusively to help you train and consume your linear models in Jai.

For more information, see the full :ref:`Linear Model class reference <source/reference/linear:linear model class>`.

:code:`LinearModel`
===================

.. code-block:: python

   >>> from jai import LinearModel
   ...
   >>> model = LinearModel()


:code:`set_parameters`
----------------------

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
