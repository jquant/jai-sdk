
##############
Trainer Module
##############

.. note::
   If you haven't yet, please check the section :ref:`How to configure your auth key <source/1-overview/set_authentication:How to configure your auth key>` 
   for more information.


The Trainer module is built to setup new collections on your Jai environment.

The methods chosen for this module are exclusively to help you train and setup a vector collection in Jai.
After the setup process, use the :ref:`Query Module <source/4-task/query:query module>` to consume the model.

For more information, see the full :ref:`Trainer class reference <source/reference/trainer:trainer class>`.


:code:`Trainer`
===============

.. code-block:: python

   >>> from jai import Trainer
   ...
   >>> trainer = Trainer()


:code:`set_parameters`
----------------------

.. code-block:: python

   >>> from jai import Trainer
   ...
   >>> trainer = Trainer()
   >>> trainer.set_parameters()

:code:`fit`
-----------

.. code-block:: python

   >>> from jai import Trainer
   ...
   >>> trainer = Trainer()
   >>> trainer.fit()

:code:`append`
--------------

.. code-block:: python

   >>> from jai import Trainer
   ...
   >>> trainer = Trainer()
   >>> trainer.append()

:code:`report`
--------------

.. code-block:: python

   >>> from jai import Trainer
   ...
   >>> trainer = Trainer()
   >>> trainer.append()

:code:`delete_ids`
------------------

.. code-block:: python

   >>> from jai import Trainer
   ...
   >>> trainer = Trainer()
   >>> trainer.append()

:code:`delete_raw_data`
-----------------------

.. code-block:: python

   >>> from jai import Trainer
   ...
   >>> trainer = Trainer()
   >>> trainer.append()

:code:`delete_database`
-----------------------

.. code-block:: python

   >>> from jai import Trainer
   ...
   >>> trainer = Trainer()
   >>> trainer.append()

:code:`get_query`
-----------------

.. code-block:: python

   >>> from jai import Trainer
   ...
   >>> trainer = Trainer()
   >>> trainer.get_query()

Inherited from :code:`TaskBase`
===============================

:code:`name`
-----------------

.. code-block:: python

   >>> from jai import Trainer
   ...
   >>> trainer = Trainer()
   >>> trainer.name

:code:`db_type`
-----------------

.. code-block:: python

   >>> from jai import Trainer
   ...
   >>> trainer = Trainer()
   >>> trainer.db_type
   
:code:`is_valid`
-----------------

.. code-block:: python

   >>> from jai import Trainer
   ...
   >>> trainer = Trainer()
   >>> trainer.is_valid()

:code:`describe`
-----------------

.. code-block:: python

   >>> from jai import Trainer
   ...
   >>> trainer = Trainer()
   >>> trainer.describe()

   
:code:`fields`
-----------------

.. code-block:: python

   >>> from jai import Trainer
   ...
   >>> trainer = Trainer()
   >>> trainer.fields()


      
:code:`download_vectors`
------------------------

.. code-block:: python

   >>> from jai import Trainer
   ...
   >>> trainer = Trainer()
   >>> trainer.download_vectors()

         
:code:`filters`
-----------------

.. code-block:: python

   >>> from jai import Trainer
   ...
   >>> trainer = Trainer()
   >>> trainer.filters()


:code:`ids`
-----------------

.. code-block:: python

   >>> from jai import Trainer
   ...
   >>> trainer = Trainer()
   >>> trainer.ids()
