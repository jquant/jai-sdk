
##############
Trainer Module
##############

.. note::
   If you haven't yet, please check the section 
   :ref:`How to configure your auth key <source/overview/set_authentication:How to configure your auth key>` 
   before running the code snippets in this page.


The Trainer module is built to setup new collections on your Jai environment.

The methods of this module are exclusively to help you train and setup a vector collection in Jai.
After the setup process, use the :ref:`Query Module <source/task/query:query module>` to consume the model.

For more information, see the full :ref:`Trainer class reference <source/reference/task_trainer:trainer class>`.


:code:`Trainer`
===============

Bellow, a simple example to instanciate the Trainer class.

You'll need a name value as identifier for the database to be created.
This value should be unique (unless you wish to overwrite an existing database).
Here are the requirements for the name value:

- name length between 2 and 32.
- start with a lowercase letter (a-z).
- contain only lowercase letters (a-z), numbers (0-9) or `_`.
- end with a lowercase letter (a-z) or a number (0-9)

.. code-block:: python

   >>> from jai import Trainer
   ...
   >>> trainer = Trainer(name = "database")


:code:`set_parameters`
----------------------

First step to create a new collection is to define its parameters.

This method will check if the parameters are valid, warning you in case any of the parameters is not recognized by the API.
The only required parameter is :code:`db_type` to define the type of the collection you wish to create.
If none of the other parameters is given, then the default values are used.

For the complete reference of the possible parameters look at :ref:`fit kwargs <source/reference/fit_kwargs:fit kwargs (keyword arguments)>`.


.. code-block:: python

   >>> from jai import Trainer
   ...
   >>> trainer = Trainer()
   >>> trainer.set_parameters(db_type="Text")  # example to create a Text type collection

If you wish to check all parameters, after using the :code:`set_parameters`, you can use the property :code:`setup_parameters`.

.. code-block:: python

   >>> trainer.setup_parameters


:code:`fit`
-----------

TODO 

.. code-block:: python

   >>> from jai import Trainer
   ...
   >>> trainer = Trainer()
   >>> trainer.fit(data)

:code:`append`
--------------

TODO 

.. code-block:: python

   >>> from jai import Trainer
   ...
   >>> trainer = Trainer()
   >>> trainer.append(data)

:code:`report`
--------------

TODO 

.. code-block:: python

   >>> from jai import Trainer
   ...
   >>> trainer = Trainer()
   >>> trainer.append()

:code:`delete_ids`
------------------

TODO 

.. code-block:: python

   >>> from jai import Trainer
   ...
   >>> trainer = Trainer()
   >>> trainer.append()

:code:`delete_raw_data`
-----------------------

TODO 

.. code-block:: python

   >>> from jai import Trainer
   ...
   >>> trainer = Trainer()
   >>> trainer.append()

:code:`delete_database`
-----------------------

TODO 

.. code-block:: python

   >>> from jai import Trainer
   ...
   >>> trainer = Trainer()
   >>> trainer.append()

:code:`get_query`
-----------------

This method returns a new :ref:`Query <source/task/query:query module>` object with the same initial values as the current `Trainer`
object.

.. code-block:: python

   >>> from jai import Trainer
   ...
   >>> trainer = Trainer()
   >>> trainer.get_query()

Inherited from :code:`TaskBase`
===============================

:code:`name`
-----------------

This attribute contains the value of the database's name.

.. code-block:: python

   >>> from jai import Trainer
   ...
   >>> trainer = Trainer()
   >>> trainer.name

:code:`db_type`
-----------------

This attribute returns the type of the database.

.. code-block:: python

   >>> from jai import Trainer
   ...
   >>> trainer = Trainer()
   >>> trainer.db_type
   
:code:`is_valid`
-----------------

This method returns a boolean indicating if the database exists or not.

.. code-block:: python

   >>> from jai import Trainer
   ...
   >>> trainer = Trainer()
   >>> trainer.is_valid()

:code:`describe`
-----------------

This method returns the full configuration information of the database.

.. code-block:: python

   >>> from jai import Trainer
   ...
   >>> trainer = Trainer()
   >>> trainer.describe()

   