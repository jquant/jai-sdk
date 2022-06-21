
###############
Explorer Module
###############

.. note::
   If you haven't yet, please check the section 
   :ref:`How to configure your auth key <source/overview/set_authentication:How to configure your auth key>` 
   before running the code snippets in this page.

The Explorer module is built to manage and explore (oh really) your Jai environment. 

The methods of this module function to manage collections and get general information about your Jai environment.

For more information, see the full :ref:`Explorer class reference <source/reference/task_explorer:explorer class>`.

:code:`Explorer`
================

Bellow, a simple example to instanciate the Explorer class:

.. code-block:: python

   >>> from jai import Explorer
   ...
   >>> explorer = Explorer()

:code:`user`
------------

You can check the user information with :code:`user` method.

.. note:: 
   We recommend the usage of this method if you wish to check if the authentication key if valid.

.. code-block:: python

   >>> from jai import Explorer
   ...
   >>> explorer = Explorer()
   >>> explorer.user()

:code:`environments`
--------------------

This method returns the list of available environments.

Check the section :ref:`Working with environments <source/advanced/environments:working with environments>` for more information.

.. code-block:: python

   >>> from jai import Explorer
   ...
   >>> explorer = Explorer()
   >>> explorer.environments()

:code:`names`
-------------

You can use the property :code:`names` to get the names of all databases on the current environment.

.. code-block:: python

   >>> from jai import Explorer
   ...
   >>> explorer = Explorer()
   >>> explorer.names

:code:`info`
------------

You can use the property :code:`info` to get more info about all databases on the curent environment.

The information current version returns are the name, type, date of creation, databases parents, size and dimension of the vectors.

.. code-block:: python

   >>> from jai import Explorer
   ...
   >>> explorer = Explorer()
   >>> explorer.info()

It's possible to trim the information returned with the parameter :code:`get_size=False`. 
It will remove the size and dimension information from the response.

.. code-block:: python

   >>> from jai import Explorer
   ...
   >>> explorer = Explorer()
   >>> explorer.info(get_size=False)


:code:`describe`
----------------

TODO 

.. code-block:: python

   >>> from jai import Explorer
   ...
   >>> explorer = Explorer()
   >>> explorer.describe("db_name")

:code:`rename`
--------------

TODO 

.. code-block:: python

   >>> from jai import Explorer
   ...
   >>> explorer = Explorer()
   >>> explorer.rename()

:code:`transfer`
----------------

TODO 

.. code-block:: python

   >>> from jai import Explorer
   ...
   >>> explorer = Explorer()
   >>> explorer.transfer()

:code:`import_database`
-----------------------

TODO 

.. code-block:: python

   >>> from jai import Explorer
   ...
   >>> explorer = Explorer()
   >>> explorer.import_database()