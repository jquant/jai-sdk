
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

Bellow, a simple example to instantiate the Explorer class:

.. code-block:: python

   >>> from jai import Explorer
   ...
   >>> explorer = Explorer()

:code:`user`
------------

You can check the user information with :code:`user` method.

.. note:: 
   If you wish to check whether **the authentication key is valid** or not, we recommend the usage of this method.

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

Provides a full description of the database's setup configuration.

.. code-block:: python

   >>> from jai import Explorer
   ...
   >>> explorer = Explorer()
   >>> explorer.describe(name)

:code:`rename`
--------------

Renames a database.

.. note:: 
   We recommend not changing database's names if it's already being used as a parent base for another databases.

.. code-block:: python

   >>> from jai import Explorer
   ...
   >>> explorer = Explorer()
   >>> explorer.rename(original_name, new_name)

:code:`transfer`
----------------

Transfers a collection from one environment to another.

.. note:: 
   We recommend not changing database's names if it's already being used as a parent base for another databases.
   If a database has parents, you'll need to transfer one by one.

.. code-block:: python

   >>> from jai import Explorer
   ...
   >>> explorer = Explorer()
   >>> explorer.transfer(original_name, to_environment)

:code:`import_database`
-----------------------

It imports a database from another user/environment.

.. note:: 
   We recommend not changing database's names if it's already being used as a parent base for another databases.
   If a database has parents, you'll need to transfer one by one.
   The environment must be `PUBLIC` type.

.. code-block:: python

   >>> from jai import Explorer
   ...
   >>> explorer = Explorer()
   >>> explorer.import_database(
   ...     database_name,
   ...     owner_id,
   ...     import_name
   ... )