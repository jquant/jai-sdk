
###############
Explorer Module
###############

.. note::
   If you haven't yet, please check the section :ref:`How to configure your auth key <source/1-overview/set_authentication:How to configure your auth key>` 
   for more information.

The Explorer module is built to manage and explore (oh really) your Jai environment. 
The methods chosen for this module are to manage collections and get general information about your Jai environment.

For more information, see the full :ref:`Explorer class reference <source/reference/explorer:explorer class>`.

:code:`Explorer`
================


.. code-block:: python

   >>> from jai import Explorer
   ...
   >>> explorer = Explorer()


:code:`names`
-------------

.. code-block:: python

   >>> from jai import Explorer
   ...
   >>> explorer = Explorer()
   >>> explorer.names

:code:`info`
------------

.. code-block:: python

   >>> from jai import Explorer
   ...
   >>> explorer = Explorer()
   >>> explorer.info()


.. code-block:: python

   >>> from jai import Explorer
   ...
   >>> explorer = Explorer()
   >>> explorer.info(get_size=False)

:code:`user`
------------
.. code-block:: python

   >>> from jai import Explorer
   ...
   >>> explorer = Explorer()
   >>> explorer.user()

:code:`environments`
--------------------
.. code-block:: python

   >>> from jai import Explorer
   ...
   >>> explorer = Explorer()
   >>> explorer.environments()

:code:`describe`
----------------

.. code-block:: python

   >>> from jai import Explorer
   ...
   >>> explorer = Explorer()
   >>> explorer.describe("db_name")

:code:`rename`
--------------
.. code-block:: python

   >>> from jai import Explorer
   ...
   >>> explorer = Explorer()
   >>> explorer.rename()

:code:`transfer`
----------------

.. code-block:: python

   >>> from jai import Explorer
   ...
   >>> explorer = Explorer()
   >>> explorer.transfer()

:code:`import_database`
-----------------------

.. code-block:: python

   >>> from jai import Explorer
   ...
   >>> explorer = Explorer()
   >>> explorer.import_database()