.. _set_authentication:

How to configure your auth key
==============================

There are several ways to configure your auth key.

.. contents:: :local: 

1. Set an Environment variable
------------------------------

The first way to configure your auth key is setting an environment variable on your system.
Simply set an new environment variable with the name :code:`JAI_AUTH`.

.. note:: 
    Every Jai class that requires the usage of the auth key has a :code:`env_var` parameter in case you need to use a different environment variable.

2. Using :code:`os.environ`
---------------------------

It's also possible to set the environment variable directly to :code:`os.environ` or use this to yield the same result:

.. code-block:: python

    >>> from jai import set_authentication
    ...
    >>> set_authentication("xXxxxXXxXXxXXxXXxXXxXXxXXxxx")


.. note:: 
    For more details on how to use :code:`os.environ` check its `documentation <os_environ>`_ .

.. warning:: 
    We recommend not writing your auth key on your scripts, specially if you share them. 
    Changes to Jai's backend are definitive. 
    Keep your auth key safe. |:lock:|

3. Set a config file
--------------------

Since setting an environment variable could be a little tricky, specially on shared environments.
We decided to use `python-decouple <decouple_github>`_ package to allow you to set the auth value using a :file:`.env` file or a :file:`.ini` file.

Bellow an example of the content of the :file:`settings.ini` file:

.. code-block:: text

    [settings]
    JAI_AUTH=xXxxxXXxXXxXXxXXxXXxXXxXXxxx

Bellow an example of the content of the :file:`.env` file:

.. code-block:: text

    JAI_AUTH="xXxxxXXxXXxXXxXXxXXxXXxXXxxx"


.. note:: 

    `Decouple <decouple_order>`_ always searches for Options in this order:

    1. Environment variables;
    2. Repository: :file:`.ini` or :file:`.env` file;

    This means that config files won't be considered when there's already a value set on environment variables


.. _decouple_github: https://github.com/henriquebastos/python-decouple
.. _decouple_order: https://github.com/henriquebastos/python-decouple#how-does-it-work
.. _os_environ: https://docs.python.org/3/library/os.html#os.environ