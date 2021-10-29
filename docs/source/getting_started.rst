.. _getting-started:

===============
Getting Started
===============

Installing JAI-SDK
------------------
The latest version of JAI-SDK can be installed from pip as follows:

.. code:: bash

    pip install jai-sdk --user

Getting an auth key
-------------------

JAI requires an Auth Key to organize and secure collections. You can easily generate your free-forever auth-key by running the command below:

.. code::

    from jai import Jai

    j = Jai.get_auth_key(email='email@mail.com', firstName='Jai', lastName='Z')

.. attention::

    Your Auth Key will be sent to your e-mail, so please make sure to use a valid address and check your spam folder.