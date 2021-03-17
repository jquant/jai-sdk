###########
Quick Start
###########

This is a quick start guide to help users improve their data with Jai. 

************
Installation
************

You can install JAI-SDK via the Python Package Index (PyPI). To install using pip:

.. code-block:: console

    $ pip install jai-sdk

*******************************
Getting your Authentication Key
*******************************

You need an Authorization Key to use the API backend of Jai. To get a Trial key, please fill in the values with your information:

.. code-block:: python

	>>> from jai import Jai
	>>> r = Jai.get_auth_key(email=EMAIL, firstName=FIRSTNAME, lastName=LASTNAME, company=COMPANY)
	>>> r.status_code
	201

If the response code is 201, you should receive an email with your Auth Key.

*********
Hello Jai!
*********

Now that you have your Auth Key, you can start using Jai:

.. code-block:: python

	>>> from jai import Jai
	>>> AUTH_KEY = "your_auth_key"
	>>> j = Jai(AUTH_KEY)


