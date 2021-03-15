###########
Quick Start
###########

This is a quick start guide to help users getting used to Jai 

************
Installation
************

You can install JAI-SDK via the Python Package Index (PyPI).

To install using pip:

.. code-block:: console

    $ pip install jai-sdk

*******************************
Getting your Authentication Key
*******************************

You need an Authorization Key to use the backend of our API.

To get an Trial key, please fill in the values with your information:

.. code-block:: python

	>>> from jai import Jai
	>>> r = Jai.get_auth_key(email=EMAIL, firstName=FIRSTNAME, lastName=LASTNAME)
	>>> r.status_code
	201

If the response code is 201, you should receive an email with your Auth Key.
