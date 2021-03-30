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

You need an Authentication Key to use the API backend of Jai. To get a Free trial key, please fill in the values with your information:

.. code-block:: python

	>>> from jai import Jai
	>>> r = Jai.get_auth_key(email=EMAIL, firstName=FIRSTNAME, lastName=LASTNAME, company=COMPANY)
	>>> r.status_code
	201

If the response code is 201, you should receive an email with your Auth Key.

The Free Trial version is limited to 20 requests/minute and 1000 requests/week. If you want to try our 30 day trial with unlimited request or know more about our paid plans, please get in touch with us on support@getjai.com.

*********
Hello Jai!
*********

Now that you have your Auth Key, you can start using Jai:

.. code-block:: python

	>>> from jai import Jai
	>>> AUTH_KEY = "your_auth_key"
	>>> j = Jai(AUTH_KEY)


