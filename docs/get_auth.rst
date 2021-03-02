###############################
Getting your Authentication Key
###############################


First, you'll need and Authorization key to use the backend API.

To get an Trial version API using the sdk, fill the values with your information:

.. code-block:: python

	>>> from jai import Jai
	>>> r = Jai.get_auth_key(email=EMAIL, firstName=FIRSTNAME, lastName=LASTNAME)


If the response code is 201, then you should receive an email with your Auth Key.
