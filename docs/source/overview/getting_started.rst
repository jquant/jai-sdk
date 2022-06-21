.. _getting_started:

===============
Getting Started
===============

Installation
------------

The latest version of JAI-SDK can be installed from pip as follows:

.. code:: bash

    pip install jai-sdk --user

Nowadays, JAI supports python 3.7+.

Getting your auth key
---------------------

JAI requires an auth key to organize and secure collections. 
You can quickly generate your free-forever auth-key by running the command below:

.. code:: python

    >>> from jai import get_auth_key
    >>> get_auth_key(email='email@mail.com', firstName='Jai', lastName='Z')

.. attention::

    Your auth key will be sent to your e-mail, so please make sure to use a valid address and check your spam folder.


How does it work?
-----------------

With JAI, you can train models in the cloud and run inference on your trained models. Besides, you can achieve all your models through a REST API endpoint. 

First, you can set your auth key into an environment variable or use a :file:`.env` file or :file:`.ini` file.
Please check the section :ref:`How to configure your auth key <source/overview/set_authentication:How to configure your auth key>` for more information.

Bellow an example of the content of the :file:`.env` file:

.. code-block:: text

    JAI_AUTH="xXxxxXXxXXxXXxXXxXXxXXxXXxxx"


In the below example, we'll show how to train a simple supervised model (regression) using the California housing dataset, run a prediction from this model, and call this prediction directly from the REST API.

.. code-block:: python

    >>> import pandas as pd
    >>> from jai import Jai
    >>> from sklearn.datasets import fetch_california_housing
    ... 
    >>> # Load dataset
    >>> data, labels = fetch_california_housing(as_frame=True, return_X_y=True)
    >>> model_data = pd.concat([data, labels], axis=1)
    ... 
    >>> # Instanciating JAI class
    >>> j = Jai()
    ... 
    >>> # Send data to JAI for feature extraction
    >>> j.fit(
    ...     name='california_supervised',   # JAI collection name 
    ...     data=model_data,    # Data to be processed
    ...     db_type='Supervised',   # Your training type ('Supervised', 'SelfSupervised' etc)
    ...     verbose=2,
    ...     hyperparams={
    ...         'learning_rate': 3e-4,
    ...         'pretraining_ratio': 0.8
    ...     },
    ...     label={
    ...         'task': 'regression',
    ...         'label_name': 'MedHouseVal'
    ...     },
    ...     overwrite=True)
    ... 
    >>> # Run prediction
    >>> j.predict(name='california_supervised', data=data)

In this example, you could train a supervised model with the California housing dataset and run a prediction with some data.

JAI supports many other training models, like self-supervised model training. 
Besides, it also can train on different data types, like text and images. 
You can find a complete list of the model types supported by JAI on :ref:`The Fit Method <the_fit_method>`.


What to do next?
----------------

Visit :ref:`Jai in 5 Minutes <jai_in_5_min>` to get a more complex and detailed example of how to use JAI correctly. 

Read about :ref:`The Fit Method <the_fit_method>` if you want a complete overview of what models JAI can train and what you can do to get your better model.