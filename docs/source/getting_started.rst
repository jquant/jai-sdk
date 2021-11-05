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

JAI requires an Auth Key to organize and secure collections. You can quickly generate your free-forever auth-key by running the command below:

.. code:: python

    from jai import Jai

    j = Jai.get_auth_key(email='email@mail.com', firstName='Jai', lastName='Z')

.. attention::

    Your *auth key* will be sent to your e-mail, so please make sure to use a valid address and check your spam folder.


How does it work?
-----------------

With JAI, you can train models in the cloud and run inference on your trained models. Besides, you can achieve all your models through a REST API endpoint. 
In the below example, we'll show how to train a simple supervised model (regression) using the California housing dataset, run a prediction from this model, and call this prediction directly from the REST API.


.. code:: python

    from jai import Jai
    from sklearn.datasets import fetch_california_housing
    
    # Your auth key 
    AUTH_KEY = 'xXxxxXXxXXxXXxXXxXXxXXxXXxxx'
    
    # Authenticating in JAI
    j = Jai(AUTH_KEY)

    # Load dataset
    data, labels = fetch_california_housing(as_frame=True, return_X_y=True)
    model_data = pd.concat([data, labels], axis=1)

    # Send data to JAI for feature extraction
    j.fit(
        name='california_supervised',   # JAI collection name
        data=model_data,  # data to be processed - a Pandas DataFrame is expected
        db_type='Supervised',   # The type of your training ('Supervised', 'SelfSupervised' etc)
        verbose=2,  # verbose 2: shows the loss graph at the end of training
        hyperparams={
            'learning_rate': 3e-4,
            'pretraining_ratio': 0.8
        },
        label={
            'task': 'regression',
            'label_name': 'MedHouseVal'
        },
        overwrite=True)

    # Run prediction
    j.predict(name='stress_california_sup', data=data)

In this example, you could train a supervised model with the California housing dataset and run a prediction with some data. JAI supports a bunch of other training models, like a self-supervised model. You can find a complete list of the model types supported by JAI on *[link] fit models* 

