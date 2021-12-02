.. _jai in 5 min:

==========================================================
JAI in 5 minutes - Flower classification with Iris dataset
==========================================================

************************
What are we going to do?
************************

In this quick demo, we will use JAI to:

* Train and deploy models into a secure and scalable production-ready environment.
* Classification - Given a list of flowers attributes, classify the flower within three possible classes.
* Model Inference - Predict the class of new flowers and check the results.

***********
Install JAI
***********

Start by installing JAI with :code:`pip`:

.. code:: bash

    pip install jai-sdk --user
      
**********
Import JAI
**********

In your python IDE of preference, you import JAI with:

.. code:: python

    >>> from jai import Jai

*****************
Generate Auth Key
*****************

JAI requires an Auth Key to organize and secure your collections, and it needs to be generated only once. You can easily generate your free-forever auth-key by running the command below:

.. code:: python

    >>> Jai.get_auth_key(email='email@mail.com', firstName='Jai', lastName='Z')
    201

.. note::
    Please note that your Auth Key will be sent to your e-mail, so please make sure to use a valid address and check your spam folder.

***************
Start JAI
***************

After receiving the authentication key, you are ready to instantiate JAI with your Auth Key and start to use your JAI account:

.. code:: python

    >>> AUTH_KEY= "xXxxxXxxxxXxxxxxXxxxXxXxxx"
    >>> j = Jai(AUTH_KEY) 

*******************
Dataset quick look
*******************

This dataset is frequently used on machine learning classification classes and a further explanation of it can be found in the `sklearn documentation <https://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html>`_. The dataset presents 150 rows equaly divided between the three flower classes, which are Setosa, Versicolour, and Virginica.        

Let's have a quick glance on come columns of this dataset below:  

.. code:: python

    >>> # Importing other necessary libs
    >>> import pandas as pd
    >>> from tabulate import tabulate
    >>> from sklearn.datasets import load_iris
    >>>
    >>> # Loading dataframe
    >>> df = pd.DataFrame(load_iris(as_frame=True).data)#return_X_y=True,
    >>> target = load_iris(as_frame=True).target
    >>> print(tabulate(df[['Time', 'V1', 'V2','V28', 'Amount','Class']].head(), headers='keys', tablefmt='rst'))
    ====  ===================  ==================  ===================  ==================
    ..    sepal length (cm)    sepal width (cm)    petal length (cm)    petal width (cm)
    ====  ===================  ==================  ===================  ==================
       0                  5.1                 3.5                  1.4                 0.2
       1                  4.9                 3                    1.4                 0.2
       2                  4.7                 3.2                  1.3                 0.2
       3                  4.6                 3.1                  1.5                 0.2
       4                  5                   3.6                  1.4                 0.2
    ====  ===================  ==================  ===================  ==================

*******************
Supervised Learning
*******************

Now we will train a Supervised Model to classify each flower example within the three classes using the attributes previously shown.
  
.. code:: python

    >>> from sklearn.model_selection import train_test_split
    >>>
    >>> # Splitting the dataset to demonstrate j.predict
    >>> X_train, X_test, y_train, y_test = train_test_split(
    >>>             df, target, test_size=0.3, random_state=42)
    >>> 
    >>> # Creating a training table with the target
    >>> train = pd.concat([X_train,y_train],axis=1)
    >>>
    >>> # For the supervised model we have to pass the dataframe with the label to JAI
    >>> train = pd.concat([X_train,y_train],axis=1)
    >>>
    >>> # Training the classification model
    >>> j.fit(
    >>>     # JAI collection name    
    >>>     name="iris_supervised", 
    >>>
    >>>     # Data to be processed - a Pandas DataFrame is expected
    >>>     data=train, 
    >>>
    >>>     # Collection type
    >>>     db_type='Supervised', 
    >>>
    >>>     # You can uncomment this line if you wish to test different parameters and maintain the same collection name
    >>>     #overwrite = True,
    >>>
    >>>     # Verbose 2 -> shows the loss graph at the end of training
    >>>     verbose=2,
    >>>
    >>>     # The split type as stratified guarantee that the same proportion of both classes are maintained for train, validation and test
    >>>     split = {'type':'stratified'},
    >>>
    >>>     # When we set task as *classification* we use CrossEntropy Loss
    >>>     label={"task": "classification",
    >>>           "label_name": "target"}
    >>> )

    Insert Data: 100%|████████████████████████████████| 1/1 [00:01<00:00,  1.07s/it]

    Recognized setup args:
    - db_type: Supervised
    - label: 
    * task      : classification
    * label_name: target
    - overwrite: True

    Training might finish early due to early stopping criteria.
    JAI is working:  45%|████████████████████▍                        |10/22 [00:16]
    [iris_supervised] Training:   0%|                       | 0/500 [00:00<?, ?it/s]
    [iris_supervised] Training:   3%|▍             | 15/500 [00:01<01:01,  7.87it/s]
    [iris_supervised] Training:   5%|▋             | 26/500 [00:04<01:26,  5.51it/s]
    [iris_supervised] Training:   7%|▉             | 33/500 [00:06<01:36,  4.82it/s]
    [iris_supervised] Training:   9%|█▎            | 47/500 [00:08<01:17,  5.83it/s]
    [iris_supervised] Training:  13%|█▊            | 64/500 [00:11<01:14,  5.87it/s]
    [iris_supervised] Training: 100%|█████████████| 500/500 [00:13<00:00, 66.82it/s]
    JAI is working: 100%|█████████████████████████████████████████████|22/22 [00:37]

    Setup Report:
    Metrics classification:
                precision    recall  f1-score   support

            0       1.00      1.00      1.00         7
            1       1.00      0.86      0.92         7
            2       0.88      1.00      0.93         7

        accuracy                           0.95        21
    macro avg       0.96      0.95      0.95        21
    weighted avg       0.96      0.95      0.95        21

    Best model at epoch: 69 val_loss: 0.07

For more information about the j.fit args you can access `this part <https://jai-sdk.readthedocs.io/en/stable/source/jai.html#setup-kwargs>`_ of our documentation.

********************
Model Inference
********************

Now that our Supervised Model is also JAI collection, we can perform predictions with it, applying the model to new examples very easily:

.. code:: python

    >>> # Every JAI collection can be queried using j.predict()
    >>> ans = j.predict(
    >>>     # collection to be queried
    >>>     name='iris_supervised',
    >>>     as_frame = True,
    >>>     # let's get the X_test we have separated before
    >>>     data=X_test
    >>> )

    Predict: 100%|████████████████████████████████████| 1/1 [00:04<00:00,  4.51s/it]
    Predict Processing: 100%|███████████████████| 45/45 [00:00<00:00, 257143.98it/s]

And now the :code:`ans` variable holds a dataframe with both predictions and true values:

.. code:: python

    >>> # Here it's possible to see how the answer will come
    >>> # **ATENTION**: Be careful when comparing the true and predicted values. The ids of the answers are ordered inside JAI
    >>> ans["y_true"] = y_test
    >>> print(tabulate(ans.head(), headers='keys', tablefmt='rst'))
    ====  =========  ========
    id    predict    y_true
    ====  =========  ========
       4          0         0
       9          0         0
      10          0         0
      11          0         0
      12          0         0
    ====  =========  ========

Manipulating the information received in **ans**, we can check the classification report of the prediction:

.. code:: python

    >>> # Checking the classification report
    >>> from sklearn import metrics
    >>> print(metrics.classification_report( ans["y_true"],ans["predict"],target_names=['0','1','2']))
                precision    recall  f1-score   support

            0       1.00      1.00      1.00        19
            1       1.00      1.00      1.00        13
            2       1.00      1.00      1.00        13

    accuracy                            1.00        45
    macro avg       1.00      1.00      1.00        45
    weighted avg    1.00      1.00      1.00        45

For more information about the j.fit args you can access `this part <https://jai-sdk.readthedocs.io/en/stable/source/jai.html#setup-kwargs>`_ of our documentation.

**********************
Always deployed (REST)
**********************

Everything in JAI is always instantly deployed and available through REST API, which makes most of the job of putting yuour model in production much easier!

.. code:: python

    >>> # Model Inference via REST API
    >>> 
    >>> # import requests libraries
    >>> import requests
    >>> 
    >>> AUTH_KEY= "xXxxxXxxxxXxxxxxXxxxXxXxxx"
    >>>
    >>> # set Authentication header
    >>> header={'Auth': AUTH_KEY}
    >>> 
    >>> # set collection name
    >>> db_name = 'iris_supervised' 
    >>> 
    >>> # model inference endpoint
    >>> url_predict = f"https://mycelia.azure-api.net/predict/{db_name}"
    >>> 
    >>> # json body
    >>> # note that we need to provide a column named 'id'
    >>> # also note that we drop the 'PRICE' column because it is not a feature
    >>> body = X_test.reset_index().rename(columns={'index':'id'}).head().to_dict(orient='records')
    >>> 
    >>> # make the request
    >>> ans = requests.put(url_predict, json=body, headers=header)
    >>> ans.json()
    [{'id': 18, 'predict': 0},
    {'id': 73, 'predict': 1},
    {'id': 76, 'predict': 1},
    {'id': 78, 'predict': 1},
    {'id': 118, 'predict': 2}]

For more discussions about this example, join our `slack community <https://join.slack.com/t/getjai/shared_invite/zt-sfkm3tpg-oJuvdziWgtaFEaIUUKWUV>`_!