===========================
Credit Card Fraud Detection
===========================

************************
What are we going to do?
************************

In this quick demo, we will use JAI to:

* Train and deploy models into a secure and scalable production-ready environment.
* Classification - Given a list of credit card users and attributes, classify which clients would default.
* Model Inference - Predict which new users would default or not and check the results.

*********
Start JAI
*********

Firstly, you'll need to install the JAI package and generate your auth key, as explained in the 
:ref:`Getting Started <source/1-overview/getting_started:Getting Started>` section. 

With your authentication key, you'll need to configure your auth key.
Please check the section :ref:`How to configure your auth key <source/1-overview/set_authentication:How to configure your auth key>` for more information.

*******************
Dataset quick look
*******************

The credit card default dataset was brought from a `Kaggle competition <https://www.kaggle.com/mlg-ulb/creditcardfraud>`_, 
where you can download the whole dataset from. The dataset contains *284807 rows* and *31 columns* including 
the label in the :code:`Class` column. 

The main peculiarity of this dataset is that the label is **highly unbalanced**. In this case, there are 
*284315 non-default users* versus *492 defaults* (only 0.172% of all transactions) on the whole database.

Let's first load the dataset of interest:

.. code-block:: python

    >>> # Importing other necessary libraries
    >>> import pandas as pd
    >>> from sklearn import metrics
    >>> from tabulate import tabulate
    >>> from sklearn.metrics import roc_auc_score
    >>> from sklearn.model_selection import train_test_split
    ...
    >>> # Loading dataframes
    >>> DATASET_PATH = "creditcard.csv"
    >>> df = pd.read_csv(DATASET_PATH)

Let's have a glance at some columns of this dataset below:  

.. code-block:: python
    
    >>> print(tabulate(df[['Time', 'V1', 'V2','V28', 'Amount','Class']].head(), headers='keys', tablefmt='rst'))
    
    ====  ======  =========  ==========  ==========  ========  =======
      ..    Time         V1          V2         V28    Amount    Class
    ====  ======  =========  ==========  ==========  ========  =======
       0       0  -1.35981   -0.0727812  -0.0210531    149.62        0
       1       0   1.19186    0.266151    0.0147242      2.69        0
       2       1  -1.35835   -1.34016    -0.0597518    378.66        0
       3       1  -0.966272  -0.185226    0.0614576    123.5         0
       4       2  -1.15823    0.877737    0.215153      69.99        0
    ====  ======  =========  ==========  ==========  ========  =======


*******************
Supervised Learning
*******************

Now we will train a Supervised Model to classify if a client will be considered default or not using JAI! 
Here we will separate part of the dataset to show how the prediction is made using JAI. 
Since we only have data of two days, we don't have to worry about data leakage when splitting our dataset randomly.
  
.. code-block:: python

    >>> from jai import Jai
    >>> from sklearn.model_selection import train_test_split
    ...
    >>> # In this case, we will take part of our dataset to demonstrate the prediction further in this tutorial
    >>> # The j.fit already takes care of the train and validation split on its backend, so in a normal situation this is not necessary
    >>> X_train, X_prediction, y_train, y_prediction = train_test_split( df.drop(["Class"],axis=1), 
    ...                                                    df["Class"], test_size=0.3, random_state=42)
    ...
    >>> # For the supervised model we have to pass the dataframe with the label to JAI
    >>> train = pd.concat([X_train,y_train],axis=1)
    ...
    >>> # Training the classification model
    >>> j = Jai()
    >>> j.fit(
    ...     # JAI collection name    
    ...     name="cc_fraud_supervised", 
    ...     # Data to be processed - a Pandas DataFrame is expected
    ...     data=train, 
    ...     # Collection type
    ...     db_type='Supervised', 
    ...     # Verbose 2 -> shows the loss graph at the end of training
    ...     verbose=2,
    ...     # The split type as a stratified guarantee that the same proportion of both classes are maintained for train, validation and test
    ...     split = {'type':'stratified'},
    ...     # When we set the task as *metric_classification* we use Supervised Contrastive Loss, which tries to make examples of the same class closer and make those of different classes apart
    ...     label={
    ...         "task": "metric_classification",
    ...         "label_name": "Class"
    ...     }
    ...     # You can uncomment this line if you wish to test different parameters and maintain the same collection name
    ...     # overwrite = True
    ... )

    Setup Report:
    Metrics classification:
               precision    recall  f1-score   support
    
            0       1.00      1.00      1.00     39821
            1       0.77      0.80      0.79        51
     
    accuracy                            1.00     39872
    macro avg       0.89      0.90      0.89     39872
    weighted avg    1.00      1.00      1.00     39872
    
    Best model at epoch: 76 val_loss: 6.93

For more information about the :code:`j.fit` args you can access `this part <https://jai-sdk.readthedocs.io/en/stable/source/jai.html#setup-kwargs>`_ of our documentation.

***************
Model Inference
***************

Now that our Supervised Model is also JAI collection, we can perform predictions with it, applying the model to new examples very easily. Let's do it first without predict_proba:

.. code-block:: python

    >>> # Now we will make the predictions
    >>> # In this case, it will use 0.5 (which is default) as a threshold to return the predicted class
    >>> ans = j.predict(
    ...    
    ...     # Collection to be queried
    ...     name='cc_fraud_supervised',
    ...    
    ...     # This will make your answer return as a dataframe
    ...     as_frame=True,
    ...     
    ...     # Here you will pass a dataframe to predict which examples are default or not
    ...     data=X_test
    ... )

Now let's put y_test alongside the predicted classes. Be careful when doing this: JAI returns the answers with sorted indexes.

.. code-block:: python

    >>> # ATTENTION: JAI ALWAYS RETURNS THE ANSWERS ORDERED BY ID! Bringing y_test like this will avoid mismatchings
    >>> ans["y_true"] = y_test
    >>> print(tabulate(ans.head(), headers='keys', tablefmt='rst'))
    
    ====  =========  ========
      id    predict    y_true
    ====  =========  ========
       0          0         0
      16          0         0
      24          0         0
      26          0         0
      41          0         0
    ====  =========  ========

    >>> print(metrics.classification_report( ans["y_true"],ans["predict"],target_names=['0','1']))
    
                  precision    recall  f1-score   support

               0       1.00      1.00      1.00     85307
               1       0.77      0.79      0.78       136

        accuracy                           1.00     85443
       macro avg       0.89      0.90      0.89     85443
    weighted avg       1.00      1.00      1.00     85443
    
If you wish to define your threshold or use the predicted probabilities to rank the answers, we can make the predictions as follows:

.. code-block:: python

    >>> ans = j.predict(
    ...     
    ...     # Collection to be queried
    ...     name='cc_fraud_supervised',
    ...     
    ...     # This will bring the probabilities predicted
    ...     predict_proba = True,
    ...     
    ...     # This will make your answer return as a dataframe
    ...     as_frame=True,
    ...     
    ...     # Here you will pass a dataframe to predict which examples are default or not
    ...     data=X_test
    ... )
    ...
    >>> # ATTENTION: JAI ALWAYS RETURNS THE ANSWERS ORDERED BY ID! Bringing y_test like this will avoid mismatchings
    >>> ans["y_true"] = y_test
    >>> print(tabulate(ans.head(), headers='keys', tablefmt='rst'))
    
    ====  ========  ==========  =========  ================  ========
      id         0           1    predict    probability(%)    y_true
    ====  ========  ==========  =========  ================  ========
       0  0.991032  0.00896752          0             99.1          0
      16  0.986639  0.0133607           0             98.66         0
      24  0.983173  0.0168269           0             98.32         0
      26  0.985789  0.014211            0             98.58         0
      41  0.979446  0.020554            0             97.94         0
    ====  ========  ==========  =========  ================  ========
    
    >>> # Calculating AUC Score using the predictions of examples being 1
    >>> roc_auc_score(ans["y_true"], ans["1"])
    
    0.9621445967815895
     
******************************
Making inference from REST API
******************************

Everything in JAI is always instantly deployed and available through REST API, which makes most 
of the job of putting your model in production much easier!

.. code-block:: python
    
    >>> # Importing requests library
    >>> import requests
    ...
    >>> AUTH_KEY = "insert_your_auth_key_here"
    ...
    >>> # Set Authentication header
    >>> header={'Auth': AUTH_KEY}
    ...
    >>> # Set collection name
    >>> db_name = 'cc_fraud_supervised' 
    ...
    >>> # Model inference endpoint
    >>> url_predict = f"https://mycelia.azure-api.net/predict/{db_name}"
    ...
    >>> # json body
    >>> # Note that we need to provide a column named 'id'
    >>> # Also note that we drop the 'PRICE' column because it is not a feature
    >>> body = X_test.reset_index().rename(columns={'index':'id'}).head().to_dict(orient='records')
    ...
    >>> # Make the request
    >>> ans = requests.put(url_predict, json=body, headers=header)
    >>> ans.json()

    [{'id': 29474, 'predict': 0},
    {'id': 43428, 'predict': 1},
    {'id': 49906, 'predict': 0},
    {'id': 276481, 'predict': 0},
    {'id': 278846, 'predict': 0}]

For more discussions about this example, 
join our `slack community <https://join.slack.com/t/getjai/shared_invite/zt-sfkm3tpg-oJuvdziWgtaFEaIUUKWUV>`_!