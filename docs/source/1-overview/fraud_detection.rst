===========================
Credit Card Fraud Detection
===========================

************************
What are we going to do?
************************

In this quick demo, we will use JAI to:

* Train and deploy models into a secure and scalable production-ready environment.
* Classification - Given a list of credit card users and attributes, classify which clients would default.
* Model Inference - Predict which new users would dfault or not and check the results.

*********
Start JAI
*********

Firstly, you'll need to install the JAI package and generate your auth key, as explained in the 
:ref:`Getting Started <source/1-overview/getting_started:Getting Started>` section. 

With your authentication key, start authenticating in your JAI account:

.. code-block:: python

    AUTH_KEY = "xXxxxXxxxxXxxxxxXxxxXxXxxx"
    j = Jai(AUTH_KEY) 


*******************
Dataset quick look
*******************

The credit card default dataset was brought from a `Kaggle competition <https://www.kaggle.com/mlg-ulb/creditcardfraud>`_, 
where you can download the whole dataset from. The dataset contains *284807 rows* and *31 columns* including 
the label in the :code:`Class` column. 

The main peculiarity of this dataset is that the label is **highly umbalanced**. In this case, there are 
*284315 non-defaultant users* versus *492 defaults* (only 0.172% of all transactions) on the whole database.

Let's first load the dataset of interest:

.. code-block:: python

    # Importing other necessary libraries
    import pandas as pd
    from sklearn import metrics
    from tabulate import tabulate
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split
    
    # Loading dataframes
    DATASET_PATH = "creditcard.csv"
    df = pd.read_csv(DATASET_PATH)

Let's have a quick glance on come columns of this dataset below:  

.. code-block:: python
    
    print(tabulate(df[['Time', 'V1', 'V2','V28', 'Amount','Class']].head(), headers='keys', tablefmt='rst'))
    
    # Output:

    # ====  ======  =========  ==========  ==========  ========  =======
    # ..    Time         V1          V2         V28    Amount    Class
    # ====  ======  =========  ==========  ==========  ========  =======
    # 0       0  -1.35981   -0.0727812  -0.0210531    149.62        0
    # 1       0   1.19186    0.266151    0.0147242      2.69        0
    # 2       1  -1.35835   -1.34016    -0.0597518    378.66        0
    # 3       1  -0.966272  -0.185226    0.0614576    123.5         0
    # 4       2  -1.15823    0.877737    0.215153      69.99        0
    # ====  ======  =========  ==========  ==========  ========  =======


*******************
Supervised Learning
*******************

Now we will train a Supervised Model to classify if a client will be considered default or not using JAI! 
Here we will separate part of the dataset to show how the prediction is made using JAI. 
Since we only have data of two days, we don't have to worry about data leakage when splitting our dataset randomly.
  
.. code-block:: python

    from sklearn.model_selection import train_test_split
    
    # In this case, we will take part of our dataset to demonstrate the prediction 
    # further in this tutorial.
    # The j.fit already takes care of the train and validation split on its backend, 
    # so in a normal situation this is not necessary.
    X_train, X_prediction, y_train, y_prediction = train_test_split( df.drop(["Class"],axis=1), 
                                                       df["Class"], test_size=0.3, random_state=42)
    
    # For the supervised model we have to pass the dataframe with the label to JAI
    train = pd.concat([X_train,y_train],axis=1)
    
    # Training the classification model
    j.fit(
        # JAI collection name    
        name="cc_fraud_supervised", 
        # Data to be processed - a Pandas DataFrame is expected
        data=train, 
        # Collection type
        db_type='Supervised', 
        # Verbose 2 -> shows the loss graph at the end of training
        verbose=2,
        # The split type as stratified guarantee that the same proportion of both 
        # classes are maintained for train, validation and test
        split = {'type':'stratified'},
        # When we set task as *metric_classification* we use Supervised Contrastive 
        # Loss, which tries to make examples of the same class closer and make those 
        # of different classes apart.
        label={
            "task": "metric_classification",
            "label_name": "Class"
        }
        # You can uncomment this line if you wish to test different parameters and 
        # maintain the same collection name
        # overwrite = True
    )

    # Output:

    # Setup Report:
    # Metrics classification:
    #             precision    recall  f1-score   support
    # 
    #         0       1.00      1.00      1.00     39821
    #         1       0.77      0.80      0.79        51
    # 
    # accuracy                            1.00     39872
    # macro avg       0.89      0.90      0.89     39872
    # weighted avg    1.00      1.00      1.00     39872
    # 
    # Best model at epoch: 76 val_loss: 6.93

For more information about the :code:`j.fit` args you can access `this part <https://jai-sdk.readthedocs.io/en/stable/source/jai.html#setup-kwargs>`_ of our documentation.

***************
Model Inference
***************

Now that our Supervised Model is also JAI collection, we can perform predictions with it, applying the model to new examples very easily:

.. code-block:: python

    # every JAI collection can be queried using j.predict()
    ans = j.predict(
        # collection to be queried
        name='cc_fraud_supervised',
        predict_proba = True,
        # let's get the X_test we have separated before
        data=X_test
    )


And now the :code:`ans` variable holds a list of predictions:

.. code-block:: python

    # Here it's possible to see how the answer will come
    print(ans)

    # Output:
    # [{'id': 0, 'predict': {'0': 0.9910324814696065, '1': 0.008967518530393502}},
    #     {'id': 16, 'predict': {'0': 0.9866393373524565, '1': 0.013360662647543594}},
    #     {'id': 24, 'predict': {'0': 0.9831731282157427, '1': 0.01682687178425728}},
    #     {'id': 26, 'predict': {'0': 0.9857890272232137, '1': 0.01421097277678632}},
    #     {'id': 41, 'predict': {'0': 0.9794459983427174, '1': 0.020554001657282574}},
    #     {'id': 87, 'predict': {'0': 0.9829296150692808, '1': 0.017070384930719124}},
    #     {'id': 88, 'predict': {'0': 0.9830230947251252, '1': 0.016976905274874853}}]

Manipulating the information received in :code:`ans`, we can check the :code:`roc_auc_score` of the model:

.. code-block:: python

    # Here we are taking the probabilities of the answer of being one
    ans = pd.DataFrame([(x["id"],x["predict"]["1"]) for x in ans],columns=["index","y_pred"]).set_index("index")
    
    # **ATENTION**: Be careful when comparing the true and predicted values. 
    # The ids of the answers are ordered inside JAI
    ans["y_true"] = y_test
    
    # Let's print the top 5 of our predictions. 
    print(tabulate(ans[['y_pred', 'y_true']].head(), headers='keys', tablefmt='rst'))
    
    # Output:
    # 
    # =======  ==========  ========
    #   index      y_pred    y_true
    # =======  ==========  ========
    #       0  0.00896752         0
    #      16  0.0133607          0
    #      24  0.0168269          0
    #      26  0.014211           0
    #      41  0.020554           0
    # =======  ==========  ========


    from sklearn.metrics import roc_auc_score
    roc_auc_score(ans["y_true"], ans["y_pred"])
    
    # Output:
    # 0.9621445967815895

******************************
Making inference from REST API
******************************

Everything in JAI is always instantly deployed and available through REST API, which makes most 
of the job of putting your model in production much easier!

.. code-block:: python
    
    # import requests libraries
    import requests
    
    AUTH_KEY = "xXxxxXxxxxXxxxxxXxxxXxXxxx"

    # set Authentication header
    header={'Auth': AUTH_KEY}
    
    # set collection name
    db_name = 'cc_fraud_supervised' 
    
    # model inference endpoint
    url_predict = f"https://mycelia.azure-api.net/predict/{db_name}"
    
    # json body
    # note that we need to provide a column named 'id'
    # also note that we drop the 'PRICE' column because it is not a feature
    body = X_test.reset_index().rename(columns={'index':'id'}).head().to_dict(orient='records')
    
    # make the request
    ans = requests.put(url_predict, json=body, headers=header)
    ans.json()

    # Output
    # [{'id': 29474, 'predict': 0},
    # {'id': 43428, 'predict': 1},
    # {'id': 49906, 'predict': 0},
    # {'id': 276481, 'predict': 0},
    # {'id': 278846, 'predict': 0}]

For more discussions about this example, 
join our `slack community <https://join.slack.com/t/getjai/shared_invite/zt-sfkm3tpg-oJuvdziWgtaFEaIUUKWUV>`_!