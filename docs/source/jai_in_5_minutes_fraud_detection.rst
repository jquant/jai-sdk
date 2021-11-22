.. _jai in 5 min:

===============================================
JAI in 5 minutes - Credit Card Fraud Detection
===============================================

************************
What are we going to do?
************************

In this quick demo, we will use JAI to:

* Train and deploy models into a secure and scalable production-ready environment.
* Metric Classification - Given a list of credit card users and attributes, classify which clients would default.
* Model Inference - Predict which new users would dfault or not and check the results.

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
    >>> j = Jai(AUTH_KEY) #Insert your AUTH_KEre

*******************
Dataset quick look
*******************
Let's first load the dataset of interest:

.. code:: python
    
    >>> DATASET_PATH = "creditcard.csv"
    >>> df = pd.read_csv(DATASET_PATH)

This dataset was brought from a [Kaggle competition](https://www.kaggle.com/mlg-ulb/creditcardfraud), where you can download the whole dataset from. The dataset contains 284807 rows and 31 columns including the label in the Class column. The main peculiarity of this dataset is that the label is **highly umbalanced**. In this case, there are 284315 non-defaultant users versus 492 defaults (only 0.172% of all transactions) on the whole database.    

In this case, since we only have data of two days, we don't have to worry about data leakage. Let's have a quick glance on come columns of this dataset below:  

.. code:: python
    
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

Now we will train a Supervised Model to classify if a client will be considered defaultant or not using JAI! Here we will separate part of the 
  
.. code:: python

    >>> from sklearn.model_selection import train_test_split
    >>>
    >>> # In this case, we will take part of our dataset to demonstrate the prediction further in this tutorial 
    >>> # The j.fit already takes care of the train and validation split on its backend, so in a normal situation this is not necessary
    >>> X_train, X_prediction, y_train, y_prediction = train_test_split( df.drop(["Class"],axis=1), 
    >>>                                                    df["Class"], test_size=0.3, random_state=42)
    >>>
    >>> # For the supervised model we have to pass the dataframe with the label to JAI
    >>> train = pd.concat([X_train,y_train],axis=1)
    >>>
    >>> # Training the classification model
    >>> j.fit(
    >>>     # JAI collection name    
    >>>     name="cc_fraud_supervised", 
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
    >>>     split = {'type':'stratified'},
    >>>
    >>>     label={"task": "metric_classification",
    >>>           "label_name": "Class"}
    >>> )

    Insert Data: 100%|██████████████████████████████| 13/13 [00:56<00:00,  4.38s/it]

    Recognized setup args:
    - db_type: Supervised
    - label: 
    * label_name: Class
    * task      : metric_classification
    - overwrite: True

    Training might finish early due to early stopping criteria.
    JAI is working:  45%|████████████████████▍                        |10/22 [10:49]
    [cc_fraud_supervised] Training:   0%|                   | 0/500 [00:00<?, ?it/s]
    [cc_fraud_supervised] Training:   1%|           | 4/500 [00:02<05:30,  1.50it/s]
    [cc_fraud_supervised] Training:   1%|▏          | 6/500 [00:04<06:00,  1.37it/s]
    [cc_fraud_supervised] Training:   2%|▏          | 8/500 [00:05<06:09,  1.33it/s]
    [cc_fraud_supervised] Training:   2%|▏         | 10/500 [00:07<06:21,  1.28it/s]
    [cc_fraud_supervised] Training:   3%|▎         | 14/500 [00:09<05:15,  1.54it/s]
    [cc_fraud_supervised] Training:   3%|▎         | 16/500 [00:11<05:36,  1.44it/s]
    [cc_fraud_supervised] Training:   4%|▎         | 18/500 [00:13<06:05,  1.32it/s]
    [cc_fraud_supervised] Training:   4%|▍         | 20/500 [00:14<06:11,  1.29it/s]
    [cc_fraud_supervised] Training:   5%|▍         | 24/500 [00:16<05:15,  1.51it/s]
    [cc_fraud_supervised] Training:   5%|▌         | 26/500 [00:18<05:32,  1.43it/s]
    [cc_fraud_supervised] Training:   6%|▌         | 28/500 [00:20<05:45,  1.37it/s]
    [cc_fraud_supervised] Training:   6%|▌         | 30/500 [00:21<05:52,  1.33it/s]
    [cc_fraud_supervised] Training:   7%|▋         | 34/500 [00:23<04:47,  1.62it/s]
    [cc_fraud_supervised] Training:   7%|▋         | 36/500 [00:25<05:07,  1.51it/s]
    [cc_fraud_supervised] Training:   8%|▊         | 38/500 [00:26<05:27,  1.41it/s]
    [cc_fraud_supervised] Training:   8%|▊         | 40/500 [00:28<05:43,  1.34it/s]
    [cc_fraud_supervised] Training:   8%|▊         | 42/500 [00:30<05:49,  1.31it/s]
    [cc_fraud_supervised] Training:   9%|▉         | 44/500 [00:31<05:50,  1.30it/s]
    [cc_fraud_supervised] Training:   9%|▉         | 46/500 [00:33<05:54,  1.28it/s]
    [cc_fraud_supervised] Training:  10%|█         | 50/500 [00:35<04:57,  1.51it/s]
    [cc_fraud_supervised] Training:  10%|█         | 51/500 [00:36<05:53,  1.27it/s]
    [cc_fraud_supervised] Training:  11%|█         | 55/500 [00:38<04:35,  1.62it/s]
    [cc_fraud_supervised] Training:  11%|█▏        | 57/500 [00:40<04:56,  1.49it/s]
    [cc_fraud_supervised] Training:  12%|█▏        | 61/500 [00:42<04:19,  1.69it/s]
    [cc_fraud_supervised] Training:  13%|█▎        | 63/500 [00:43<04:39,  1.57it/s]
    [cc_fraud_supervised] Training:  13%|█▎        | 67/500 [00:45<04:20,  1.66it/s]
    [cc_fraud_supervised] Training:  14%|█▍        | 70/500 [00:47<04:11,  1.71it/s]
    [cc_fraud_supervised] Training:  14%|█▍        | 72/500 [00:49<04:33,  1.57it/s]
    [cc_fraud_supervised] Training:  15%|█▌        | 76/500 [00:50<03:54,  1.81it/s]
    [cc_fraud_supervised] Training:  16%|█▌        | 78/500 [00:52<04:17,  1.64it/s]
    [cc_fraud_supervised] Training:  16%|█▌        | 81/500 [00:54<04:05,  1.70it/s]
    [cc_fraud_supervised] Training:  17%|█▋        | 85/500 [00:55<03:34,  1.93it/s]
    [cc_fraud_supervised] Training: 100%|█████████| 500/500 [00:57<00:00, 77.16it/s]
    JAI is working: 100%|█████████████████████████████████████████████|22/22 [13:45]

    Setup Report:
    Metrics classification:
                precision    recall  f1-score   support

            0       1.00      1.00      1.00     39821
            1       0.77      0.80      0.79        51

    accuracy                            1.00     39872
    macro avg       0.89      0.90      0.89     39872
    weighted avg       1.00      1.00      1.00     39872

    Best model at epoch: 76 val_loss: 6.93



********************
Model Inference
********************

Now that our Supervised Model is also JAI collection, we can perform predictions with it, applying the model to new examples very easily:

.. code:: python

    >>> # every JAI collection can be queried using j.predict()
    >>> ans = j.predict(
    >>>     # collection to be queried
    >>>     name='cc_fraud_supervised',
    >>>     predict_proba = True,
    >>>     # let's get the X_test we have separated before
    >>>     data=X_test
    >>> )

    Predict: 100%|████████████████████████████████████| 6/6 [02:13<00:00, 22.26s/it]

And now the :code:`ans` variable holds a list of predictions:

.. code:: python

    >>> # Here it's possible to see how the answer will come
    >>> ans
    [{'id': 0, 'predict': {'0': 0.9910324814696065, '1': 0.008967518530393502}},
        {'id': 16, 'predict': {'0': 0.9866393373524565, '1': 0.013360662647543594}},
        {'id': 24, 'predict': {'0': 0.9831731282157427, '1': 0.01682687178425728}},
        {'id': 26, 'predict': {'0': 0.9857890272232137, '1': 0.01421097277678632}},
        {'id': 41, 'predict': {'0': 0.9794459983427174, '1': 0.020554001657282574}},
        {'id': 87, 'predict': {'0': 0.9829296150692808, '1': 0.017070384930719124}},
        {'id': 88, 'predict': {'0': 0.9830230947251252, '1': 0.016976905274874853}}]

Manipulating the information received in **ans**, we can check the roc_auc_score of the model:

.. code:: python

    >>> # Here we are taking the probabilities of the answer of being one
    >>> ans = pd.DataFrame([(x["id"],x["predict"]["1"]) for x in ans],columns=["index","y_pred"]).set_index("index")
    >>>
    >>> # **ATENTION**: Be careful when comparing the true and predicted values. The ids of the answers are ordered inside JAI
    >>> ans["y_true"] = y_test
    >>>
    >>> # Let's print the top 5 of our predictions. 
    >>> print(tabulate(ans[['y_pred', 'y_true']].head(), headers='keys', tablefmt='rst'))
    =======  ==========  ========
      index      y_pred    y_true
    =======  ==========  ========
          0  0.00896752         0
         16  0.0133607          0
         24  0.0168269          0
         26  0.014211           0
         41  0.020554           0
    =======  ==========  ========


    >>> from sklearn.metrics import roc_auc_score
    >>> roc_auc_score(ans["y_true"], ans["y_pred"])
    0.9621445967815895

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
    >>> db_name = 'cc_fraud_supervised' 
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
    [{'id': 29474, 'predict': 0},
    {'id': 43428, 'predict': 1},
    {'id': 49906, 'predict': 0},
    {'id': 276481, 'predict': 0},
    {'id': 278846, 'predict': 0}]


