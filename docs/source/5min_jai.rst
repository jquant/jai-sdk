#################################
JAI in 5 minutes - Boston Housing
#################################

***************
Install JAI
***************

Install JAI with pip

.. code:: bash

    pip install jai-sdk --user
      
*****************
Generate Auth Key
*****************

Import JAI and Generate your Community Auth Key (free forever)

.. code:: python

    >>> from jai import Jai
    >>> Jai.get_auth_key(email='email@mail.com', firstName='Jai', lastName='Z')
    201

Please note that yout Auth Key will be sent to your e-mail, so please make sure to use a valid address and check your spam folder.

***************
Start Jai
***************

* Use your Auth Key to instantiate JAI:

.. code:: python

    >>> j = Jai('AUTH KEY')

************************
Self-Supervised Learning
************************

* Now let's use JAI to transform the Boston Housing dataset into vectors using SelfSupervised Learning

.. code:: python

    from sklearn.datasets import load_boston
      
    # load dataset
    boston = load_boston()

    # note that we are not loading the target column "PRICE"
    data = pd.DataFrame(boston.data, columns=boston.feature_names)
    
    # send data to JAI for feature extraction
    j.setup(
        # JAI collection name
        name='boston',

        # data to be processed - a Pandas DataFrame is expected
        data=data,

        # collection type
        db_type='SelfSupervised',

        # verbose 2 -> shows the loss graph at the end of training
        verbose=2,

        # let's set some hyperparams!
        hyperparams={
        'learning_rate': 3e-4,
        'pretraining_ratio':0.8
        }
    )

Output:

.. code:: bash

      Insert Data: 100%|██████████| 1/1 [00:01<00:00,  1.39s/it]
      Recognized setup args:
      hyperparams: {'learning_rate': 0.0003, 'pretraining_ratio': 0.8}
      JAI is working:  38%|███▊      |6/16
      [boston] Training:   0%|          | 0/500 [00:00<?, ?it/s]ATraining might not take 500 steps due to early stopping criteria.
      
      [boston] Training:   1%|          | 6/500 [00:02<02:45,  2.99it/s]A
      [boston] Training:   2%|▏         | 11/500 [00:04<03:22,  2.41it/s]A
      [boston] Training:   3%|▎         | 14/500 [00:06<04:08,  1.96it/s]A
      [boston] Training:   3%|▎         | 17/500 [00:08<04:20,  1.85it/s]A
      [boston] Training:   5%|▍         | 23/500 [00:11<03:57,  2.01it/s]A
      [boston] Training:   6%|▌         | 28/500 [00:13<03:42,  2.12it/s]A
      [boston] Training:   6%|▋         | 32/500 [00:15<03:48,  2.05it/s]A
      [boston] Training: 100%|██████████| 500/500 [00:17<00:00, 74.20it/s]A
      JAI is working:  56%|█████▋    |9/16
      Done training.
      JAI is working: 100%|██████████|16/16


*****************
Similarity Search
*****************

* Now that our Boston Housing data is in a JAI collection, we can perform Similarity Search, i.e. find similar houses, very easily:

.. code:: python

    # every JAI collection can be queried using j.similar()
    ans = j.similar(
        # collection to be queried
        name='boston',
        # let's find houses that are similar to ids 1 and 10
        data=[1, 10]
    )

Output:

.. code:: bash

    Similar: 100%|██████████| 1/1 [00:01<00:00,  1.36s/it]

And now the 'ans' variable holds a JSON:

.. code:: bash

    [{'query_id': 1,
    'results': [{'id': 1, 'distance': 0.0},
    {'id': 96, 'distance': 0.012930447235703468},
    {'id': 235, 'distance': 0.02305753342807293},
    {'id': 176, 'distance': 0.02424568682909012},
    {'id': 90, 'distance': 0.025710342451930046}]},
    
    {'query_id': 10,
    'results': [{'id': 10, 'distance': 0.0},
    {'id': 7, 'distance': 0.0065054153092205524},
    {'id': 9, 'distance': 0.020906779915094376},
    {'id': 11, 'distance': 0.04773647338151932},
    {'id': 6, 'distance': 0.09080290794372559}]}]

And by indexing it back to the original dataframe id's, we have:

.. code:: python

    >>> # id 1
    >>> # List of top 5 similar houses (house 1 itself + 4)
    >>> data.loc[pd.DataFrame(ans[0]['results']).id]
    ====  =======  ====  =======  ======  =====  =====  =====  ======  =====  =====  =========  ======  =======
      ..     CRIM    ZN    INDUS    CHAS    NOX     RM    AGE     DIS    RAD    TAX    PTRATIO       B    LSTAT
    ====  =======  ====  =======  ======  =====  =====  =====  ======  =====  =====  =========  ======  =======
       1  0.02731     0     7.07       0  0.469  6.421   78.9  4.9671      2    242       17.8  396.9      9.14
      96  0.11504     0     2.89       0  0.445  6.163   69.6  3.4952      2    276       18    391.83    11.34
     235  0.33045     0     6.2        0  0.507  6.086   61.5  3.6519      8    307       17.4  376.75    10.88
     176  0.07022     0     4.05       0  0.51   6.02    47.2  3.5549      5    296       16.6  393.23    10.11
      90  0.04684     0     3.41       0  0.489  6.417   66.1  3.0923      2    270       17.8  392.18     8.81
    ====  =======  ====  =======  ======  =====  =====  =====  ======  =====  =====  =========  ======  =======


.. code:: python

    >>> # id 10
    >>> # List of top 5 similar houses (house 10 itself + 4)
    >>> data.loc[pd.DataFrame(ans[1]['results']).id]
    ====  =======  ====  =======  ======  =====  =====  =====  ======  =====  =====  =========  ======  =======
      ..     CRIM    ZN    INDUS    CHAS    NOX     RM    AGE     DIS    RAD    TAX    PTRATIO       B    LSTAT
    ====  =======  ====  =======  ======  =====  =====  =====  ======  =====  =====  =========  ======  =======
      10  0.22489  12.5     7.87       0  0.524  6.377   94.3  6.3467      5    311       15.2  392.52    20.45
       7  0.14455  12.5     7.87       0  0.524  6.172   96.1  5.9505      5    311       15.2  396.9     19.15
       9  0.17004  12.5     7.87       0  0.524  6.004   85.9  6.5921      5    311       15.2  386.71    17.1
      11  0.11747  12.5     7.87       0  0.524  6.009   82.9  6.2267      5    311       15.2  396.9     13.27
       6  0.08829  12.5     7.87       0  0.524  6.012   66.6  5.5605      5    311       15.2  395.6     12.43
    ====  =======  ====  =======  ======  =====  =====  =====  ======  =====  =====  =========  ======  =======

*******************
Supervised Learning
*******************

* And of course we can also train a Supervised Model to predict house prices!
  
.. code:: python

    # j.fit === j.setup
    ans = j.fit(

        # JAI collection name
        name='boston_regression',
        
        # verbose 2 -> shows the loss graph at the end of training
        verbose=2,
        
        # data to be processed - a Pandas DataFrame is expected
        data=data,
        
        # collection type
        db_type='Supervised',
        
        # JAI Collection Foreign Key
        # reference an id column ('id_name') to an already processed JAI collection ('db_parent')
        mycelia_bases=[
            {
            'db_parent':'boston',
            'id_name':'id_house'
            }
        ],

        # Set the column label name and the task type for the Supervised Model
        # Task can be: Regression, Quantile Regression, Classification or Metric Classification
        label=
        {
            'task':'regression',
            'label_name':'PRICE'
        }
    )

Output:

.. code:: bash

      Insert Data: 100%|██████████| 1/1 [00:01<00:00,  1.15s/it]
      Recognized setup args:
      mycelia_bases: [{'db_parent': 'boston', 'id_name': 'id_house'}]
      label: {'task': 'regression', 'label_name': 'PRICE'}
      JAI is working:  50%|█████     |9/18
      [boston_regression] Training:   0%|          | 0/500 [00:00<?, ?it/s]ATraining might not take 500 steps due to early stopping criteria.
      
      [boston_regression] Training:   1%|          | 4/500 [00:01<03:59,  2.07it/s]A
      [boston_regression] Training:   2%|▏         | 8/500 [00:03<03:42,  2.21it/s]A
      [boston_regression] Training:   2%|▏         | 11/500 [00:05<04:27,  1.83it/s]A
      [boston_regression] Training:   3%|▎         | 15/500 [00:07<04:10,  1.94it/s]A
      [boston_regression] Training:   4%|▍         | 20/500 [00:09<03:34,  2.24it/s]A
      [boston_regression] Training:   5%|▌         | 25/500 [00:11<03:25,  2.31it/s]A
      [boston_regression] Training:   6%|▌         | 30/500 [00:13<03:16,  2.39it/s]A
      [boston_regression] Training:   7%|▋         | 34/500 [00:15<03:31,  2.20it/s]A
      [boston_regression] Training:   8%|▊         | 38/500 [00:17<03:32,  2.18it/s]A
      [boston_regression] Training:   9%|▊         | 43/500 [00:19<03:15,  2.34it/s]A
      [boston_regression] Training: 100%|██████████| 500/500 [00:21<00:00, 73.86it/s]A
                                                                                    A
      Done training.
      JAI is working: 100%|██████████|18/18

      Metrics Regression:
      MAE: 2.258793354034424
      MSE: 12.593908309936523
      
      Best model at epoch: 33 val_loss: 0.13

********************
Model Inference
********************

* Now that our Supervised Boston Housing Model is also JAI collection, we can perform Similarity Search, i.e. find similar houses - **also according to the supervised label**, very easily:

.. code:: python

    # every JAI collection can be queried using j.similar()
    ans = j.similar(
        # collection to be queried
        name='boston_regression',
        # let's find houses that are similar to ids 1 and 10
        data=[1, 10]
    )

Output:

.. code:: bash

    Similar: 100%|██████████| 1/1 [00:01<00:00,  1.36s/it]

And now the 'ans' variable holds a JSON:

.. code:: bash

    [{'query_id': 1,
    'results': [{'id': 1, 'distance': 0.0},
    {'id': 91, 'distance': 0.017999378964304924},
    {'id': 94, 'distance': 0.02219889685511589},
    {'id': 96, 'distance': 0.03483652323484421},
    {'id': 90, 'distance': 0.050415001809597015}]},

    {'query_id': 10,
    'results': [{'id': 10, 'distance': 0.0},
    {'id': 7, 'distance': 0.024717235937714577},
    {'id': 209, 'distance': 0.05477815866470337},
    {'id': 211, 'distance': 0.056917279958724976},
    {'id': 9, 'distance': 0.05909169092774391}]}]

And by indexing it back to the original dataframe id's, we have:

.. code:: python

    >>> # id 1
    >>> # List of top 5 similar houses (house 1 itself + 4)
    >>> data.loc[pd.DataFrame(ans[0]['results']).id]
    ====  =======  ====  =======  ======  =====  =====  =====  ======  =====  =====  =========  ======  =======  ==========  =======
      ..     CRIM    ZN    INDUS    CHAS    NOX     RM    AGE     DIS    RAD    TAX    PTRATIO       B    LSTAT    id_house    PRICE
    ====  =======  ====  =======  ======  =====  =====  =====  ======  =====  =====  =========  ======  =======  ==========  =======
       1  0.02731     0     7.07       0  0.469  6.421   78.9  4.9671      2    242       17.8  396.9      9.14           1     21.6
      91  0.03932     0     3.41       0  0.489  6.405   73.9  3.0921      2    270       17.8  393.55     8.2           91     22
      94  0.04294    28    15.04       0  0.464  6.249   77.3  3.615       4    270       18.2  396.9     10.59          94     20.6
      96  0.11504     0     2.89       0  0.445  6.163   69.6  3.4952      2    276       18    391.83    11.34          96     21.4
      90  0.04684     0     3.41       0  0.489  6.417   66.1  3.0923      2    270       17.8  392.18     8.81          90     22.6
    ====  =======  ====  =======  ======  =====  =====  =====  ======  =====  =====  =========  ======  =======  ==========  =======

.. code:: python

    >>> # id 10
    >>> # List of top 5 similar houses (house 10 itself + 4)
    >>> data.loc[pd.DataFrame(ans[1]['results']).id]
    ====  =======  ====  =======  ======  =====  =====  =====  ======  =====  =====  =========  ======  =======  ==========  =======
      ..     CRIM    ZN    INDUS    CHAS    NOX     RM    AGE     DIS    RAD    TAX    PTRATIO       B    LSTAT    id_house    PRICE
    ====  =======  ====  =======  ======  =====  =====  =====  ======  =====  =====  =========  ======  =======  ==========  =======
      10  0.22489  12.5     7.87       0  0.524  6.377   94.3  6.3467      5    311       15.2  392.52    20.45          10     15
       7  0.14455  12.5     7.87       0  0.524  6.172   96.1  5.9505      5    311       15.2  396.9     19.15           7     27.1
     209  0.43571   0      10.59       1  0.489  5.344  100    3.875       4    277       18.6  396.9     23.09         209     20
     211  0.37578   0      10.59       1  0.489  5.404   88.6  3.665       4    277       18.6  395.24    23.98         211     19.3
       9  0.17004  12.5     7.87       0  0.524  6.004   85.9  6.5921      5    311       15.2  386.71    17.1            9     18.9
    ====  =======  ====  =======  ======  =====  =====  =====  ======  =====  =====  =========  ======  =======  ==========  =======

* We can also, of course, perform inference on our model:

.. code:: python

      # every JAI Supervised collection can be used for inference using j.predict()
      ans = j.predict(
         # collection to be queried
         name='boston_regression',
         # let's get prices for the first five houses in the dataset, using their ids
         # also we are dropping the label, as it is not a feature
         data=data.head().drop('PRICE',axis=1)
      )

Output:

.. code:: python

    Predict: 100%|██████████| 1/1 [00:01<00:00,  1.59s/it]

   And now the 'ans' variable holds a JSON:

.. code:: python

    [{'id': 0, 'predict': [24.70072364807129]},
    {'id': 1, 'predict': [21.706649780273438]},
    {'id': 2, 'predict': [31.775901794433594]},
    {'id': 3, 'predict': [34.41084289550781]},
    {'id': 4, 'predict': [34.54452896118164]}]

And by indexing it back to the original dataframe id's, we have:

.. code:: python

    >>> # id 1
    >>> # List of top 5 similar houses (house 1 itself + 4)
    >>> predict_df = pd.DataFrame(ans)
    >>> predict_df = predict_df.set_index('id')
    >>> predict_df.loc[:,'predict'] = predict_df['predict'].apply(lambda x: x[0])
    >>> predict_df['true'] = data['PRICE']
    ====  =========  ======
      ..    predict    true
    ====  =========  ======
       0    24.7007    24
       1    21.7066    21.6
       2    31.7759    34.7
       3    34.4108    33.4
       4    34.5445    36.2
    ====  =========  ======

**********************
Always deployed (REST)
**********************

* Everything in JAI is always instantly deployed and available through REST API.

.. code:: python

    # Similarity Search via REST API

    # import requests libraries
    import requests

    # set Authentication header
    header={'Auth': 'AUTH KEY'}

    # set collection name
    db_name = 'boston'

    # similarity search endpoint
    url_similar = f"https://mycelia.azure-api.net/similar/id/{db_name}"
    body = [1, 10]

    #make the request (PUT)
    ans = requests.put(url_similar, json=body, headers=header)

Output - ans.json():

.. code:: bash

    {
        'similarity': [

        {'query_id': 1,
        'results': [{'id': 1, 'distance': 0.0},
        {'id': 96, 'distance': 0.012930447235703468},
        {'id': 235, 'distance': 0.02305753342807293},
        {'id': 176, 'distance': 0.02424568682909012},
        {'id': 90, 'distance': 0.025710342451930046}]},
        
        {'query_id': 10,
        'results': [{'id': 10, 'distance': 0.0},
        {'id': 7, 'distance': 0.0065054153092205524},
        {'id': 9, 'distance': 0.020906779915094376},
        {'id': 11, 'distance': 0.04773647338151932},
        {'id': 6, 'distance': 0.09080290794372559}]}

        ]
    }

.. code:: python

    # Model Inference via REST API

    # import requests libraries
    import requests
    
    # set Authentication header
    header={'Auth': 'AUTH KEY'}

    # set collection name
    db_name = 'boston_regression'

    # model inference endpoint
    url_predict = f"https://mycelia.azure-api.net/predict/{db_name}"

    # json body
    # note that we need to provide a column named 'id'
    # also note that we drop the 'PRICE' column because it is not a feature
    body = data.reset_index().rename(columns={'index':'id'}).head().drop('PRICE',axis=1).to_dict(orient='records')
    
    #make the request
    ans = requests.put(url_predict, json=body, headers=header)

Output - ans.json():

.. code:: bash

    [{'id': 0, 'predict': [24.70072364807129]},
    {'id': 1, 'predict': [21.706649780273438]},
    {'id': 2, 'predict': [31.775901794433594]},
    {'id': 3, 'predict': [34.41084289550781]},
    {'id': 4, 'predict': [34.54452896118164]}]