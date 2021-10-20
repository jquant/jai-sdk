#####################################
JAI in 5 minutes - California Housing
#####################################

************************
What are we going to do?
************************

In this quick demo, we will use JAI to:

* Train and deploy models into a secure and scalable production-ready environment
* Similarity Search - Given a house Id or attributes, retrieve similar houses 
* Model Inference - Predict house prices using a supervised machine learning model


***************
Install JAI
***************

Install JAI with pip

.. code:: bash

    pip install jai-sdk --user
      
*****************
Import JAI
*****************

.. code:: python

    >>> from jai import Jai

*****************
Generate Auth Key
*****************

JAI requires an Auth Key to organize and secure collections. You can easily generate your free-forever auth-key by running the command below:

.. code:: python

    >>> Jai.get_auth_key(email='email@mail.com', firstName='Jai', lastName='Z')
    201

Please note that your Auth Key will be sent to your e-mail, so please make sure to use a valid address and check your spam folder.

***************
Start JAI
***************

* Use your Auth Key to instantiate JAI:

.. code:: python

    >>> j = Jai('AUTH KEY')

************************
Self-Supervised Learning
************************

* JAI 

.. code:: python
    import pandas as pd
    from sklearn.datasets import fetch_california_housing
      
    # load dataset
    data, labels = fetch_california_housing(as_frame=True, return_X_y=True)
    
    # send data to JAI for feature extraction
    j.fit(
        # JAI collection name
        name='california',

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

    Insert Data: 100%|██████████| 2/2 [00:03<00:00,  1.87s/it]

    Recognized setup args:
    - db_type: SelfSupervised
    - hyperparams: 
      * pretraining_ratio: 0.8
      * learning_rate    : 0.0003
    - overwrite: False

    Training might finish early due to early stopping criteria.
    JAI is working:  56%|█████▋    |9/16 [00:15]
    [california] Training:   0%|          | 0/500 [00:00<?, ?it/s]
    [california] Training:   1%|▏         | 7/500 [00:02<03:03,  2.68it/s]
    [california] Training:   3%|▎         | 14/500 [00:04<02:25,  3.35it/s]
    [california] Training:   3%|▎         | 17/500 [00:05<02:56,  2.74it/s]
    [california] Training:   5%|▍         | 24/500 [00:07<02:25,  3.27it/s]
    [california] Training:   6%|▌         | 28/500 [00:09<02:37,  3.00it/s]
    [california] Training:   7%|▋         | 33/500 [00:10<02:31,  3.07it/s]
    [california] Training:   8%|▊         | 41/500 [00:12<02:05,  3.66it/s]
    [california] Training:   9%|▉         | 46/500 [00:14<02:09,  3.50it/s]
    [california] Training: 100%|██████████| 500/500 [00:15<00:00, 88.85it/s]
    JAI is working: 100%|██████████|16/16 [00:39]                           

*****************
Similarity Search
*****************

* Now that our California Housing data is in a JAI collection, we can perform Similarity Search, i.e. find similar houses, very easily:

.. code:: python

    # every JAI collection can be queried using j.similar()
    ans = j.similar(
        # collection to be queried
        name='california',
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
    {'id': 17178, 'distance': 0.01419779472053051},
    {'id': 17644, 'distance': 0.015902765095233917},
    {'id': 1551, 'distance': 0.017771419137716293},
    {'id': 1614, 'distance': 0.019414082169532776}]},
    {'query_id': 10,
    'results': [{'id': 10, 'distance': 0.0},
    {'id': 559, 'distance': 0.00288062053732574},
    {'id': 12496, 'distance': 0.0029994570650160313},
    {'id': 16056, 'distance': 0.0062744226306676865},
    {'id': 16036, 'distance': 0.006555804051458836}]}]

And by indexing it back to the original dataframe id's, we have:

.. code:: python
    >>> # import tabulate only to print the results 
    >>> from tabulate import tabulate  # (not required)
    >>>
    >>> # id 1
    >>> # List of top 5 similar houses (house 1 itself + 4)
    >>> result_1 = data.loc[pd.DataFrame(ans[0]['results'])['id']]
    >>> print(tabulate(result_1, headers='keys', tablefmt='rst'))
    =====  ========  ==========  ==========  ===========  ============  ==========  ==========  ===========
       ..    MedInc    HouseAge    AveRooms    AveBedrms    Population    AveOccup    Latitude    Longitude
    =====  ========  ==========  ==========  ===========  ============  ==========  ==========  ===========
        1    8.3014          21     6.23814     0.97188           2401     2.10984       37.86      -122.22
    17178    6.7606          15     6.42636     0.98708           2222     2.8708        37.51      -122.47
    17644    6.8088          20     6.73788     1.00152           2062     3.12424       37.26      -121.9
     1551    6.6204          16     6.7293      0.965834          2464     3.23784       37.75      -121.94
     1614    7.6202          27     7.12208     0.987013          2212     2.87273       37.86      -122.09
    =====  ========  ==========  ==========  ===========  ============  ==========  ==========  ===========


.. code:: python

    >>> # id 10
    >>> # List of top 5 similar houses (house 10 itself + 4)
    >>> result_10 = data.loc[pd.DataFrame(ans[1]['results'])['id']]
    >>> print(tabulate(result_10, headers='keys', tablefmt='rst'))
    =====  ========  ==========  ==========  ===========  ============  ==========  ==========  ===========
       ..    MedInc    HouseAge    AveRooms    AveBedrms    Population    AveOccup    Latitude    Longitude
    =====  ========  ==========  ==========  ===========  ============  ==========  ==========  ===========
       10    3.2031          52     5.47761      1.0796            910     2.26368       37.85      -122.26
      559    3.4762          52     5.30508      1.09322           979     2.07415       37.76      -122.24
    12496    3.2963          52     5.22396      1.07292           825     2.14844       38.57      -121.45
    16056    3.5302          52     5.58606      1.09368          1092     2.37908       37.76      -122.49
    16036    3.2875          48     5.33123      1.0694            962     3.0347        37.72      -122.46
    =====  ========  ==========  ==========  ===========  ============  ==========  ==========  ===========

*******************
Supervised Learning
*******************

* And of course we can also train a Supervised Model to predict house prices!
  
.. code:: python

    # j.fit === j.setup
    data_sup = labels.reset_index().rename(columns={"index": "id_house"})
    ans = j.fit(
        # JAI collection name
        name='california_regression',
        
        # verbose 2 -> shows the loss graph at the end of training
        verbose=2,
        
        # data to be processed - a Pandas DataFrame is expected
        data=data_sup,
        
        # collection type
        db_type='Supervised',
        
        # JAI Collection Foreign Key
        # reference an id column ('id_name') to an already processed JAI collection ('db_parent')
        mycelia_bases=[
            {
            'db_parent':'california',
            'id_name':'id_house'
            }
        ],

        # Set the column label name and the task type for the Supervised Model
        # Task can be: Regression, Quantile Regression, Classification or Metric Classification
        label=
        {
            'task':'regression',
            'label_name':'MedHouseVal'
        }
    )

Output:

.. code:: bash

    Insert Data: 100%|██████████| 2/2 [00:02<00:00,  1.34s/it]

    Recognized setup args:
    - db_type: Supervised
    - pretrained_bases: [{"db_parent": "california", "id_name": "id_house", "embedding_dim": 128, "aggregation_method": "sum"}]
    - label: 
      * label_name: MedHouseVal
      * task      : regression
    - overwrite: False

    Training might finish early due to early stopping criteria.
    JAI is working:  44%|████▍     |8/18 [00:27]
    [california_regression] Training:   0%|          | 0/500 [00:00<?, ?it/s]
    [california_regression] Training:   2%|▏         | 11/500 [00:02<01:56,  4.19it/s]
    [california_regression] Training:   3%|▎         | 16/500 [00:04<02:14,  3.61it/s]
    [california_regression] Training:   6%|▌         | 28/500 [00:05<01:32,  5.11it/s]
    [california_regression] Training:   8%|▊         | 39/500 [00:07<01:22,  5.55it/s]
    [california_regression] Training:   9%|▉         | 44/500 [00:09<01:36,  4.75it/s]
    [california_regression] Training:  11%|█         | 56/500 [00:11<01:20,  5.49it/s]
    [california_regression] Training: 100%|██████████| 500/500 [00:12<00:00, 90.62it/s]
    JAI is working: 100%|██████████|18/18 [00:48]      

    Setup Report:
    Metrics Regression:
    MAE: 0.48097676038742065
    MSE: 0.44630882143974304
    MAPE: 0.32101190090179443
    R2 Score: 0.6594125834889224
    Pinball Loss 0.5: 0.24048838019371033

    Best model at epoch: 53 val_loss: 0.37

********************
Model Inference
********************

* Now that our Supervised California Housing Model is also JAI collection, we can perform Similarity Search, i.e. find similar houses - **also according to the supervised label**, very easily:

.. code:: python

    # every JAI collection can be queried using j.similar()
    ans = j.similar(
        # collection to be queried
        name='california_regression',
        # let's find houses that are similar to ids 1 and 10
        data=[1, 10]
    )

Output:

.. code:: bash

    Similar: 100%|██████████| 1/1 [00:00<00:00,  1.16it/s]

And now the 'ans' variable holds a JSON:

.. code:: bash

    [{'query_id': 1,
    'results': [{'id': 1, 'distance': 0.0},
    {'id': 1639, 'distance': 0.8954934477806091},
    {'id': 16009, 'distance': 1.019099473953247},
    {'id': 9404, 'distance': 1.3721085786819458},
    {'id': 17098, 'distance': 1.373133897781372}]},
    {'query_id': 10,
    'results': [{'id': 10, 'distance': 0.0},
    {'id': 18599, 'distance': 0.09487645328044891},
    {'id': 553, 'distance': 0.41577231884002686},
    {'id': 1759, 'distance': 0.4182438850402832},
    {'id': 12, 'distance': 0.607153594493866}]}]

And by indexing it back to the original dataframe id's, we have:

.. code:: python

    >>> # id 1
    >>> # List of top 5 similar houses (house 1 itself + 4)
    >>> result_1 = data.loc[pd.DataFrame(ans[0]['results'])['id']]
    >>> print(tabulate(result_1, headers='keys', tablefmt='rst'))
    =====  ========  ==========  ==========  ===========  ============  ==========  ==========  ===========
       ..    MedInc    HouseAge    AveRooms    AveBedrms    Population    AveOccup    Latitude    Longitude
    =====  ========  ==========  ==========  ===========  ============  ==========  ==========  ===========
        1    8.3014          21     6.23814     0.97188           2401     2.10984       37.86      -122.22
     1639    8.1489          18     6.60082     1.00136           1634     2.22616       37.89      -122.18
    16009    6.203           38     6.26432     1.02423           2263     2.49229       37.74      -122.45
     9404    6.7809          30     5.88188     0.98255           1775     2.38255       37.88      -122.54
    17098    7.1088          33     6.98061     0.969388          2681     2.73571       37.46      -122.25
    =====  ========  ==========  ==========  ===========  ============  ==========  ==========  ===========

.. code:: python

    >>> # id 10
    >>> # List of top 5 similar houses (house 10 itself + 4)
    >>> result_10 = data.loc[pd.DataFrame(ans[1]['results'])['id']]
    >>> print(tabulate(result_10, headers='keys', tablefmt='rst'))
    =====  ========  ==========  ==========  ===========  ============  ==========  ==========  ===========
       ..    MedInc    HouseAge    AveRooms    AveBedrms    Population    AveOccup    Latitude    Longitude
    =====  ========  ==========  ==========  ===========  ============  ==========  ==========  ===========
       10    3.2031          52     5.47761      1.0796            910     2.26368       37.85      -122.26
    18599    2.7933          51     5.56092      1.11494          1078     2.47816       37.12      -122.12
      553    2.9899          52     5.07748      1.07506           915     2.2155        37.77      -122.26
     1759    3.5848          47     5.50292      1.05556           797     2.33041       37.94      -122.33
       12    3.075           52     5.32265      1.01282          1098     2.34615       37.85      -122.26
    =====  ========  ==========  ==========  ===========  ============  ==========  ==========  ===========

* We can also, of course, perform inference on our model:

.. code:: python

      # every JAI Supervised collection can be used for inference using j.predict()
      ans = j.predict(
         # collection to be queried
         name='california_regression',
         # let's get prices for the first five houses in the dataset, using their ids
         data=data.head()
      )

Output:

.. code:: bash

    Predict: 100%|██████████| 1/1 [00:04<00:00,  4.68s/it]

And now the 'ans' variable holds a JSON:

.. code:: bash

    [{'id': 0, 'predict': 4.297857761383057},
    {'id': 1, 'predict': 4.351778507232666},
    {'id': 2, 'predict': 4.426850318908691},
    {'id': 3, 'predict': 3.6801629066467285},
    {'id': 4, 'predict': 2.8943865299224854}]

And by indexing it back to the original dataframe id's, we have:

.. code:: python

    >>> # id 1
    >>> # List of top 5 similar houses (house 1 itself + 4)
    >>> predict_df = pd.DataFrame(ans)
    >>> predict_df = predict_df.set_index('id')
    >>> predict_df['true'] = labels
    >>> print(tabulate(predict_df, headers='keys', tablefmt='rst'))
    ====  =========  ======
      id    predict    true
    ====  =========  ======
       0    4.29786   4.526
       1    4.35178   3.585
       2    4.42685   3.521
       3    3.68016   3.413
       4    2.89439   3.422
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
    db_name = 'california'

    # similarity search endpoint
    url_similar = f"https://mycelia.azure-api.net/similar/id/{db_name}"
    body = [1, 10]

    #make the request (PUT)
    ans = requests.put(url_similar, json=body, headers=header)

Output - ans.json():

.. code:: bash

    {'similarity': [{'query_id': 1,
                     'results': [{'distance': 0.0, 'id': 1},
                                 {'distance': 0.01419779472053051, 'id': 17178},
                                 {'distance': 0.015902765095233917, 'id': 17644},
                                 {'distance': 0.017771419137716293, 'id': 1551},
                                 {'distance': 0.019414082169532776, 'id': 1614}]},
                     {'query_id': 10,
                     'results': [{'distance': 0.0, 'id': 10},
                                 {'distance': 0.00288062053732574, 'id': 559},
                                 {'distance': 0.0029994570650160313, 'id': 12496},
                                 {'distance': 0.0062744226306676865, 'id': 16056},
                                 {'distance': 0.006555804051458836, 'id': 16036}]}]}

.. code:: python

    # Model Inference via REST API

    # import requests libraries
    import requests
    
    # set Authentication header
    header={'Auth': 'AUTH KEY'}

    # set collection name
    db_name = 'california_regression'

    # model inference endpoint
    url_predict = f"https://mycelia.azure-api.net/predict/{db_name}"

    # json body
    # note that we need to provide a column named 'id'
    # also note that we drop the 'PRICE' column because it is not a feature
    body = data.reset_index().rename(columns={'index':'id'}).head().to_dict(orient='records')
    
    #make the request
    ans = requests.put(url_predict, json=body, headers=header)

Output - ans.json():

.. code:: bash

    [{'id': 0, 'predict': 4.297857761383057},
     {'id': 1, 'predict': 4.351778507232666},
     {'id': 2, 'predict': 4.426850318908691},
     {'id': 3, 'predict': 3.6801629066467285},
     {'id': 4, 'predict': 2.8943865299224854}]