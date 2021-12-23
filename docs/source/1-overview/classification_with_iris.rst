.. _flower classification:

=======================================
Flower classification with Iris dataset
=======================================

************************
What are we going to do?
************************

In this quick demo, we will use JAI to:

* Train and deploy models into a secure and scalable production-ready environment.
* Classification - Given a list of flowers attributes, classify the flower within three possible classes.
* Model Inference - Predict the class of new flowers and check the results.


*********
Start JAI
*********

Firstly, you'll need to install the JAI package and generate your auth key, as explained in the 
:ref:`Getting Started <source/1-overview/getting_started:Getting Started>` section. 

With your authentication key, start authenticating in your JAI account:

.. code-block:: python

    >>> AUTH_KEY = "insert_your_auth_key_here"
    >>> j = Jai(AUTH_KEY) 

*******************
Dataset quick look
*******************

This dataset is frequently used on machine learning classification classes and a further explanation of it 
can be found in the `sklearn documentation <https://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html>`_. 
The dataset presents *150 rows* equally divided between the three flower classes, which are *Setosa*, *Versicolour* and *Virginica*.        

Let's have a quick glance on come columns of this dataset below:  

.. code-block:: python

    >>> # Importing other necessary libs
    >>> import numpy as np
    >>> import pandas as pd
    >>> from tabulate import tabulate
    >>> from sklearn.datasets import load_iris
    ...
    >>> # Loading dataframe
    >>> df = pd.DataFrame(load_iris(as_frame=True).data)
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

Now we will train a Supervised Model to classify each flower example within the three classes using the attributes 
previously shown.
  
.. code-block:: python
    
    >>> from sklearn.model_selection import train_test_split
    ...
    >>> # Splitting the dataset to demonstrate j.predict
    >>> X_train, X_test, y_train, y_test = train_test_split(
    ...             df, target, test_size=0.3, random_state=42)
    ...
    >>> # Creating a training table with the target
    >>> train = pd.concat([X_train,y_train],axis=1)
    ...
    >>> # For the supervised model we have to pass the dataframe with the label to JAI
    >>> train = pd.concat([X_train,y_train],axis=1)
    ...
    >>> # Training the classification model
    >>> j.fit(
    ...     # JAI collection name    
    ...     name="iris_supervised",  
    ...     # Data to be processed - a Pandas DataFrame is expected
    ...     data=train, 
    ...     # Collection type
    ...     db_type='Supervised', 
    ...     # Verbose 2 -> shows the loss graph at the end of training
    ...     verbose=2,
    ...     # The split type as stratified guarantee that the same proportion of both classes are maintained for train, validation and test
    ...     split = {'type':'stratified'},
    ...     # When we set task as *classification* we use CrossEntropy Loss
    ...     label = {
    ...         "task": "classification",
    ...         "label_name": "target"
    ...         }
    ...     # You can uncomment this line if you wish to test different parameters and maintain the same collection name
    ...     # overwrite = True
    ... )
    
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

For more information about the :code:`j.fit` args you can access 
:ref:`the reference part <source/reference/jai:jai python api>` of our documentation.

***************
Model Inference
***************

Now that our Supervised Model is also JAI collection, we can perform predictions with it, applying the model to new examples very easily. Let's do it firstly without predict_proba:

.. code-block:: python

    >>> # Now we will make the predictions
    >>> # In this case, it will use 0.5 (which is default) as threshold to return the predicted class
    >>> ans = j.predict(
    ...
    ...     # Collection to be queried
    ...     name='iris_supervised',
    ...    
    ...     # This will make your ansewer return as a dataframe
    ...     as_frame=True,
    ...     
    ...     # Here you will pass a dataframe to predict which examples are default or not
    ...     data=X_test
    ... )

Now let's put y_test alongside the predicted classes. Be careful when doing this: JAI returns the answers with sorted indexes.

.. code-block:: python

    >>> # ATTENTION: JAI ALWAYS RETURNS THE ANSWERS ORDERED BY ID! Bringing y_test like this will avoid mismatching
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

    >>> print(metrics.classification_report( ans["y_true"],ans["predict"],target_names=['0','1','2']))
    
                  precision    recall  f1-score   support

               0       1.00      1.00      1.00        19
               1       1.00      1.00      1.00        13
               2       1.00      1.00      1.00        13

        accuracy                           1.00        45
       macro avg       1.00      1.00      1.00        45
    weighted avg       1.00      1.00      1.00        45
    
If you wish to define your threshold or use the predicted probabilities to rank the answers, we can make the predictions as follows:

.. code-block:: python
    
    >>> ans = j.predict(
    ...     
    ...     # Collection to be queried
    ...     name='iris_supervised',
    ...     
    ...     # This will bring the probabilities predicted
    ...     predict_proba = True,
    ...     
    ...     # This will make your ansewer return as a dataframe
    ...     as_frame=True,
    ...     
    ...     # Here you will pass a dataframe to predict which examples are default or not
    ...     data=X_test
    ... )
    ...
    >>> # ATTENTION: JAI ALWAYS RETURNS THE ANSWERS ORDERED BY ID! Bringing y_test like this will avoid mismatching
    >>> ans["y_true"] = y_test
    >>> print(tabulate(ans.head(), headers='keys', tablefmt='rst'))
    
    ====  ========  =========  =========  =========  ================  ========
      id         0          1          2    predict    probability(%)    y_true
    ====  ========  =========  =========  =========  ================  ========
       4  0.967401  0.0158325  0.0167661          0             96.74         0 
       9  0.975747  0.0116164  0.0126364          0             97.57         0
      10  0.962914  0.0186806  0.0184058          0             96.29         0
      11  0.969209  0.0147728  0.0160187          0             96.92         0
      12  0.977361  0.0108368  0.0118019          0             97.74         0
    ====  ========  =========  =========  =========  ================  =======
    
    >>> # Calculating AUC Score
    >>> roc_auc_score(ans["y_true"], np.array(ans[["0","1","2"]]), multi_class='ovr')
     
    1.0
    
Even though this result might scare you, JAI backend is made to provide a robust performance and prevent overfitting. 


******************************
Making inference from REST API
******************************

Everything in JAI is always instantly deployed and available through REST API, which makes most 
of the job of putting your model in production much easier!

.. code-block:: python
    
    >>> # Import requests libraries
    >>> import requests
    ...
    >>> AUTH_KEY = "insert_your_auth_key_here"
    ...
    >>> # Set Authentication header
    >>> header = {'Auth': AUTH_KEY}
    ...
    >>> # Set collection name
    >>> db_name = 'iris_supervised' 
    ...
    >>> # Model inference endpoint
    >>> url_predict = f"https://mycelia.azure-api.net/predict/{db_name}"
    ...
    >>> # Json body
    >>> # Note that we need to provide a column named 'id'
    >>> # Also note that we drop the 'PRICE' column because it is not a feature
    >>> body = X_test.reset_index().rename(columns={'index':'id'}).head().to_dict(orient='records')
    ...
    >>> # Make the request
    >>> ans = requests.put(url_predict, json=body, headers=header)
    >>> ans.json()

    [{'id': 18, 'predict': 0},
    {'id': 73, 'predict': 1},
    {'id': 76, 'predict': 1},
    {'id': 78, 'predict': 1},
    {'id': 118, 'predict': 2}]

For more discussions about this example, 
join our `slack community <https://join.slack.com/t/getjai/shared_invite/zt-sfkm3tpg-oJuvdziWgtaFEaIUUKWUV>`_!