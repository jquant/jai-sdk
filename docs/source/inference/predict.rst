
###############
Model Inference
###############

After fitting your database, you can perform predictions using your model.

.. code-block:: python

    >>> preds = j.predict(name, DATA_ITEM)
    >>> print(preds)
    
    [{"id":0, "predict": "class1"}, {"id":1, "predict": "class0"}]

If you trained a classification model, you can have the probabilities for each class.

.. code-block:: python

    >>> preds = j.predict(name, DATA_ITEM, predict_proba=True)
    >>> print(preds)

    [{"id": 0 , "predict"; {"class0": 0.1, "class1": 0.6, "class2": 0.3}}]

.. note::

    The method :code:`predict` has a default :code:`batch_size=2**20`, which will result in 
    :code:`ceil(n_samples/batch_size) + 2` requests. We **DON'T** recommend changing the default 
    value as it could reduce the performance of the API.