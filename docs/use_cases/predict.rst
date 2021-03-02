Running predicts
================

After you're done setting up your database and it is a Supervised database, you can make predictions with your model:

.. code-block:: python

    >>> preds = j.predict(name, DATA_ITEM)
    >>> print(preds)
    [{"id":0, "predict": "class1"}, {"id":1, "predict": "class0"}]

If you trained a classification model, you can have the probabilities for each class.

.. code-block:: python

    >>> preds = j.predict(name, DATA_ITEM, predict_proba=True)
    >>> print(preds)
    [{"id": 0 , "predict"; {"class0": 0.1, "class1": 0.6, "class2": 0.3}}]
