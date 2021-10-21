##########################
Creating Supervised Models
##########################

Fit applying Supervised Model
===============================

.. code-block:: python

    >>> j.fit(name, data, db_type='Supervised', label={"task": "classification", "label_name": "my_label"})


Tasks
-----

Here are the possible tasks when using a Supervised model:

- classification
- metric_classification
- regression
- quantile_regression


.. note::
    In case of usage of datetime data types, make sure to use a good format. We suggest the format :code:`"%Y-%m-%d %H:%M:%S "`.
    The code used to identify the datetime columns is as follows:
    
    .. code-block:: python
    
        cat_columns = dataframe.select_dtypes("O").columns
        dataframe[cat_columns] = dataframe[cat_columns].apply(pd.to_datetime,
                                                            errors="ignore")
