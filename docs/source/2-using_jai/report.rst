###################
Setting Fit Reports
###################

After SelfSupervised and Supervised setups are done, we display details about the setup process through 
the method :code:`j.report`. 
In this page, we'll take a look on the information that is available.

**************
Verbose Levels
**************

There's 2 levels of verbose that set the amount of information that is retrieved. 
The default value for default on the setup/fit method is :code:`verbose=1`, while using the :code:`j.report` 
the default value is :code:`verbose=2`.
And :code:`verbose=0` will just not return any information.

Some information is only displayed when accessing the dictionary returned on :code:`j.report(return_report=True)`. 
Here are the information that is available with the reporting system:

verbose = 1
===========
Always printed when report method is used.

* **Loading from checkpoint**: Specifies which epoch the model had the best performance and is used as the final 
  model for inference.
* **Model Evaluation** [*Supervised only*]: Metrics on the test set. MSE and MAE for regression, quantile 0.5 is 
  used when quantile_regression. Scikit-learn's `classification_report`_ (precision, recall and f1) when classification 
  or metric_classification.

verbose = 2 
===========
Contains all content of :code:`verbose=1` plus:

* **Model Training**: Plots the loss graph of the training. When :code:`return_report=True`, returns the values of the 
  loss for the training set and validation set for each epoch.

All content below is available only when :code:`return_report=True`

* **Auto lr finder**: If :code:`learning_rate=0`, then we'll try and find the best appropriate learning rate.

All content below is available only on Supervised models.

* **Validation Ids/Evaluation Ids**: List the ids chosen for the validation set and evaluation/test set
* **Metrics Train/Metrics Validation**: The same metrics calculated on "Model Evaluation" but on the training set and 
  validation set respectively.
* **Baseline Model**: The same metrics of "Model Evaluation" on the test set, but using a Baseline Model (please check 
  `sklearn.dummy`_ models). For regression cases, we evaluate the mean and the median as a baseline. For classification 
  cases, we use stratified, uniform and most_frequent models as baseline.
* **Optimal Thresholds**: List the probability thresholds that maximize :code:`true positive rate - false negative rate` 
  for each class. Since it's calculated in a OneVsAll manner, the probabilities don't sum up to one.

.. _sklearn.dummy: https://scikit-learn.org/stable/modules/classes.html#module-sklearn.dummy
.. _classification_report: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html#sklearn.metrics.classification_report
