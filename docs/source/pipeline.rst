.. _common_use_pipeline:

##############
The Fit Method
##############
 
.. image:: /source/images/j_fit.png
   :scale: 15
   :align: center
   :class: no-scaled-link

To start setting your dataset, you first need to add and fit it to your JAI environment. Using the :code:`j.fit` method is the better way to achieve this. This method adds your raw data to your JAI environment, trains the data based on the chosen model type, and stores your model's latent vector representation in a collection.

.. note::

    A collection is an effective way to store vectors that permits a fast similarity search between these vectors. 

.. note::

    *Latent vectors* are created when training some model in a neural network (NN) and don't directly correlate with the data passed by the NN. They are just part of your trained model that helps to define a correct output when we need to predict something on this model.

The fit method allows training on different types of data, such as Tabular, Text and Image. 
Here is the list of models that JAI supports:

- **Tabular data:** supervised and self-supervised models
- **Text data:** NLP Transformers models, FastText and Edit Distance Language model
- **Image data:** JAI works with all torchvision image models.

The sections below show more information about using these models in JAI. For the complete reference of the fit method, look at "API reference".

.. important:: 
    
    JAI deletes all raw data you sent after running :code:`j.fit`, keeping internally only with the latent vector representation of your training data. 


Basics
------
The :code:`j.fit` method has three main parameters: :code:`name`, :code:`data` and :code:`db_type`:

.. code:: python

    j.fit(
        name='Collection_name',
        data=data,
        db_type='SelfSupervised'
    )

- The :code:`name` parameter is the name you will give for the data you are fitting. It must be a string with a **maximum of 32 characters**.

- :code:`data` is the data that you want to fit. It must be a :code:`pandas.DataFrame` or a :code:`pandas.Series`. For using image data, the images first have to be encoded to, after, being inserted to fit, as shown in "Fitting Images".

- :code:`db_type` is the parameter that defines what type of training will be realized by the fit method. The possible values are :code:`'Supervised'`, :code:`'SelfSupervised'`, :code:`'Text'`, :code:`'FastText'`, :code:`'TextEdit'` and :code:`'Image'`. Each of these has its own set of parameters and hyperparameters. For more information about them, check "Fitting Tabular data", "Fitting Text data", and "Fitting Image data".

JAI uses your data index to perform a lot of methods internally. You can define the index of your data in two ways: using the pandas' index or creating a column named :code:`'id'`. When you don't make an :code:`'id'` column, JAI automatically considers your data pandas' index; on the other hand, JAI uses your :code:`'id'` column as your data index. So, take care with duplicated values.


