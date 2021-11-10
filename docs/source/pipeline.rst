.. _common_use_pipeline:

###################
Common use pipeline
###################
 
For almost all cases, when starting creating your database environment in JAI, a user should follow a standard JAI methods pipeline.

To start setting your dataset, you first need to add it to your JAI environment. Then, based on data type (Tabular, Text or Image), you should choose the desired model training type and fit it using your defined training hyperparameters. After fit completion, you're able to make inferences on your provided data using the :code:`j.predict` or :code:`j.similar`` methods.

.. important:: 
    
    JAI deletes all raw data you sent after running :code:`j.fit`, keeping internally only with the latent vector representation of your training data. 

.. note::

    *Latent vectors* are created when training some model in a neural network (NN) and don't directly correlate with the data passed by the NN. They are just part of your trained model that helps to define a correct output when we need to predict something on this model.