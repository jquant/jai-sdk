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

