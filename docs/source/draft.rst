The most common pipeline when using JAI is composed of the following steps:
Adding data
Choosing a model type to train
Running the fit method
Making inference

This section will show what the possibilities to do all of these steps are.

Adding data

To get started with JAI, first, it's needed that you add some data to JAI. Calling ":code:`j.add_data()` you can add some data to your environment on JAI. 

".. code:: python
	

It's on this data that JAI will perform train and generate your model after running a ":code:`fit`" method. JAI trains a model on this data, get the latent vectors of this training and deletes the raw data you inserted.