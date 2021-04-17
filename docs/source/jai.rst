##########
Jai Module
##########

*********
Jai class
*********

.. automodule:: jai.jai
   :members:
   :undoc-members:
   :show-inheritance:

************
Setup kwargs
************

Parameters that should be passed as a dictionary in compliance with the
API methods. In other words, every kwarg argument should be passed as if
it were in the body of a POST method.

* **overwrite** (*bool*) -- If setup should continue even if there's a database 
  set up with the given name. *Default is False*
   
* **hyperparams** (*dict*) -- Model hyperparams depending on the type of the 
  database: 

  * **Image**:  
    
    * **model_name** (*torchvision*) -- Model for image preprocessing
      {"resnet50", "resnet18", "alexnet", "squeezenet", "vgg16", "densenet", 
      "inception", "googlenet", "shufflenet", "mobilenet", "resnext50_32x4d",
      "wide_resnet50_2", "mnasnet"}. *Default is "vgg16"*.
    * **mode** -- last layer of the model, varies for each model
      {"classifier", "dense", "conv", "avgpool" or "int"}. *Default is -3*.
    * **resize_H** (*int*) -- Height of image resizing, must be greater or
      equal to 224. *Default is 224*.
    * resize_W: (int) width of image resizing, must be greater or
      equal to 224. *Default is 224*.

  * **FastText**:

    * **minn** (*int*) -- Min length of char ngram. *Default is 0*.
    * **maxn** (*int*) -- Max length of char ngram. *Default is 0*.
    * **dim** (*int*) -- Final latent layer dimension. *Default is 128*.
    * **epoch** (*int*) -- Number of epochs. *Default is 10*.
    * **model** (*str*) -- Unsupervised fasttext model {"cbow", "skipgram"},
      *Default is "skipgram"*.
    * **lr** (*float*) -- Learning rate. *Default is 0.05*.
    * **ws** (*int*) -- Size of the context window. *Default is 5*.
    * **minCount** (*int*) -- Minimal number of word occurences. *Default is 0*.
    * **neg** (*int*) -- Number of negatives sampled. *Default is 5*.
    * **wordNgrams** (*int*) -- Max length of word ngram. *Default is 1*.
    * **loss** (*str*) -- Loss function {"ns", "hs", "softmax", "ova"}. *Default is "ns"*.
    * **bucket** (*int*) -- Number of buckets. *Default is 2000000*.
    * **lrUpdateRate** (*int*) -- Change the rate of updates for the
      learning rate. *Default is 1000*.
    * **t** (*float*) -- Sampling threshold. *Default is 0.0001*.

  * **Text**:
    
    * **nlp_model** (*transformers*) -- Model name for text preprocessing.
    * **max_length** (*int*) -- Controls the maximum length to use by one
      of the truncation/padding parameters. *Default is 100*.

  * **TextEdit**:

    * **nt** (*int*) -- Amount of training samples. *Default is 1000*.
    * **nr** (*int*) -- Amount of generated training samples. *Default is 1000*.
    * **nb** (*int*) -- Amount of  base items. *Default is 1385451*.
    * **k** (*int*) -- Amount sampling threshold for knn. Only trains with samples below 2*k closest pairs. *Default is 100*.
    * **epochs** (*int*) -- Amount of epochs. *Default is 20*.
    * **shuffle_seed** (*int*) -- Seed for shuffle. *Default is 808*.
    * **batch_size** (*int*) -- Batch size for sgd. *Default is 128*.
    * **test_batch_size** (*int*) -- Batch size for test. *Default is 1024*.
    * **channel** (*int*) -- Amount of channels. *Default is 8*.
    * **embed_dim** (*int*) -- Output dimension. *Default is 128*.
    * **random_train** (*bool*) -- Generate random training samples and replace. 
      *Default is False*.
    * **random_append_train** (*bool*) -- Generate random training samples and 
      append. *Default is False*.
    * **maxl** (*int*) -- Max length of strings. *Default is 0*.

  * **"Supervised" or "Unsupervised"**:

    * **batch_size** (*int*) -- Batch size for training. *Default is 512*.
    * **learning_rate** (*float*) -- Initial learning rate. *Default is 0.001*.
    * **encoder_layer** (*str*) -- Structure for the encoder layer {"2L", "tabnet"}
      *Default is "2L"*.
    * **decoder_layer** (*str*) -- Structure for the decoder layer {"2L", "2L_LN", "2L_BN", "1L"}. 
      *Default is "2L"*.
    * **dropout_rate** (*int*) -- Dropout rate for the encoder layer. *Default is 0.1*.
    * **momentum** (*int*) -- momentum param for batch norm for the encoder layer. *Default is 64*.
    * **pretraining_ratio** (*int*) -- rate of feature masking on self-supervised training. 
      *Default is 0.1*.
    * **hidden_latent_dim** (*int*) -- Hidden layer size. *Default is 64*.
    * **encoder_steps** (*int*) -- Number of sucessive steps in the newtork (usually 
      between 3 and 10), only when encoder is tabnet. *Default is 3*.


* **num_process** (*dict*) -- (*Only for db_type Supervised and Unsupervised*) 
  Parameters defining how numeric values will be processed.
   
  * **embedding_dim** (*int*) -- Initial embedding dimension. If set to 0 then 
    no embedding is made before the encoder. *Default is 8*.
  * **scaler** (*sklearn*) -- Scaler for numeric values {"maxabs", "minmax", "normalizer", 
    "quantile", "robust", "standard"}. *Default is "standard"*
  * **fill_value** (*number*) -- Fill value for missing values. *Default is 0*.

* **cat_process** (*dict*) -- (*Only for db_type Supervised and Unsupervised*) 
  Parameters defining how categorical values will be processed.
   
  * **embedding_dim** (*int*) -- Initial embedding dimension. If set to 0 then 
    no embedding is made before the encoder. *Default is 32*.
  * **fill_value** (*str*) -- Fill value for missing values. *Default is "_other"*.
  * **min_freq** (*str*) -- Number of times a category has to occur to be valid,
    otherwise we substitute by fill_value. *Default is 3*.

* **datetime_process** (*dict*) -- (*Only for db_type Supervised and Unsupervised*) 
  Parameters defining how datetime values will be processed.
    
  * **embedding_dim** (*int*) -- Initial embedding dimension. *Default is 32*.

* **mycelia_bases** (*list of dicts*) -- (*Only for db_type Supervised and Unsupervised*) Related already 
  processed data that will be used in the setup of this new one. If a column has id values that 
  represent a database already preprocessed, then:

  * **db_parent** (*str*) -- (*required*) Name of the preprocessed database.
  * **id_name** (*str*) -- (*required*) Name of the column with the id values in the current table.
  * **embedding_dim** (*int*) -- Initial embedding dimension. *Default is 128*.

* **label** (*dict*) -- (*Only for db_type Supervised*) Label of each ID.

  * **task** (*str*) -- (*required*) Supervised task type {"classification", "metric_classification", "regression", 
    "quantile_regression"}.
  * **label_name** (*str*) -- (*required*) Column name with target values.
  * **quantiles** (*list of floats*) -- quantiles for quantile_regression.

* **split** (*dict*) -- (*Only for db_type Supervised*) How data will be split in the training process.
   
  * **type** (*str*) -- How to split the data in train and test {random, stratified}. *Default is "random"*.
  * **split_column** (*str*) -- (*Mandatory when type is stratified*) Name of column as reference for the split. *Default is ""*.
  * **test_size** (*float*) -- Size of test for the split. *Default is 0.2*.

* **patience** (*int*) -- (Supervised and Self-Supervised only) Number of validation checks with no improvement after which training will be stopped.
  *Default is 7*.

* **min_delta** (*float*) -- (Supervised and Self-Supervised only) Minimum change in the monitored quantity (loss) to qualify as an improvement,
  i.e. an absolute change of less than min_delta, will count as no improvement. *Default is 1e-5*.
