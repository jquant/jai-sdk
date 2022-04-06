##############
JAI Python API
##############

*********
JAI Class
*********

.. automodule:: jai.jai
   :members:
   :undoc-members:
   :show-inheritance:

**********
Fit kwargs
**********

Parameters that should be passed as a dictionary in compliance with the
API methods. In other words, every keyword argument should be passed as if
it were in the body of a POST method.

====================
All types parameters
====================
Here are the parameters that can be used for all types of models:

* **overwrite** (*bool*) -- If setup should continue even if there's a database 
  set up with the given name. *Default is False*.

=====
Image
=====
Here are the parameters that can be used for Image models:

* **hyperparams** (*dict*) -- Image model hyperparams:  
 
  * **model_name** (*torchvision*) -- Model for image preprocessing
    {"resnet50", "resnet18", "alexnet", "squeezenet", "vgg16", "densenet", 
    "inception", "googlenet", "shufflenet", "mobilenet", "resnext50_32x4d",
    "wide_resnet50_2", "mnasnet"}. *Default is "resnet50"*.
  * **mode** -- last layer of the model, varies for each model
    {"classifier", "dense", "conv", "avgpool" or "int"}. *Default is "classifier"*.
  * **resize_H** (*int*) -- Height of image resizing, must be greater or
    equal to 224. *Default is 224*.
  * **resize_W**: (*int*) width of image resizing, must be greater or
    equal to 224. *Default is 224*.

====
Text
====
Here are the parameters that can be used for Text models:

* **hyperparams** (*dict*) -- Text model hyperparams:   
  
  * **nlp_model** (*transformers*) -- Model name for text preprocessing. *Default is "distilroberta-base"*.
  * **max_length** (*int*) -- Controls the maximum length to use by one
    of the truncation/padding parameters. *Default is 100*.

========
FastText
========
Here are the parameters that can be used for FastText models:

* **hyperparams** (*dict*) -- FastText model hyperparams: 

  * **minn** (*int*) -- Min length of char ngram. *Default is 0*.
  * **maxn** (*int*) -- Max length of char ngram. *Default is 0*.
  * **dim** (*int*) -- Final latent layer dimension. *Default is 128*.
  * **epoch** (*int*) -- Number of epochs. *Default is 10*.
  * **model** (*str*) -- SelfSupervised fasttext model {"cbow", "skipgram"},
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
  
========
TextEdit
========
Here are the parameters that can be used for TextEdit models:

* **hyperparams** (*dict*) -- TextEdit model hyperparams: 

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
  
==============
SelfSupervised
==============
Here are the parameters that can be used for SelfSupervised models:

* **hyperparams** (*dict*) -- SelfSupervised model hyperparams: 

  * **batch_size** (*int*) -- Batch size for training. *Default is 512*.
  * **learning_rate** (*float*) -- Initial learning rate. *Default is 0.001*.
  * **encoder_layer** (*str*) -- Structure for the encoder layer {"2L", "tabnet"}
    *Default is "2L"*.
  * **decoder_layer** (*str*) -- Structure for the decoder layer {"2L", "2L_LN", "2L_BN", "1L"}. 
    *Default is "2L"*.
  * **hidden_latent_dim** (*int*) -- Hidden layer size. *Default is 64*.
  * **dropout_rate** (*int*) -- Dropout rate for the encoder layer. *Default is 0.1*.
  * **momentum** (*int*) -- momentum param for batch norm for the encoder layer. *Default is 0.1*.
  * **pretraining_ratio** (*int*) -- rate of feature masking on self-supervised training. 
    *Default is 0.1*.
  * **noise_level** (*float*) -- noise level on masking process, if 0 then data is masked else noise is added. *Default is 0*.
  * **check_val_every_n_epoch** (*int*) -- number of epochs to check the validation set. *Default is 1*.
  * **gradient_clip_val** (*float*) -- The value at which to clip gradients. *Default is 0*.
  * **gradient_clip_algorithm** (*str*) -- The gradient clipping algorithm to use {"norm", "value"}. *Default is "norm"*.
  * **min_epochs** (*int*) -- Force training for at least these many epochs. *Default is 15*.
  * **max_epochs** (*int*) -- Stop training once this number of epochs is reached. *Default is 500*.
  * **patience** (*int*) -- Number of validation checks with no improvement after which training will be stopped.
    If check_val_every_n_epoch is 2 and patience is 10, it means 20 epochs without improvement will stop training. *Default is 10*.    
  * **min_delta** (*float*) -- Minimum change in the monitored quantity (loss) to qualify as an improvement,
    i.e. an absolute change of less than min_delta, will count as no improvement. *Default is 1e-5*.
  * **random_seed** (*int*) -- Random seed. *Default is 42*.
  * **stochastic_weight_avg** (*bool*) -- stochastic weight avgeraging. *Default is False*.
  * **pruning_method** (*str*) -- name of any torch.nn.utils.prune function. *Default is l1_unstructured*.
  * **pruning_amount** (*float*) -- quantity of parameters to prune. 
    If float, should be between 0.0 and 1.0 and represent the fraction of parameters to prune.
    If int, it represents the absolute number of parameters to prune. *Default is 0*.
  * **training_type** (*str*) -- type of SelfSupervised model {"contrastive", "reconstruction"}. *Default is contrastive*.

* **num_process** (*dict*) -- Parameters defining how numeric values will be processed.
   
  * **embedding_dim** (*int*) -- Initial embedding dimension. If set to 0 then 
    no embedding is made before the encoder. *Default is 8*.
  * **scaler** (*sklearn*) -- Scaler for numeric values {"maxabs", "minmax", "normalizer", 
    "quantile", "robust", "standard"}. *Default is "standard"*
  * **fill_value** (*number*) -- Fill value for missing values. *Default is 0*.

* **cat_process** (*dict*) -- Parameters defining how categorical values will be processed.
   
  * **embedding_dim** (*int*) -- Initial embedding dimension. If set to 0 then 
    no embedding is made before the encoder. *Default is 32*.
  * **fill_value** (*str*) -- Fill value for missing values. *Default is "_other"*.
  * **min_freq** (*str*) -- Number of times a category has to occur to be valid,
    otherwise we substitute by fill_value. *Default is 3*.

* **datetime_process** (*dict*) -- Parameters defining how datetime values will be processed.
    
  * **embedding_dim** (*int*) -- Initial embedding dimension. *Default is 32*.

* **features** (*dict*) -- Alternative to specify the preprocessing for each feature rather than for each type of feature. Unspecified columns will 
  follow the num_process, cat_process or datetime_process:
  
  * **dtype** (*str*) -- (*required*) possible values are "int32", "int64", "float32", "float64", "category" or "datetime"
  * **embedding_dim** (*int*) -- Initial embedding dimension. *Default is 128*.
  * **fill_value** (*str, float or int*) -- (*required*) value to fill nans for dtype numerical/category.
  * **scaler** (*str*) -- (*required*) scaling for dtype numerical features.
  * **min_freq** (*int*) -- categories with less than that will be discarted, for dtype category. *Default is 0*.

* **pretrained_bases** (*list of dicts*) -- Related already processed data that will be used in the setup of this new one. If a column has id values that 
  represent a database already preprocessed, then:

  * **db_parent** (*str*) -- (*required*) Name of the preprocessed database.
  * **id_name** (*str*) -- (*required*) Name of the column with the id values in the current table.
  * **embedding_dim** (*int*) -- Initial embedding dimension. *Default is 128*.
  
* **split** (*dict*) -- How data will be split in the training process.
   
  * **type** (*str*) -- How to split the data in train and test {sequential, sequential_exclusive, random, stratified}. *Default is "random"*.
  * **split_column** (*str*) -- (*Mandatory when type is stratified*) Name of column as reference for the split. *Default is ""*.
  * **test_size** (*float*) -- Size of test for the split. *Default is 0.2*.
  * **gap** (*int*) -- when type is sequential, Number of samples to exclude from the end of each train set before the test set. *Default is 0*

==========
Supervised
==========
Here are the parameters that can be used for Supervised models:

* **hyperparams** (*dict*) -- Supervised model hyperparams: 

  * **batch_size** (*int*) -- Batch size for training. *Default is 512*.
  * **learning_rate** (*float*) -- Initial learning rate. *Default is 0.001*.
  * **encoder_layer** (*str*) -- Structure for the encoder layer {"2L", "tabnet"}
    *Default is "2L"*.
  * **decoder_layer** (*str*) -- Structure for the decoder layer {"2L", "2L_LN", "2L_BN", "1L"}. 
    *Default is "2L"*.
  * **hidden_latent_dim** (*int*) -- Hidden layer size. *Default is 64*.
  * **dropout_rate** (*int*) -- Dropout rate for the encoder layer. *Default is 0.1*.
  * **momentum** (*int*) -- momentum param for batch norm for the encoder layer. *Default is 0.001*.
  * **pretraining_ratio** (*int*) -- rate of feature masking on self-supervised training. 
    *Default is 0.1*.
  * **noise_level** (*float*) -- noise level on masking process, if 0 then data is masked else noise is added. *Default is 0*.
  * **check_val_every_n_epoch** (*int*) -- number of epochs to check the validation set. *Default is 1*.
  * **gradient_clip_val** (*float*) -- The value at which to clip gradients. *Default is 0*.
  * **gradient_clip_algorithm** (*str*) -- The gradient clipping algorithm to use {"norm", "value"}. *Default is norm*.
  * **min_epochs** (*int*) -- Force training for at least these many epochs. *Default is 15*.
  * **max_epochs** (*int*) -- Stop training once this number of epochs is reached. *Default is 500*.
  * **patience** (*int*) -- Number of validation checks with no improvement after which training will be stopped.
    If check_val_every_n_epoch is 2 and patience is 10, it means 20 epochs without improvement will stop training. *Default is 10*.    
  * **min_delta** (*float*) -- Minimum change in the monitored quantity (loss) to qualify as an improvement,
    i.e. an absolute change of less than min_delta, will count as no improvement. *Default is 1e-5*.
  * **random_seed** (*int*) -- Random seed. *Default is 42*.
  * **stochastic_weight_avg** (*bool*) -- stochastic weight avgeraging. *Default is False*.
  * **pruning_method** (*str*) -- name of any torch.nn.utils.prune function. *Default is l1_unstructured*.
  * **pruning_amount** (*float*) -- quantity of parameters to prune. 
    If float, should be between 0.0 and 1.0 and represent the fraction of parameters to prune.
    If int, it represents the absolute number of parameters to prune. *Default is 0*.


* **num_process** (*dict*) -- Parameters defining how numeric values will be processed.
   
  * **embedding_dim** (*int*) -- Initial embedding dimension. If set to 0 then 
    no embedding is made before the encoder. *Default is 8*.
  * **scaler** (*sklearn*) -- Scaler for numeric values {"maxabs", "minmax", "normalizer", 
    "quantile", "robust", "standard"}. *Default is "standard"*
  * **fill_value** (*number*) -- Fill value for missing values. *Default is 0*.

* **cat_process** (*dict*) -- Parameters defining how categorical values will be processed.
   
  * **embedding_dim** (*int*) -- Initial embedding dimension. If set to 0 then 
    no embedding is made before the encoder. *Default is 32*.
  * **fill_value** (*str*) -- Fill value for missing values. *Default is "_other"*.
  * **min_freq** (*str*) -- Number of times a category has to occur to be valid,
    otherwise we substitute by fill_value. *Default is 3*.

* **datetime_process** (*dict*) -- Parameters defining how datetime values will be processed.
    
  * **embedding_dim** (*int*) -- Initial embedding dimension. *Default is 32*.

* **features** (*dict*) -- Alternative to specify the preprocessing for each feature rather than for each type of feature. Unspecified columns will 
  follow the num_process, cat_process or datetime_process:
  
  * **dtype** (*str*) -- (*required*) possible values are "int32", "int64", "float32", "float64", "category" or "datetime"
  * **embedding_dim** (*int*) -- Initial embedding dimension. *Default is 128*.
  * **fill_value** (*str, float or int*) -- (*required*) value to fill nans for dtype numerical/category.
  * **scaler** (*str*) -- (*required*) scaling for dtype numerical features.
  * **min_freq** (*int*) -- categories with less than that will be discarted, for dtype category. *Default is 0*.

* **pretrained_bases** (*list of dicts*) -- Related already processed data that will be used in the setup of this new one. If a column has id values that 
  represent a database already preprocessed, then:

  * **db_parent** (*str*) -- (*required*) Name of the preprocessed database.
  * **id_name** (*str*) -- (*required*) Name of the column with the id values in the current table.
  * **embedding_dim** (*int*) -- Initial embedding dimension. *Default is 128*.
  * **aggregation_method** (*str*) -- If value is a list of ids, defines how to aggregate the vectors {"sum", "mean", "max"}. *Default is sum*

* **label** (*dict*) -- Label of each ID.

  * **task** (*str*) -- (*required*) Supervised task type {"classification", "metric_classification", "regression", 
    "quantile_regression"}.
  * **label_name** (*str*) -- (*required*) Column name with target values.
  * **regression_scaler** (*str*) -- type of scaling to apply to label on regression models {"None", "log1p", "standard", "log1p+standard"}. *Default is None*.
  * **quantiles** (*list of floats*) -- quantiles for quantile_regression. *Default is [0.1, 0.5, 0.9]*.

* **split** (*dict*) -- How data will be split in the training process.
   
  * **type** (*str*) -- How to split the data in train and test {random, stratified}. *Default is "random"*.
  * **split_column** (*str*) -- (*Mandatory when type is stratified*) Name of column as reference for the split. *Default is ""*.
  * **test_size** (*float*) -- Size of test for the split. *Default is 0.2*.
  * **gap** (*int*) -- when type is sequential, Number of samples to exclude from the end of each train set before the test set. *Default is 0*
