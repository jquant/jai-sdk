.. jai-sdk documentation master file, created by
   sphinx-quickstart on Thu Feb 25 13:40:47 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

##########################
JAI-SDK - Trust your data!
##########################

Great to see you here!

JAI is a productivity-oriented, vector-based ML Platform, served via REST API or Python SDK (other languages soon!).

3-minute JAI - [Full notebook on Colab]:

* Install JAI via pip
  
   ::

      pip install jai-sdk --user
      
* Import JAI and Generate your Community Auth Key (free forever)

   ::

      from jai import Jai as j
      j.generate_auth_key('my-email@email.com')
      201

Please note that yout Auth Key will be sent to your e-mail, so please make sure to use a valid address and check your spam folder.

* Use JAI to transform the Boston Housing dataset into vectors using SelfSupervised Learning

   ::

      from sklearn.datasets import load_boston
      
      #load dataset
      boston = load_boston()

      #note that we are not loading the target column "PRICE"
      data = pd.DataFrame(boston.data, columns=boston.feature_names)
      
      #send data to JAI for feature extraction
      j.setup(
         #JAI collection name
         name='boston',

         #data to be processed - a Pandas DataFrame is expected
         data=data,

         #collection type
         db_type='SelfSupervised',

         #verbose 2 -> shows the loss graph at the end of training
         verbose=2,

         #let's set some hyperparams!
         hyperparams={
            'learning_rate': 3e-4,
            'pretraining_ratio':0.8
         }
      )

   Output:

   ::

      Insert Data: 100%|██████████| 1/1 [00:01<00:00,  1.39s/it]
      Recognized setup args:
      hyperparams: {'learning_rate': 0.0003, 'pretraining_ratio': 0.8}
      JAI is working:  38%|███▊      |6/16
      [boston] Training:   0%|          | 0/500 [00:00<?, ?it/s]ATraining might not take 500 steps due to early stopping criteria.
      
      [boston] Training:   1%|          | 6/500 [00:02<02:45,  2.99it/s]A
      [boston] Training:   2%|▏         | 11/500 [00:04<03:22,  2.41it/s]A
      [boston] Training:   3%|▎         | 14/500 [00:06<04:08,  1.96it/s]A
      [boston] Training:   3%|▎         | 17/500 [00:08<04:20,  1.85it/s]A
      [boston] Training:   5%|▍         | 23/500 [00:11<03:57,  2.01it/s]A
      [boston] Training:   6%|▌         | 28/500 [00:13<03:42,  2.12it/s]A
      [boston] Training:   6%|▋         | 32/500 [00:15<03:48,  2.05it/s]A
      [boston] Training: 100%|██████████| 500/500 [00:17<00:00, 74.20it/s]A
      JAI is working:  56%|█████▋    |9/16
      Done training.
      JAI is working: 100%|██████████|16/16

* Now that our Boston Housing data in a JAI collection, we can perform Similarity Search, i.e. find similar houses, very easily:

   ::

      #every JAI collection can be queried using j.similar()
      ans = j.similar(
            #collection to be queried
            name='boston',
            #let's find houses that are similar to ids 1 and 10
            data=[1, 10]
      )

   Output:

   ::

      Similar: 100%|██████████| 1/1 [00:01<00:00,  1.36s/it]

   And now the variable ans holds a JSON:

   ::

      [{'query_id': 1,
      'results': [{'id': 1, 'distance': 0.0},
      {'id': 96, 'distance': 0.012930447235703468},
      {'id': 235, 'distance': 0.02305753342807293},
      {'id': 176, 'distance': 0.02424568682909012},
      {'id': 90, 'distance': 0.025710342451930046}]},
      
      {'query_id': 10,
      'results': [{'id': 10, 'distance': 0.0},
      {'id': 7, 'distance': 0.0065054153092205524},
      {'id': 9, 'distance': 0.020906779915094376},
      {'id': 11, 'distance': 0.04773647338151932},
      {'id': 6, 'distance': 0.09080290794372559}]}]

   And by indexing it back to the original dataframe id's, yields:

   ::

      #id 1
      data.loc[pd.DataFrame(ans[0]['results']).id]


   ====  =======  ====  =======  ======  =====  =====  =====  ======  =====  =====  =========  ======  =======
     ..     CRIM    ZN    INDUS    CHAS    NOX     RM    AGE     DIS    RAD    TAX    PTRATIO       B    LSTAT
   ====  =======  ====  =======  ======  =====  =====  =====  ======  =====  =====  =========  ======  =======
      1  0.02731     0     7.07       0  0.469  6.421   78.9  4.9671      2    242       17.8  396.9      9.14
     96  0.11504     0     2.89       0  0.445  6.163   69.6  3.4952      2    276       18    391.83    11.34
    235  0.33045     0     6.2        0  0.507  6.086   61.5  3.6519      8    307       17.4  376.75    10.88
    176  0.07022     0     4.05       0  0.51   6.02    47.2  3.5549      5    296       16.6  393.23    10.11
     90  0.04684     0     3.41       0  0.489  6.417   66.1  3.0923      2    270       17.8  392.18     8.81
   ====  =======  ====  =======  ======  =====  =====  =====  ======  =====  =====  =========  ======  =======

   ::

      #id 10
      data.loc[pd.DataFrame(ans[1]['results']).id]


   ====  =======  ====  =======  ======  =====  =====  =====  ======  =====  =====  =========  ======  =======
     ..     CRIM    ZN    INDUS    CHAS    NOX     RM    AGE     DIS    RAD    TAX    PTRATIO       B    LSTAT
   ====  =======  ====  =======  ======  =====  =====  =====  ======  =====  =====  =========  ======  =======
     10  0.22489  12.5     7.87       0  0.524  6.377   94.3  6.3467      5    311       15.2  392.52    20.45
      7  0.14455  12.5     7.87       0  0.524  6.172   96.1  5.9505      5    311       15.2  396.9     19.15
      9  0.17004  12.5     7.87       0  0.524  6.004   85.9  6.5921      5    311       15.2  386.71    17.1
     11  0.11747  12.5     7.87       0  0.524  6.009   82.9  6.2267      5    311       15.2  396.9     13.27
      6  0.08829  12.5     7.87       0  0.524  6.012   66.6  5.5605      5    311       15.2  395.6     12.43
   ====  =======  ====  =======  ======  =====  =====  =====  ======  =====  =====  =========  ======  =======
   


.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   source/quick_start
   source/applications
   
.. toctree::
   :maxdepth: 1
   :caption: Basic Operations

   use_cases/setup_data
   use_cases/check_database
   use_cases/similarity
   use_cases/predict
   use_cases/remove_data

.. toctree::
   :maxdepth: 2
   :caption: API Reference
   
   source/jai
   source/auxiliar

##################
Indices and tables
##################

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
