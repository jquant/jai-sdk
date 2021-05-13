.. jai-sdk documentation master file, created by
   sphinx-quickstart on Thu Feb 25 13:40:47 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

##########################
JAI-SDK - Trust your data!
##########################

Great to see you here!

JAI is a productivity-oriented, vector-based, ML Platform, served via REST API or Python SDK (other languages coming soon).

*****************
JAI in 5 minutes:
*****************

* Install JAI via pip
  
   ::

      pip install jai-sdk --user
      
* Import JAI and Generate your Community Auth Key (free forever)

   ::

      from jai import Jai
      Jai.generate_auth_key('my-email@email.com')
      201

Please note that yout Auth Key will be sent to your e-mail, so please make sure to use a valid address and check your spam folder.

* Use your Auth Key to instantiate JAI:

   ::

      j = Jai('AUTH KEY')

* Now let's use JAI to transform the Boston Housing dataset into vectors using SelfSupervised Learning

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

* Now that our Boston Housing data is in a JAI collection, we can perform Similarity Search, i.e. find similar houses, very easily:

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

   And now the 'ans' variable holds a JSON:

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

   And by indexing it back to the original dataframe id's, we have:

   ::

      #id 1
      #List of top 5 similar houses (house 1 itself + 4)
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
      #List of top 5 similar houses (house 10 itself + 4)
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
   
* And we can also train our Supervised Model to predict house prices!
  
   ::

      ans = j.setup(
         name='boston_regression',
         verbose=2,
         data=data,
         db_type='Supervised',
         mycelia_bases=[
            {
               'db_parent':'boston',
               'id_name':'id_house'
            }
         ],
         label=
         {
            'task':'regression'
            'label_name':'PRICE'
         },
         hyperparams={'learning_rate':3e-4, 'patience':'500'},
         overwrite=True
      )

   Output:

   ::

      Insert Data: 100%|██████████| 1/1 [00:01<00:00,  1.20s/it]
      Recognized setup args:
      hyperparams: {'learning_rate': 0.0003, 'patience': '500'}
      mycelia_bases: [{'db_parent': 'boston', 'id_name': 'id_house'}]
      label: {'task': 'regression', 'label_name': 'PRICE'}
      JAI is working:  50%|█████     |9/18
      [boston_regression] Training:   0%|          | 0/500 [00:00<?, ?it/s] Training might not take 500 steps due to early stopping criteria.
      
      [boston_regression] Training:   2%|▏         | 8/500 [00:03<03:15,  2.51it/s]
      [boston_regression] Training:   2%|▏         | 11/500 [00:05<03:51,  2.11it/s]
      [boston_regression] Training:   4%|▎         | 18/500 [00:07<03:23,  2.36it/s]
      [boston_regression] Training:   6%|▌         | 29/500 [00:11<02:56,  2.66it/s]
      [boston_regression] Training:   7%|▋         | 37/500 [00:13<02:43,  2.82it/s]
      [boston_regression] Training:   9%|▉         | 45/500 [00:15<02:22,  3.20it/s]
      [boston_regression] Training:  10%|█         | 50/500 [00:17<02:24,  3.11it/s]
      [boston_regression] Training:  12%|█▏        | 58/500 [00:19<02:12,  3.34it/s]
      [boston_regression] Training:  13%|█▎        | 63/500 [00:21<02:15,  3.21it/s]
      [boston_regression] Training:  14%|█▍        | 69/500 [00:23<02:13,  3.22it/s]
      [boston_regression] Training:  15%|█▍        | 74/500 [00:25<02:19,  3.06it/s]
      [boston_regression] Training:  17%|█▋        | 84/500 [00:27<02:10,  3.20it/s]
      [boston_regression] Training:  18%|█▊        | 90/500 [00:29<02:07,  3.21it/s]
      [boston_regression] Training:  19%|█▉        | 94/500 [00:31<02:19,  2.92it/s]
      [boston_regression] Training:  20%|█▉        | 98/500 [00:33<02:26,  2.74it/s]
      [boston_regression] Training:  21%|██        | 103/500 [00:35<02:30,  2.64it/s]
      [boston_regression] Training:  23%|██▎       | 113/500 [00:38<02:10,  2.96it/s]
      [boston_regression] Training:  24%|██▍       | 119/500 [00:40<02:06,  3.02it/s]
      [boston_regression] Training:  25%|██▌       | 125/500 [00:42<01:59,  3.13it/s]
      [boston_regression] Training:  26%|██▌       | 131/500 [00:43<01:58,  3.12it/s]
      [boston_regression] Training:  27%|██▋       | 137/500 [00:45<01:53,  3.21it/s]
      [boston_regression] Training:  29%|██▊       | 143/500 [00:47<01:48,  3.30it/s]
      [boston_regression] Training:  30%|██▉       | 149/500 [00:49<01:44,  3.37it/s]
      [boston_regression] Training:  31%|███       | 155/500 [00:50<01:41,  3.39it/s]
      [boston_regression] Training:  32%|███▏      | 161/500 [00:52<01:39,  3.41it/s]
      [boston_regression] Training:  33%|███▎      | 166/500 [00:54<01:44,  3.19it/s]
      [boston_regression] Training:  35%|███▍      | 173/500 [00:56<01:38,  3.32it/s]
      [boston_regression] Training:  36%|███▌      | 179/500 [00:58<01:35,  3.36it/s]
      [boston_regression] Training:  37%|███▋      | 184/500 [00:59<01:39,  3.18it/s]
      [boston_regression] Training:  38%|███▊      | 190/500 [01:01<01:35,  3.26it/s]
      [boston_regression] Training:  39%|███▉      | 196/500 [01:03<01:33,  3.26it/s]
      [boston_regression] Training:  41%|████      | 203/500 [01:05<01:27,  3.40it/s]
      [boston_regression] Training:  42%|████▏     | 211/500 [01:07<01:24,  3.40it/s]
      [boston_regression] Training:  43%|████▎     | 216/500 [01:09<01:27,  3.24it/s]
      [boston_regression] Training:  44%|████▍     | 222/500 [01:11<01:24,  3.29it/s]
      [boston_regression] Training:  46%|████▌     | 228/500 [01:13<01:28,  3.09it/s]
      [boston_regression] Training:  47%|████▋     | 234/500 [01:15<01:26,  3.08it/s]
      [boston_regression] Training:  48%|████▊     | 239/500 [01:17<01:26,  3.01it/s]
      [boston_regression] Training:  49%|████▉     | 245/500 [01:18<01:20,  3.16it/s]
      [boston_regression] Training:  50%|█████     | 251/500 [01:20<01:17,  3.23it/s]
      [boston_regression] Training:  51%|█████     | 256/500 [01:22<01:18,  3.10it/s]
      [boston_regression] Training:  52%|█████▏    | 262/500 [01:24<01:17,  3.06it/s]
      [boston_regression] Training:  54%|█████▎    | 268/500 [01:26<01:13,  3.14it/s]
      [boston_regression] Training:  55%|█████▍    | 273/500 [01:28<01:19,  2.87it/s]
      [boston_regression] Training:  56%|█████▌    | 278/500 [01:30<01:17,  2.86it/s]
      [boston_regression] Training:  57%|█████▋    | 284/500 [01:32<01:15,  2.86it/s]
      [boston_regression] Training:  58%|█████▊    | 289/500 [01:34<01:17,  2.73it/s]
      [boston_regression] Training:  59%|█████▊    | 293/500 [01:36<01:24,  2.44it/s]
      [boston_regression] Training:  59%|█████▉    | 297/500 [01:38<01:31,  2.23it/s]
      [boston_regression] Training:  60%|██████    | 300/500 [01:40<01:35,  2.09it/s]
      [boston_regression] Training:  61%|██████    | 304/500 [01:42<01:37,  2.00it/s]
      [boston_regression] Training:  62%|██████▏   | 309/500 [01:45<01:32,  2.06it/s]
      [boston_regression] Training:  63%|██████▎   | 317/500 [01:48<01:20,  2.26it/s]
      [boston_regression] Training:  65%|██████▍   | 324/500 [01:50<01:10,  2.49it/s]
      [boston_regression] Training:  66%|██████▌   | 329/500 [01:52<01:06,  2.58it/s]
      [boston_regression] Training:  67%|██████▋   | 335/500 [01:54<01:00,  2.74it/s]
      [boston_regression] Training:  68%|██████▊   | 341/500 [01:56<00:56,  2.83it/s]
      [boston_regression] Training:  69%|██████▉   | 347/500 [01:57<00:51,  2.99it/s]
      [boston_regression] Training:  71%|███████   | 353/500 [01:59<00:47,  3.07it/s]
      [boston_regression] Training:  72%|███████▏  | 361/500 [02:02<00:45,  3.02it/s]
      [boston_regression] Training:  74%|███████▍  | 370/500 [02:05<00:41,  3.13it/s]
      [boston_regression] Training:  75%|███████▌  | 375/500 [02:06<00:40,  3.08it/s]
      [boston_regression] Training:  76%|███████▌  | 380/500 [02:08<00:41,  2.89it/s]
      [boston_regression] Training:  77%|███████▋  | 386/500 [02:10<00:38,  2.97it/s]
      [boston_regression] Training:  78%|███████▊  | 391/500 [02:12<00:37,  2.93it/s]
      [boston_regression] Training:  79%|███████▉  | 397/500 [02:14<00:33,  3.04it/s]
      [boston_regression] Training:  80%|████████  | 402/500 [02:16<00:33,  2.96it/s]
      [boston_regression] Training:  82%|████████▏ | 408/500 [02:18<00:30,  2.97it/s]
      [boston_regression] Training:  83%|████████▎ | 414/500 [02:19<00:27,  3.10it/s]
      [boston_regression] Training:  84%|████████▍ | 419/500 [02:21<00:26,  3.02it/s]
      [boston_regression] Training:  85%|████████▍ | 424/500 [02:24<00:28,  2.67it/s]
      [boston_regression] Training:  85%|████████▌ | 426/500 [02:26<00:33,  2.18it/s]
      [boston_regression] Training:  86%|████████▌ | 431/500 [02:27<00:29,  2.30it/s]
      [boston_regression] Training:  87%|████████▋ | 435/500 [02:29<00:29,  2.23it/s]
      [boston_regression] Training:  88%|████████▊ | 438/500 [02:31<00:30,  2.02it/s]
      [boston_regression] Training:  89%|████████▉ | 444/500 [02:33<00:23,  2.35it/s]
      [boston_regression] Training:  90%|████████▉ | 449/500 [02:35<00:20,  2.48it/s]
      [boston_regression] Training:  91%|█████████ | 455/500 [02:37<00:16,  2.74it/s]
      [boston_regression] Training:  92%|█████████▏| 460/500 [02:39<00:15,  2.56it/s]
      [boston_regression] Training:  93%|█████████▎| 465/500 [02:41<00:13,  2.58it/s]
      [boston_regression] Training:  94%|█████████▍| 470/500 [02:43<00:11,  2.58it/s]
      [boston_regression] Training:  95%|█████████▌| 475/500 [02:45<00:10,  2.48it/s]
      [boston_regression] Training:  96%|█████████▌| 478/500 [02:47<00:10,  2.10it/s]
      [boston_regression] Training:  96%|█████████▌| 481/500 [02:50<00:10,  1.84it/s]
      [boston_regression] Training:  97%|█████████▋| 486/500 [02:51<00:06,  2.09it/s]
      [boston_regression] Training:  98%|█████████▊| 492/500 [02:54<00:03,  2.22it/s]
      [boston_regression] Training: 100%|█████████▉| 498/500 [02:56<00:00,  2.53it/s]
      [boston_regression] Training: 100%|██████████| 500/500 [02:57<00:00,  2.17it/s]
                                                                                    
      Done training.
      JAI is working: 100%|██████████|18/18

      Metrics Regression:
      MAE: 1.818383812904358
      MSE: 7.095381736755371

      Best model at epoch: 274 val_loss: 0.07

   BTW: that's some very competitive evaluation metrics!

* Now that our Supervised Boston Housing Model is also JAI collection, we can perform Similarity Search, i.e. find similar houses - **also according to the supervised label**, very easily:

   ::

      #every JAI collection can be queried using j.similar()
      ans = j.similar(
            #collection to be queried
            name='boston_regression',
            #let's find houses that are similar to ids 1 and 10
            data=[1, 10]
      )

   Output:

   ::

      Similar: 100%|██████████| 1/1 [00:01<00:00,  1.36s/it]

   And now the 'ans' variable holds a JSON:

   ::

      [{'query_id': 1,
      'results': [{'id': 1, 'distance': 0.0},
      {'id': 91, 'distance': 0.017999378964304924},
      {'id': 94, 'distance': 0.02219889685511589},
      {'id': 96, 'distance': 0.03483652323484421},
      {'id': 90, 'distance': 0.050415001809597015}]},

      {'query_id': 10,
      'results': [{'id': 10, 'distance': 0.0},
      {'id': 7, 'distance': 0.024717235937714577},
      {'id': 209, 'distance': 0.05477815866470337},
      {'id': 211, 'distance': 0.056917279958724976},
      {'id': 9, 'distance': 0.05909169092774391}]}]

   And by indexing it back to the original dataframe id's, we have:

   ::

      #id 1
      #List of top 5 similar houses (house 1 itself + 4)
      data.loc[pd.DataFrame(ans[0]['results']).id]


   ====  =======  ====  =======  ======  =====  =====  =====  ======  =====  =====  =========  ======  =======  ==========  =======
     ..     CRIM    ZN    INDUS    CHAS    NOX     RM    AGE     DIS    RAD    TAX    PTRATIO       B    LSTAT    id_house    PRICE
   ====  =======  ====  =======  ======  =====  =====  =====  ======  =====  =====  =========  ======  =======  ==========  =======
      1  0.02731     0     7.07       0  0.469  6.421   78.9  4.9671      2    242       17.8  396.9      9.14           1     21.6
     91  0.03932     0     3.41       0  0.489  6.405   73.9  3.0921      2    270       17.8  393.55     8.2           91     22
     94  0.04294    28    15.04       0  0.464  6.249   77.3  3.615       4    270       18.2  396.9     10.59          94     20.6
     96  0.11504     0     2.89       0  0.445  6.163   69.6  3.4952      2    276       18    391.83    11.34          96     21.4
     90  0.04684     0     3.41       0  0.489  6.417   66.1  3.0923      2    270       17.8  392.18     8.81          90     22.6
   ====  =======  ====  =======  ======  =====  =====  =====  ======  =====  =====  =========  ======  =======  ==========  =======

   ::

      #id 10
      #List of top 5 similar houses (house 10 itself + 4)
      data.loc[pd.DataFrame(ans[1]['results']).id]


   ====  =======  ====  =======  ======  =====  =====  =====  ======  =====  =====  =========  ======  =======  ==========  =======
     ..     CRIM    ZN    INDUS    CHAS    NOX     RM    AGE     DIS    RAD    TAX    PTRATIO       B    LSTAT    id_house    PRICE
   ====  =======  ====  =======  ======  =====  =====  =====  ======  =====  =====  =========  ======  =======  ==========  =======
     10  0.22489  12.5     7.87       0  0.524  6.377   94.3  6.3467      5    311       15.2  392.52    20.45          10     15
      7  0.14455  12.5     7.87       0  0.524  6.172   96.1  5.9505      5    311       15.2  396.9     19.15           7     27.1
    209  0.43571   0      10.59       1  0.489  5.344  100    3.875       4    277       18.6  396.9     23.09         209     20
    211  0.37578   0      10.59       1  0.489  5.404   88.6  3.665       4    277       18.6  395.24    23.98         211     19.3
      9  0.17004  12.5     7.87       0  0.524  6.004   85.9  6.5921      5    311       15.2  386.71    17.1            9     18.9
   ====  =======  ====  =======  ======  =====  =====  =====  ======  =====  =====  =========  ======  =======  ==========  =======

* We can also, of course, perform inference on our model:

   ::

      #every JAI Supervised collection can be used for inference using j.predict()
      ans = j.predict(
         #collection to be queried
         name='boston_regression',
         #let's get prices for the first five houses in the dataset, using their ids
         #also we are dropping the label, as it is not a feature
         data=data.head().drop('PRICE',axis=1)
      )

   Output:

   ::

      Predict: 100%|██████████| 1/1 [00:01<00:00,  1.59s/it]

   And now the 'ans' variable holds a JSON:

   ::

      [{'id': 0, 'predict': [24.70072364807129]},
      {'id': 1, 'predict': [21.706649780273438]},
      {'id': 2, 'predict': [31.775901794433594]},
      {'id': 3, 'predict': [34.41084289550781]},
      {'id': 4, 'predict': [34.54452896118164]}]

   And by indexing it back to the original dataframe id's, we have:

   ::

      #id 1
      #List of top 5 similar houses (house 1 itself + 4)
      predict_df = pd.DataFrame(ans)
      predict_df = predict_df.set_index('id')
      predict_df.loc[:,'predict'] = predict_df['predict'].apply(lambda x: x[0])
      predict_df['true'] = data['PRICE']


   ====  =========  ======
   ..    predict    true
   ====  =========  ======
      0    24.7007    24
      1    21.7066    21.6
      2    31.7759    34.7
      3    34.4108    33.4
      4    34.5445    36.2
   ====  =========  ======

* Everything in JAI is always instantly deployed and available through REST API.

   ::

      #Similarity Search via REST API

      import requests
      import json

      header={'Auth': '9c290424179e4f7485c182622ae82490'}
      db_name = 'boston'

      url_similar = f"https://mycelia.azure-api.net/similar/id/{db_name}"
      body = json.dumps([1,10])

      ans = requests.put(url_similar, data=body, headers=header)

   Output - ans.json():

   ::

      {
         'similarity': [

            {'query_id': 1,
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
            {'id': 6, 'distance': 0.09080290794372559}]}

         ]
      }

   ::

      #Model Inference via REST API

      import requests
      import json
      
      header={'Auth': '9c290424179e4f7485c182622ae82490'}
      db_name = 'boston_regression'
      url_predict = f"https://mycelia.azure-api.net/predict/{db_name}"
      body = data.reset_index().rename(columns={'index':'id'}).head().drop('PRICE',axis=1).to_json(orient='records')
      
      ans = requests.put(url_predict, data=body, headers=header)

   Output - ans.json():

   ::

      [{'id': 0, 'predict': [24.70072364807129]},
      {'id': 1, 'predict': [21.706649780273438]},
      {'id': 2, 'predict': [31.775901794433594]},
      {'id': 3, 'predict': [34.41084289550781]},
      {'id': 4, 'predict': [34.54452896118164]}]

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
