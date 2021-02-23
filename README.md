# jai-sdk
jai SDKs

# Examples
Instanciating your base class
```python
from jai import Jai
j = Jai(AUTH_KEY)
```

## Setting up your databases

*All data should be an pandas.DataFrame or pandas.Series*

Aplication using the model NLP FastText
```python
### fasttext implementation
# save this if you wish to work in the same database later
name = 'text_data'

### Data insertion and train the unsupervised FastText model
# data can be a list of texts, pandas Series or DataFrame.
# if data is a list, then ids will be set with range(len(data_list))
# if data is a pandas type, then the ids will be the index values, index must not contain duplicated values
j.setup(name, data, db_type='FastText')

# wait for the train to finish
j.wait_setup(name, 10)
```

Aplication using the model NLP BERT
```python
### bert implementation
# generate a random name for identification of the base, can be a user input
name = j.generate_name(20, prefix='sdk_', suffix='_text')

# this time we choose db_type="Text", applying the pre-trained BERT model
j.setup(name, data, db_type='Text', batch_size=1024)
j.wait_setup(name, 10)
```

## Checking database

Here are some methods to check your databases:

The name of your database should appear in:

```python
>>> j.names
['jai_database', 'jai_unsupervised', 'jai_supervised']
```

or you can check if it's valid:

```python
>>> j.is_valid(name)
True
```


and you can check the databases types for each of your databases with:

```python
>>> j.info
                        db_name       db_type
0                  jai_database          Text
1              jai_unsupervised  Unsupervised
2                jai_supervised    Supervised
```

if you want to check which ids are in your database:

```python
>>> j.ids(name)
['1000 items from 0 to 999']
```

## Similarity
After you're done with setting up your database, you can find similarity:

- Using the indexes of the inputed data
```python
# Find the 5 most similar values for the ids 0 and 1
results = j.similar(name, [0, 1], top_k=5)

# Find the 20 most similar values for every id in 0 to 100
ids = list(range(100))
results = j.similar(name, ids, top_k=20)

# Find the 100 most similar values for every inputed value
results = j.similar(name, data.index, top_k=100, batch_size=1024)
```

- Using new data to be processed
*All data should be an pandas.DataFrame or pandas.Series*
```python
# Find the 100 most similar values for every new_data
results = j.similar(name, new_data, top_k=100, batch_size=1024)
```

The output will be a list of dictionaries with ("query_id") the id of the value you want to find similars and ("results") a list with `top_k` dictionaries with the "id" and the "distance" between "query_id" and "id".
```
[
  {
    'query_id': 0,
    'results':
    [
      {'id': 0, 'distance': 0.0},
      {'id': 3836, 'distance': 2.298321008682251},
      {'id': 9193, 'distance': 2.545339584350586},
      {'id': 832, 'distance': 2.5819168090820312},
      {'id': 6162, 'distance': 2.638622283935547},
      ...
    ]
  },
  ...,
  {
    'query_id': 9,
    'results':
    [
      {'id': 9, 'distance': 0.0},
      {'id': 54, 'distance': 5.262974262237549},
      {'id': 101, 'distance': 5.634262561798096},
      ...
    ]
  },
  ...
]
```

# Removing data

After you're done with the model setup, you can delete the inserted raw data
```python
# Delete the raw data inputed as it won't be needed anymore
j.delete_raw_data(name)
```

If you want to keep the environment clean
``` python
j.delete_database(name)
```
