# mycelia-sdk
Mycelia SDKs

# Examples
Instanciating your base class
```python
mycelia = Mycelia(AUTH_KEY)
```

## Setting up your databases

Aplication using the model NLP FastText
```python
### fasttext implementation
# save this if you wish to work in the same database later
name = 'text_data'

### Data insertion and train the unsupervised FastText model
# data can be a list of texts, pandas Series or DataFrame.
# if data is a list, then ids will be set with range(len(data_list))
# if data is a pandas type, then the ids will be the index values, index must not contain duplicated values
mycelia.setup(name, data, db_type='FastText')

# wait for the train to finish
mycelia.wait_setup(10)
```

Aplication using the model NLP BERT
```python
### bert implementation
# generate a random name for identification of the base, can be a user input
name = mycelia.generate_name(20, prefix='sdk_', suffix='_text')

# this time we choose db_type="Text", applying the pre-trained BERT model
mycelia.setup(name, data, db_type='Text', batch_size=1024)
mycelia.wait_setup(10)
```

## Checking database

Here are some methods to check your databases:

The name of your database should appear in:

```python
>>> mycelia.names
['mycelia_database', 'mycelia_unsupervised', 'mycelia_supervised']
```

or you can check if it's valid:

```python
>>> mycelia.is_valid(name)
{'value': True, 'message': 'mycelia_database is a valid database name.'}
```


and you can check the databases types for each of your databases with:

```python
>>> mycelia.info
                        db_name       db_type
0   		   mycelia_database          Text
1          mycelia_unsupervised  Unsupervised
2            mycelia_supervised    Supervised
```

if you want to check which ids are in your database:

```python
>>> mycelia.ids(name)
['1000 items from 0 to 999']
```

## Similarity
After you're done with setting up your database, you can find similarity:

- Using the indexes of the inputed data
```python
# Find the 5 most similar values for the ids 0 and 1
results = mycelia.similar(name, [0, 1], top_k=5)

# Find the 20 most similar values for every id in 0 to 100
ids = list(range(100))
results = mycelia.similar(name, ids, top_k=20)

# Find the 100 most similar values for every inputed value
results = mycelia.similar(name, data.index, top_k=100, batch_size=1024)
```

- Using new data to be processed
```python
# Find the 100 most similar values for every new_data
results = mycelia.similar(name, new_data, top_k=100, batch_size=1024)
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

After you're done with the model setup, you can delete your raw data
```python
# Delete the raw data inputed as it won't be needed anymore
mycelia.delete_raw_data(name)
```

If you want to keep the environment clean
``` python
mycelia.delete_database(name)
```
