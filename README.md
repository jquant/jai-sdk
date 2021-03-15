# Jai SDK - Trust your data
[![PyPI Latest Release](https://img.shields.io/pypi/v/jai-sdk.svg)](https://pypi.org/project/jai-sdk/)
[![Python Version](https://img.shields.io/badge/python-3.7%20%7C%203.8-blue)](https://img.shields.io/badge/python-3.7%20%7C%203.8-blue)
[![Documentation Status](https://readthedocs.org/projects/jai-sdk/badge/?version=latest)](https://jai-sdk.readthedocs.io/en/latest/?badge=latest)
[![codecov](https://codecov.io/gh/jquant/jai-sdk/branch/main/graph/badge.svg)](https://codecov.io/gh/jquant/jai-sdk)
[![License](https://img.shields.io/pypi/l/jai-sdk.svg)](https://github.com/jquant/jai-sdk/blob/main/LICENSE)
[![Code style: yapf](https://img.shields.io/badge/code%20style-yapf-blue)](https://github.com/google/yapf)

# Installation
The source code is currently hosted on GitHub at: [https://github.com/jquant/jai-sdk](https://github.com/jquant/jai-sdk)

Installing jai-sdk using `pip`:

```sh
pip install jai-sdk
```

# Get your Auth Key
First, you'll need and Authorization key to use the backend API.

To get an Trial version API using the sdk, fill the values with your information:

```python
from jai import Jai

r = Jai.get_auth_key(email=EMAIL, firstName=FIRSTNAME, lastName=LASTNAME)
```

If the response code is 201, then you should be receiving an email with your Auth Key.


# Get Started
If you already have an Auth Key, then you can use the sdk:
```python
from jai import Jai
j = Jai(AUTH_KEY)
```

## Setting up your databases

*All data should be in pandas.DataFrame or pandas.Series format*

Aplication using the NLP FastText model
```python
### fasttext implementation
# save this if you want to work in the same database later
name = 'text_data'

### Insert data and train the FastText model
# data can be a list of texts, pandas Series or DataFrame.
# if data is a list, then the ids will be set with range(len(data_list))
# if data is a pandas type, then the ids will be the index values.
# heads-up: index values must not contain duplicates.
j.setup(name, data, db_type='FastText')
```

Aplication using the NLP BERT model
```python
### BERT implementation
# generate a random name for identification of the base; it can be a user input
name = j.generate_name(20, prefix='sdk_', suffix='_text')

# this time we choose db_type="Text", applying the pre-trained BERT model
j.setup(name, data, db_type='Text', batch_size=1024)
```

## Checking database

Here are some methods to check your databases.

The name of your database should appear in:

```python
>>> j.names
['jai_database', 'jai_unsupervised', 'jai_supervised']
```

or you can check if a given database name is valid:

```python
>>> j.is_valid(name)
True
```


You can also check the types for each of your databases with:

```python
>>> j.info
                        db_name       db_type
0                  jai_database          Text
1              jai_unsupervised  Unsupervised
2                jai_supervised    Supervised
```

If you want to check which ids are in your database:

```python
>>> j.ids(name)
['1000 items from 0 to 999']
```

## Similarity
After you're done setting up your database, you perform similarity searches:

- Using the indexes of the input data
```python
# Find the 5 most similar values for ids 0 and 1
results = j.similar(name, [0, 1], top_k=5)

# Find the 20 most similar values for every id from [0, 99]
ids = list(range(100))
results = j.similar(name, ids, top_k=20)

# Find the 100 most similar values for every input value
results = j.similar(name, data.index, top_k=100, batch_size=1024)
```

- Using new data to be processed
*All data should be in pandas.DataFrame or pandas.Series format*
```python
# Find the 100 most similar values for every new_data
results = j.similar(name, new_data, top_k=100, batch_size=1024)
```

The output will be a list of dictionaries with ("query_id") being the id of the value you want to find similars and ("results") a list with `top_k` dictionaries with the "id" and the "distance" between "query_id" and "id".
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
j.delete_raw_data(name)
```

If you no longer need the model or anything else related to your database:
``` python
j.delete_database(name)
```
