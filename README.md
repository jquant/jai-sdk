# mycelia-sdk
Mycelia SDKs

# Examples
```python
mycelia = Mycelia(AUTH_KEY)
```

## Setting up your databases

Aplication using the model NLP FastText
```
### fasttext implementation
# generate a random name for identification of the base, can be a user input
# save this if you wish to work in the same database later
name = mycelia.generate_name(20, prefix='sdk_', suffix='_fasttext')
# Data insertion, data can be a list of texts, pandas Series or DataFrame.
# if data is a list, then ids will be set with range(len(data_list))
# if data is a pandas type, then the ids will be the index values, index must not contain duplicated values
mycelia.insert_data(name, data)
# Train the unsupervised FastText model
mycelia.setup_database(name, db_type='FastText')
mycelia.wait_setup(10)  # wait for the train to finish
```

Aplication using the model NLP BERT
```python
### bert implementation
# Same initial steps
name = mycelia.generate_name(20, prefix='sdk_', suffix='_text')
mycelia.insert_data(name, data, batch_size=1024)
# this time we choose db_type="Text", applying the pre-trained BERT model
mycelia.setup_database(name, db_type='Text')
mycelia.wait_setup(10)
```

## Similarity
After you're done with setting up your database, you can find similarity:

- Using the indexes of the inputed data
```python
# Find the 100 most similar values for every inputed value
results = mycelia.similar_list(name, data.index, top_k=100, batch_size=1024)
```

- Using new data to be processed
```python
# Find the 100 most similar values for every inputed value
results = mycelia.similar_data(name, data, top_k=100, batch_size=1024)
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
