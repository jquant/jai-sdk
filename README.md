# Jai SDK - Trust your data

[![PyPI Latest Release](https://img.shields.io/pypi/v/jai-sdk.svg)](https://pypi.org/project/jai-sdk/)
[![Python Version](https://img.shields.io/badge/python-3.7%20%7C%203.8-blue)](https://img.shields.io/badge/python-3.7%20%7C%203.8-blue)
[![Documentation Status](https://readthedocs.org/projects/jai-sdk/badge/?version=latest)](https://jai-sdk.readthedocs.io/en/latest/?badge=latest)
[![codecov](https://codecov.io/gh/jquant/jai-sdk/branch/main/graph/badge.svg)](https://codecov.io/gh/jquant/jai-sdk)
[![License](https://img.shields.io/pypi/l/jai-sdk.svg)](https://github.com/jquant/jai-sdk/blob/main/LICENSE)
[![Code style: yapf](https://img.shields.io/badge/code%20style-yapf-blue)](https://github.com/google/yapf)
[![Downloads](https://pepy.tech/badge/jai-sdk)](https://pepy.tech/project/jai-sdk)

# Installation

The source code is currently hosted on GitHub at: [https://github.com/jquant/jai-sdk](https://github.com/jquant/jai-sdk)

Installing jai-sdk using `pip`:

```sh
pip install jai-sdk
```

For more information, here is our [documentation](https://jai-sdk.readthedocs.io/en/latest/).

# [Get your Auth Key](https://jai-sdk.readthedocs.io/en/latest/source/quick_start.html#getting-your-authentication-key)

First, you'll need an Authorization key to use the backend API.

To get a Trial version API using the sdk, fill the values with your information:

```python
from jai import Jai

r = Jai.get_auth_key(email="EMAIL", firstName="FIRSTNAME", lastName="LASTNAME")

# or
r = Jai.get_auth_key(email="EMAIL", firstName="FIRSTNAME", lastName="LASTNAME", company="COMPANY")
```

If the response code is 201, then you should be receiving an email with your Auth Key.

# Get Started

If you already have an Auth Key, then you can use the sdk:

```python
from jai import Jai
j = Jai(AUTH_KEY)
```

## Read our documentation

For more information, here is our [documentation](https://jai-sdk.readthedocs.io/en/latest/).
