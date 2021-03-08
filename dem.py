# -*- coding: utf-8 -*-
"""
Created on Sat Mar  6 16:18:21 2021

@author: Kazu
"""

from jai import experimental
import pandas as pd

df = pd.read_pickle("data/raw_eans_780_categorias.pickle")
AUTH_KEY = "22d98a05a24f43d09a1e9f82ec25d6f0"

j = experimental.Name(AUTH_KEY)
j.data = df