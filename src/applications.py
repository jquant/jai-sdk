# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 21:39:24 2021

@author: Kazu
"""


import numpy as np
from jai import Jai

from auxiliar_funcs.utils_funcs import process_similar


def match(data1, data2, auth_key, only_duplicated = False, threshold = None, top_k = 10):

    jai = Jai(auth_key)
    nt = np.clip(np.round(len(data1)/10, -3), 1000, 10000)
    name = jai.generate_name(20, prefix='sdk_', suffix='_textedit')
    print(f"name: {name}")
    jai.setup(name, data1, batch_size=10000, db_type='TextEdit',
              hyperparams={"nt": nt})
    jai.wait_setup(name, 20)
    results = jai.similar(name, data2, top_k=top_k, batch_size=10000)
    jai.delete_raw_data(name)
    return process_similar(results, return_self=True)



def resolution(data, auth_key,  only_duplicated = False, threshold = None, top_k = 10):

    jai = Jai(auth_key)
    nt = np.clip(np.round(len(data)/10, -3), 1000, 10000)
    name = jai.generate_name(20, prefix='sdk_', suffix='_textedit')
    print(f"name: {name}")
    jai.setup(name, data, batch_size=10000, db_type='TextEdit',
              hyperparams={"nt": nt})
    jai.wait_setup(name, 20)
    results = jai.similar(name, data.index, top_k=top_k, batch_size=10000)
    jai.delete_raw_data(name)
    return process_similar(results, return_self=False)