# -*- coding: utf-8 -*-
"""
Created on Sat Mar  6 16:17:31 2021

@author: Kazu
"""

import pandas as pd
import numpy as np
from .jai import Jai


class Name():
    def __init__(self, auth):
        self.__jai = Jai(auth)
        self.__data = None

    @property
    def data(self):
        if self.__data is None:
            raise ValueError("No data inserted yet.")
        return self.__data

    @data.setter
    def data(self, data):
        if isinstance(data, pd.DataFrame):
            self.__data = data
        elif isinstance(data, pd.Series):
            self.__data = data.to_frame()
        elif isinstance(data, [np.ndarray, list, set, tuple]):
            self.__data = pd.DataFrame(data)
        else:
            raise ValueError("Data type not recognized.")
        self._analyze()

    def _analyze(self):
        self._analysis = self.data.describe(include='all')
        self._analysis.loc['empty', :] = self.data.isna().sum()
        self._analysis.loc['dtype', :] = self.data.dtypes
        pd.set_option("display.max_columns", max(self._analysis.shape))
        n, m = self._analysis.shape
        ind_order = self._analysis.T.columns[[-1, -2] + list(range(n - 2))]
        self._analysis = self._analysis.loc[ind_order]
        return self._analysis
