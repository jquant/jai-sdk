# -*- coding: utf-8 -*-
"""
Created on Sat Mar  6 16:17:31 2021

@author: Kazu
"""

import pandas as pd
import numpy as np
from .jai import Jai
from .processing import process_predict, process_similar


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
        self.columns = data.columns
        self._analyze()
        self.schedule()

    def _analyze(self):
        self._analysis = self.data.describe(include='all')
        self._analysis.loc['empty', :] = self.data.isna().sum()
        self._analysis.loc['dtype', :] = self.data.dtypes
        self._analysis.loc['unique', :] = self.data.nunique()
        pd.set_option("display.max_columns", max(self._analysis.shape))
        n, m = self._analysis.shape
        ind_order = self._analysis.T.columns[[-1, -2] + list(range(n - 2))]
        self._analysis = self._analysis.loc[ind_order]
        is_resolution = self._analysis.loc['dtype'] == 'object'
        is_fill = (self.data.isna().sum() > 0) & (
            self._analysis.loc['unique'] / self._analysis.loc['count'] < .05)
        is_sanity = self._analysis.loc['dtype'] == 'object'
        self._process = pd.DataFrame([is_resolution, is_fill, is_sanity],
                                     index=['resolution', 'fill', 'sanity'],
                                     columns=self.columns)
        return self._analysis

    def schedule(self):
        print("The processes scheduled on setup are:")
        print()

        print("Resolution")
        print("The following columns will be processed for resolution")
        print(*self.columns[self._process.loc['resolution']], sep=', ')
        print()

        print("Fill")
        print("The following columns will have their missing values filled")
        print(*self.columns[self._process.loc['fill']], sep=', ')
        print()

        print("Sanity")
        print("The following columns will be used on sanity check")
        print(*self.columns[self._process.loc['sanity']], sep=', ')
        print()

    def add_schedule(self, task, column):
        self._process.loc[task, column] = True

    def remove_schedule(self, task, column):
        self._process.loc[task, column] = False

    def fit(self, name=None, overwrite=False):
        if name is None:
            self.gen_name = self.__jai.generate_name(8)
        else:
            self.gen_name = name

        for col in self.columns[self._process.loc['resolution']]:
            print(f"Resolution {col}")
            name = 'r' + self.gen_name + col.lower().replace("-", "_").replace(
                " ", "_")
            self.__jai.embedding(name, self.data[col], overwrite=overwrite)

        for col in self.columns[self._process.loc['fill']]:
            print(f"Fill {col}")
            name = 'f' + self.gen_name + col.lower().replace("-", "_").replace(
                " ", "_")
            self.__jai.fill(name, self.data, col)

        print("Sanity")
        name = 's' + self.gen_name
        self.__jai.sanity(
            name,
            self.data,
            columns_ref=self.columns[self._process.loc['sanity']])

    def solve(self, task, column):
        if column not in self.columns:
            raise ValueError(f"Unable to find column {column}")

        if task == "resolution":
            name = 'r' + self.gen_name + column.lower().replace(
                "-", "_").replace(" ", "_")
            results = self.__jai.resolution(name, self.data[column])
            return process_similar(results, return_self=True)

        elif task == "fill":
            name = 'f' + self.gen_name + column.lower().replace(
                "-", "_").replace(" ", "_")
            results = self.__jai.fill(name, self.data, column)
            return process_predict(results)

        elif task == "sanity":
            name = 's' + self.gen_name
            results = self.__jai.sanity(
                name,
                self.data,
                columns_ref=self.columns[self._process.loc['sanity']])
            return process_predict(results)
        else:
            raise ValueError(f"unable to do task {task}")
