import concurrent
import json
import time
import warnings
from fnmatch import fnmatch
from typing import Optional

import matplotlib.pyplot as plt
import psutil
from pandas.api.types import is_numeric_dtype
from tqdm import tqdm, trange

from jai.core.base import BaseJai
from jai.core.types import PossibleDtypes
from jai.core.utils_funcs import data2json
from jai.core.validations import (check_dtype_and_clean, kwargs_validation)

import os

__all__ = ["Setup"]


class Setup(BaseJai):
    """
    Base class for communication with the Mycelia API.

    Used as foundation for more complex applications for data validation such
    as matching tables, resolution of duplicated values, filling missing values
    and more.

    """

    def __init__(self,
                 name: str,
                 auth_key: str = None,
                 url: str = None,
                 environment: str = "default",
                 var_env: str = "JAI_SECRET",
                 verbose: int = 1):
        """
        Initialize the Jai class.

        An authorization key is needed to use the Mycelia API.

        Parameters
        ----------

        Returns
        -------
            None

        """
        self.name = name
        self._type = None

        if url is None:
            self.__url = "https://mycelia.azure-api.net"
        else:
            self.__url = url[:-1] if url.endswith("/") else url

        if auth_key is None:
            auth_key = os.environ.get(var_env, "")

        self.header = {"Auth": auth_key, "environment": environment}
        self.user = self._user()

        if verbose:
            user_print = '\n'.join(
                [f"- {k}: {v}" for k, v in self.user.items()])
            print(f"Connection established.\n{user_print}")

    @property
    def url(self):
        """
        Get name and type of each database in your environment.
        """
        return self.__url

    def check_setted(self):
        if self._type is None:
            raise ValueError("Model not setted yet.")

    def set_text_model(self):
        pass

    @property
    def insert_params(self):
        return self._insert_params

    @insert_params.setter
    def set_insert(self,
                   batch_size: int = 16384,
                   filter_name: str = None,
                   overwrite: bool = False,
                   max_insert_workers: Optional[int] = None):
        self._insert_params = {
            "batch_size": batch_size,
            "filter_name": filter_name,
            "max_insert_workers": max_insert_workers,
            "overwrite": overwrite
        }

    def run(self,
            data,
            *,
            overwrite: bool = False,
            frequency_seconds: int = 1,
            verbose: int = 1):
        name = self.name
        if name in self.names:
            if overwrite:
                self.delete_database(name)
            else:
                raise KeyError(
                    f"Database '{name}' already exists in your environment.\
                        Set overwrite=True to overwrite it.")

        # make sure our data has the correct type and is free of NAs
        data = check_dtype_and_clean(data=data, db_type=db_type)

        # insert data
        insert_responses = self._insert_data(data=data)

        # train model
        body = kwargs_validation(db_type=db_type, **kwargs)
        setup_response = self._setup(name, self.body, overwrite)
        if "kwargs" in setup_response:
            print("\nRecognized setup args:")
            for key, value in setup_response["kwargs"].items():
                kwarg_input = kwargs.get(key, None)
                if isinstance(kwarg_input, dict):
                    value = json.loads(value)
                    intersection = kwarg_input.keys() & value.keys()
                    m = max([len(s) for s in intersection] + [0])
                    value = "".join(
                        [f"\n  * {k:{m}s}: {value[k]}" for k in intersection])
                print(f"- {key}: {value}")

        if frequency_seconds >= 1:
            self.wait_setup(name=name, frequency_seconds=frequency_seconds)

            if db_type in [
                    PossibleDtypes.selfsupervised, PossibleDtypes.supervised,
                    PossibleDtypes.recommendation_system
            ]:
                self.report(name, verbose)

        return insert_responses, setup_response

    def status(self, max_tries=5, patience=25):
        """
        Get the status of your JAI environment when training.

        Return
        ------
        response : dict
            A `JSON` file with the current status of the training tasks.

        Example
        -------
        >>> j.status
        {
            "Task": "Training",
            "Status": "Completed",
            "Description": "Training of database YOUR_DATABASE has ended."
        }
        """
        for _ in range(max_tries):
            try:
                return self._status()
            except BaseException:
                time.sleep(patience // max_tries)
        return self._status()

    def _insert_data(self,
                     data,
                     name,
                     db_type,
                     batch_size,
                     overwrite: bool = False,
                     max_insert_workers: Optional[int] = None,
                     filter_name: str = None,
                     predict: bool = False):
        """
        Insert raw data for training. This is a protected method.

        Args
        ----------
        name : str
            String with the name of a database in your JAI environment.
        db_type : str
            Database type (Supervised, SelSupervised, Text...)
        batch_size : int
            Size of batch to send the data.
        predict : bool
            Allows table type data to have only one column for predictions,
            if False, then tables must have at least 2 columns. `Default is False`.

        Return
        ------
        insert_responses : dict
            Dictionary of responses for each batch. Each response contains
            information of whether or not that particular batch was successfully inserted.
        """
        if max_insert_workers is None:
            pcores = psutil.cpu_count(logical=False)
        elif not isinstance(max_insert_workers, int):
            raise TypeError(
                f"Variable 'max_insert_workers' must be 'None' or 'int' instance, not {max_insert_workers.__class__.__name__}."
            )
        elif max_insert_workers > 0:
            pcores = max_insert_workers
        else:
            pcores = 1

        if self._check_ids_consistency(
                name=name, data=data, handle_error="bool") and not overwrite:
            return {0: "Data was already inserted. No operation was executed."}
        else:
            self.delete_raw_data(name)

        dict_futures = {}
        with concurrent.futures.ThreadPoolExecutor(
                max_workers=pcores) as executor:

            for i, b in enumerate(range(0, len(data), batch_size)):
                _batch = data.iloc[b:b + batch_size]
                data_json = data2json(_batch,
                                      dtype=db_type,
                                      filter_name=filter_name,
                                      predict=predict)
                task = executor.submit(self._insert_json, name, data_json,
                                       filter_name)
                dict_futures[task] = i

            with tqdm(total=len(dict_futures), desc="Insert Data") as pbar:
                insert_responses = {}
                for future in concurrent.futures.as_completed(dict_futures):
                    arg = dict_futures[future]
                    insert_responses[arg] = future.result()
                    pbar.update(1)

        # check if we inserted everything we were supposed to
        self._check_ids_consistency(name=name, data=data)

        return insert_responses

    def report(self, name, verbose: int = 2, return_report: bool = False):
        """
        Get a report about the training model.

        Parameters
        ----------
        name : str
            String with the name of a database in your JAI environment.
        verbose : int, optional
            Level of description. The default is 2.
            Use verbose 2 to get the loss graph, verbose 1 to get only the
            metrics result.

        Returns
        -------
        dict
            Dictionary with the information.

        """
        dtype = self.get_dtype(name)
        if dtype not in [
                PossibleDtypes.selfsupervised, PossibleDtypes.supervised,
                PossibleDtypes.recommendation_system
        ]:
            return None

        result = self._report(name, verbose)
        if return_report:
            return result

        if 'Model Training' in result.keys():
            plots = result['Model Training']

            plt.plot(*plots['train'], label="train loss")
            plt.plot(*plots['val'], label="val loss")
            plt.title("Training Losses")
            plt.legend()
            plt.xlabel("epoch")
            plt.show()

        print("\nSetup Report:")
        print(result['Model Evaluation']) if 'Model Evaluation' in result.keys(
        ) else None
        print()
        print(result["Loading from checkpoint"].split("\n")
              [1]) if 'Loading from checkpoint' in result.keys() else None

    def _check_ids_consistency(self, name, data, handle_error="raise"):
        """
        Check if inserted data is consistent with what we expect.
        This is mainly to assert that all data was properly inserted.

        Args
        ----
        name : str
            Database name.
        data : pandas.DataFrame or pandas.Series
            Inserted data.
        handle_error : 'raise' or 'bool'
            If data is inconsistent:
            - `raise`: delete data and raise an error.
            - `bool`: returns False.

        Return
        ------
        bool or Exception
            If an inconsistency is found, an error is raised. If no inconsistency is found, returns True.
        """
        handle_error = handle_error.lower()
        if handle_error not in ['raise', 'bool']:
            warnings.warn(
                f"handle_error must be `raise` or `bool`, found: `{handle_error}`. Using `raise`."
            )
            handle_error = 'raise'

        # using mode='simple' to reduce the volume of data transit.
        try:
            inserted_ids = self._temp_ids(name, "simple")
        except ValueError as error:
            if handle_error == "raise":
                raise error
            return False

        if len(data) != int(inserted_ids[0].split()[0]):
            if handle_error == "raise":
                print(f"Found invalid ids: {inserted_ids[0]}")
                print(self.delete_raw_data(name))
                raise Exception(
                    "Something went wrong on data insertion. Please try again."
                )
            return False
        return True

    def wait_setup(self, name: str, frequency_seconds: int = 1):
        """
        Wait for the setup (model training) to finish

        Placeholder method for scripts.

        Args
        ----
        name : str
            String with the name of a database in your JAI environment.
        frequency_seconds : int, optional
            Number of seconds apart from each status check. `Default is 5`.

        Return
        ------
        None.
        """

        def get_numbers(sts):
            curr_step, max_iterations = sts["Description"].split(
                "Iteration: ")[1].strip().split(" / ")
            return int(curr_step), int(max_iterations)

        status = self.status[name]
        starts_at, max_steps = status["CurrentStep"], status["TotalSteps"]

        step = starts_at
        aux = 0
        sleep_time = frequency_seconds
        try:
            with tqdm(total=max_steps,
                      desc="JAI is working",
                      bar_format='{l_bar}{bar}|{n_fmt}/{total_fmt} [{elapsed}]'
                      ) as pbar:
                while status['Status'] != 'Task ended successfully.':
                    if status['Status'] == 'Something went wrong.':
                        raise BaseException(status['Description'])
                    elif fnmatch(status["Description"], "*Iteration:*"):
                        # create a second progress bar to track
                        # training progress
                        _, max_iterations = get_numbers(status)
                        with tqdm(total=max_iterations,
                                  desc=f"[{name}] Training",
                                  leave=False) as iteration_bar:
                            while fnmatch(status["Description"],
                                          "*Iteration:*"):
                                curr_step, _ = get_numbers(status)
                                step_update = curr_step - iteration_bar.n
                                if step_update > 0:
                                    iteration_bar.update(step_update)
                                    sleep_time = frequency_seconds
                                else:
                                    sleep_time += frequency_seconds
                                time.sleep(sleep_time)
                                status = self.status[name]
                            # training might stop early, so we make the progress bar appear
                            # full when early stopping is reached -- peace of mind
                            iteration_bar.update(max_iterations -
                                                 iteration_bar.n)

                    if (step == starts_at) and (aux == 0):
                        pbar.update(starts_at)
                    else:
                        pbar.update(step - starts_at)
                        starts_at = step

                    step = status["CurrentStep"]
                    time.sleep(frequency_seconds)
                    status = self.status[name]
                    aux += 1

                if (starts_at != max_steps) and aux != 0:
                    pbar.update(max_steps - starts_at)
                elif (starts_at != max_steps) and aux == 0:
                    pbar.update(max_steps)

        except KeyboardInterrupt:
            print("\n\nInterruption caught!\n\n")
            raise KeyboardInterrupt(self._cancel_setup(name))

        self._delete_status(name)
        return status

    def delete_raw_data(self, name: str):
        """
        Delete raw data. It is good practice to do this after training a model.

        Args
        ----
        name : str
            String with the name of a database in your JAI environment.

        Return
        -------
        response : dict
            Dictionary with the API response.

        Example
        ----------
        >>> name = 'chosen_name'
        >>> j = Jai(AUTH_KEY)
        >>> j.delete_raw_data(name=name)
        'All raw data from database 'chosen_name' was deleted!'
        """
        return self._delete_raw_data(name)
