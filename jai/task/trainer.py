import time
from fnmatch import fnmatch

import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd

from .base import TaskBase
from ..core.utils_funcs import print_args
from ..core.validations import check_response, check_dtype_and_clean

from ..types.generic import PossibleDtypes
from ..types.hyperparams import InsertParams

from ..types.responses import (UserResponse, Report1Response, Report2Response,
                               AddDataResponse, StatusResponse, InfoResponse)

from typing import Dict
import sys

__all__ = ["Trainer"]


def get_numbers(status):
    if fnmatch(status["Description"], "*Iteration:*"):
        curr_step, max_iterations = status["Description"].split(
            "Iteration: ")[1].strip().split(" / ")
        return int(curr_step), int(max_iterations)
    return False, 0, 0


class Trainer(TaskBase):
    """
    Base class for communication with the Mycelia API.

    Used as foundation for more complex applications for data validation such
    as matching tables, resolution of duplicated values, filling missing values
    and more.

    """

    def __init__(self,
                 name: str,
                 environment: str = "default",
                 env_var: str = "JAI_AUTH",
                 verbose: int = 1,
                 safe_mode: bool = False):
        """
        Initialize the Jai class.

        An authorization key is needed to use the Mycelia API.

        Parameters
        ----------

        Returns
        -------
            None

        """
        super(Trainer, self).__init__(name=name,
                                      environment=environment,
                                      env_var=env_var,
                                      verbose=verbose,
                                      safe_mode=safe_mode)

        self._verbose = verbose
        self._setup_params = None

        self._insert_params = {
            "batch_size": 16384,
            "filter_name": None,
            "overwrite": True,
            "max_insert_workers": None
        }

    @property
    def insert_params(self):
        return self._insert_params

    @insert_params.setter
    def insert_params(self, value):
        self._insert_params = InsertParams(value).dict()

    @property
    def setup_params(self):
        if self._setup_params is None:
            raise ValueError(
                "Generic error message.")  #TODO: run set_params first message.
        return self._setup_params

    def set_params(self,
                   db_type: str,
                   hyperparams=None,
                   features=None,
                   num_process: dict = None,
                   cat_process: dict = None,
                   datetime_process: dict = None,
                   pretrained_bases: list = None,
                   label: dict = None,
                   split: dict = None):

        kwargs = dict(db_type=db_type,
                      hyperparams=hyperparams,
                      features=features,
                      num_process=num_process,
                      cat_process=cat_process,
                      datetime_process=datetime_process,
                      pretrained_bases=pretrained_bases,
                      label=label,
                      split=split)

        self._setup_params = self._check_params(
            db_type=db_type,
            hyperparams=hyperparams,
            features=features,
            num_process=num_process,
            cat_process=cat_process,
            datetime_process=datetime_process,
            pretrained_bases=pretrained_bases,
            label=label,
            split=split)

        print(self.setup_params)
        print(kwargs)
        print_args(kwargs, self.setup_params)

    def setup(self,
              data,
              *,
              overwrite: bool = False,
              frequency_seconds: int = 1):

        if self.is_valid():
            if overwrite:
                self.delete_database()
            else:
                raise KeyError(
                    f"Database '{self.name}' already exists in your environment.\
                        Set overwrite=True to overwrite it.")
        params = self.setup_params
        # make sure our data has the correct type and is free of NAs
        data = check_dtype_and_clean(data=data, db_type=self.db_type)

        # insert data
        insert_responses = self._insert_data(
            data=data,
            name=self.name,
            db_type=self.setup_params['db_type'],
            batch_size=self.insert_params['batch_size'],
            overwrite=self.insert_params['overwrite'],
            max_insert_workers=self.insert_params['max_insert_workers'],
            filter_name=self.insert_params['filter_name'],
            predict=False)

        # train model
        setup_response = self._setup(self.name, params, overwrite)

        print_args(params, setup_response['kwargs'])

        if frequency_seconds >= 1:
            self.wait_setup(frequency_seconds=frequency_seconds)
            self.report(self._verbose)

        return insert_responses, setup_response

    def append(self, data, *, frequency_seconds: int = 1):
        """
        Insert raw data and extract their latent representation.

        This method should be used when we already setup up a database using `setup()`
        and want to create the vector representations of new data
        using the model we already trained for the given database.

        Args
        ----
        data : pandas.DataFrame or pandas.Series
            Data to be inserted and used for training.
        frequency_seconds : int
            Time in between each check of status. `Default is 10`.

        Return
        -------
        insert_responses: dict
            Dictionary of responses for each batch. Each response contains
            information of whether or not that particular batch was successfully inserted.
        """
        if not self.is_valid():
            raise KeyError(
                f"Database '{self.name}' does not exist in your environment.\n"
                "Run a `setup` set your database up first.")

        # delete data reamains
        self.delete_raw_data()

        # make sure our data has the correct type and is free of NAs
        data = check_dtype_and_clean(data=data, db_type=self.db_type)

        # insert data
        insert_responses = self._insert_data(
            data=data,
            name=self.name,
            db_type=self.db_type,
            batch_size=self.insert_params['batch_size'],
            overwrite=self.insert_params['overwrite'],
            max_insert_workers=self.insert_params['max_insert_workers'],
            filter_name=self.insert_params['filter_name'],
            predict=True)

        # add data per se
        add_data_response = self._append(name=self.name)
        if self.safe_mode:
            add_data_response = check_response(AddDataResponse,
                                               add_data_response)

        if frequency_seconds >= 1:
            self.wait_setup(frequency_seconds=frequency_seconds)

        return insert_responses, add_data_response

    def status(self):
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
        status = self._status()[self.name]
        if self.safe_mode:
            return check_response(StatusResponse, status).dict()
        return status

    def report(self, verbose: int = 2, return_report: bool = False):
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
        if self.db_type not in [
                PossibleDtypes.selfsupervised, PossibleDtypes.supervised,
                PossibleDtypes.recommendation_system
        ]:
            return None

        report = self._report(self.name, verbose)

        if self.safe_mode:
            if verbose >= 2:
                report = check_response(Report2Response,
                                        report).dict(by_alias=True)
            elif verbose == 1:
                report = check_response(Report1Response,
                                        report).dict(by_alias=True)
            else:
                report = check_response(Report1Response,
                                        report).dict(by_alias=True)

        if return_report:
            return report

        if 'Model Training' in report.keys():
            plots = report['Model Training']

            plt.plot(*plots['train'], label="train loss")
            plt.plot(*plots['val'], label="val loss")
            plt.title("Training Losses")
            plt.legend()
            plt.xlabel("epoch")
            plt.show()

        print("\nSetup Report:")
        print(report['Model Evaluation']) if 'Model Evaluation' in report.keys(
        ) else None
        print()
        print(report["Loading from checkpoint"].split("\n")
              [1]) if 'Loading from checkpoint' in report.keys() else None

    def wait_setup(self, frequency_seconds: int = 1):
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

        if frequency_seconds < 1:
            return

        end_message = 'Task ended successfully.'
        error_message = 'Something went wrong.'

        status = self.status()
        current, max_steps = status["CurrentStep"], status["TotalSteps"]

        step = current
        is_init = True
        sleep_time = frequency_seconds
        try:
            with tqdm(total=max_steps,
                      desc="JAI is working",
                      bar_format='{l_bar}{bar}|{n_fmt}/{total_fmt} [{elapsed}]'
                      ) as pbar:
                while status['Status'] != end_message:
                    if status['Status'] == error_message:
                        raise BaseException(status['Description'])

                    iteration, _, max_iterations = get_numbers(status)
                    if iteration:
                        with tqdm(total=max_iterations,
                                  desc=f"[{self.name}] Training",
                                  leave=False) as iteration_bar:
                            while iteration:
                                iteration, curr_step, _ = get_numbers(status)
                                step_update = curr_step - iteration_bar.n
                                if step_update > 0:
                                    iteration_bar.update(step_update)
                                    sleep_time = frequency_seconds
                                else:
                                    sleep_time += frequency_seconds
                                time.sleep(sleep_time)
                                status = self.status()
                            # training might stop early, so we make the progress bar appear
                            # full when early stopping is reached -- peace of mind
                            iteration_bar.update(max_iterations -
                                                 iteration_bar.n)

                    if (step == current) and is_init:
                        pbar.update(current)
                    else:
                        pbar.update(step - current)
                        current = step

                    step = status["CurrentStep"]
                    time.sleep(frequency_seconds)
                    status = self.status()
                    is_init = False

                if (current != max_steps) and not is_init:
                    pbar.update(max_steps - current)
                elif (current != max_steps) and is_init:
                    pbar.update(max_steps)

        except KeyboardInterrupt:
            print("\n\nInterruption caught!\n\n")
            response = self._cancel_setup(self.name)
            if self.safe_mode:
                response = check_response(str, response)
            raise KeyboardInterrupt(response)

        response = self._delete_status(self.name)
        if self.safe_mode:
            check_response(str, response)
        return status

    def delete_ids(self, ids):
        """
        Delete the specified ids from database.

        Args
        ----

        ids : list
            List of ids to be removed from database.

        Return
        -------
        response : dict
            Dictionary with the API response.
        """
        response = self._delete_ids(self.name, ids)
        if self.safe_mode:
            return check_response(str, response)
        return response

    def delete_raw_data(self):
        """
        Delete raw data. It is good practice to do this after training a model.


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
        response = self._delete_raw_data(self.name)
        if self.safe_mode:
            return check_response(str, response)
        return response

    def delete_database(self):
        """
        Delete a database and everything that goes with it (I thank you all).

        Args
        ----
        name : str
            String with the name of a database in your JAI environment.

        Return
        ------
        response : dict
            Dictionary with the API response.

        Example
        -------
        >>> name = 'chosen_name'
        >>> j = Jai(AUTH_KEY)
        >>> j.delete_database(name=name)
        'Bombs away! We nuked database chosen_name!'
        """
        response = self._delete_database(self.name)
        if self.safe_mode:
            return check_response(str, response)
        return response
