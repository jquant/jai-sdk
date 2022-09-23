import json
import time
from collections.abc import Iterable
from fnmatch import fnmatch
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from ..core.utils_funcs import check_filters, print_args
from ..core.validations import check_dtype_and_clean
from ..types.generic import PossibleDtypes
from ..types.hyperparams import InsertParams
from .base import TaskBase
from .query import Query

__all__ = ["Trainer"]


def get_numbers(status):
    if fnmatch(status["Description"], "*Iteration:*"):
        curr_step, max_iterations = (
            status["Description"].split("Iteration: ")[1].strip().split(" / ")
        )
        return True, int(curr_step), int(max_iterations)
    return False, 0, 0


def flatten_sample(sample):
    for el in sample:
        if isinstance(el, Iterable) and not isinstance(el, (str, bytes)):
            yield from flatten_sample(el)
        else:
            yield el


DEFAULT_NAME = "main"


class Trainer(TaskBase):
    """
    Trainer task class.

    An authorization key is needed to use the Jai API.

    Parameters
    ----------
    name : str
        String with the name of a database in your JAI environment.
    environment : str
        Jai environment id or name to use. Defaults to "default"
    env_var : str
        The environment variable that contains the JAI authentication token.
        Defaults to "JAI_AUTH".
    verbose : int
        The level of verbosity. Defaults to 1
    safe_mode : bool
        When safe_mode is True, responses from Jai API are validated.
        If the validation fails, the current version you are using is probably incompatible with the current API version.
        We advise updating it to a newer version. If the problem persists and you are on the latest SDK version, please open an issue so we can work on a fix.
        Defaults to False.

    Example
    -------
    >>> from jai import Trainer
    ...
    >>> trainer = Trainer(name)
    """

    def __init__(
        self,
        name: str,
        auth_key: str = None,
        environment: str = "default",
        env_var: str = "JAI_AUTH",
        verbose: int = 1,
        safe_mode: bool = False,
    ):

        super(Trainer, self).__init__(
            name=name,
            auth_key=auth_key,
            environment=environment,
            env_var=env_var,
            verbose=verbose,
            safe_mode=safe_mode,
        )

        self._fit_parameters = None
        self._insert_parameters = {"batch_size": 16384, "max_insert_workers": None}

    @property
    def insert_parameters(self):
        """
        Parameters used for insert data.
        """
        return self._insert_parameters

    @insert_parameters.setter
    def insert_parameters(self, value: InsertParams):
        """
        The method takes a dictionary of parameters

        Returns:
          A dictionary of the InsertParams class.
        """
        self._insert_parameters = InsertParams(**value).dict()

    @property
    def fit_parameters(self):
        if self._fit_parameters is None:
            raise ValueError(
                "No parameter was set, please use `set_parameters` method first."
            )
        return self._fit_parameters

    def update_database(self, name: str, display_name: str = None, project: str = None):
        return self._update_database(
            name=name, display_name=display_name, project=project
        )

    def set_parameters(
        self,
        db_type: str,
        hyperparams: Dict[str, Dict] = None,
        features: Dict[str, Dict] = None,
        num_process: dict = None,
        cat_process: dict = None,
        datetime_process: dict = None,
        pretrained_bases: list = None,
        label: dict = None,
        split: dict = None,
        verbose: int = 1,
    ):
        """
        It checks the input parameters and sets the `fit_parameters` attribute for setup.

        Args:
        db_type (str):Type of the database to be created.
        hyperparams (dict): Dictionary of the fit parameters. Varies for each database type.
        features (dict): Dictionary of name of the features as keys and dictionary of parameters for each feature.
        num_process (dict): Dictionary defining the default process for numeric features.
        cat_process (dict): Dictionary defining the default process for categorical features.
        datetime_process (dict): Dictionary defining the default process for datetime features.
        pretrained_bases (list): List of dictionaries mapping the features to the databases trained previously.
        label (dict): Dictionary defining the label.
        split (dict): Dictionary defining the train/validation split for the model training.

        Example
        -------
        >>> from jai import Trainer
        ...
        >>> trainer = Trainer(name)
        >>> trainer.set_parameters(db_type)

        """

        self._input_kwargs = dict(
            db_type=db_type,
            hyperparams=hyperparams,
            features=features,
            num_process=num_process,
            cat_process=cat_process,
            datetime_process=datetime_process,
            pretrained_bases=pretrained_bases,
            label=label,
            split=split,
        )

        # I figure we don't need a safe_mode validation here
        # because this is already a validation method.
        self._fit_parameters = self._check_parameters(
            db_type=db_type,
            hyperparams=hyperparams,
            features=features,
            num_process=num_process,
            cat_process=cat_process,
            datetime_process=datetime_process,
            pretrained_bases=pretrained_bases,
            label=label,
            split=split,
        )

        print_args(self.fit_parameters, self._input_kwargs, verbose=verbose)

    def _check_pretrained_bases(self, data: pd.DataFrame, pretrained_bases: List):
        """
        Processes the following checks:
        - If RecommendationSystem:
          - checks if both tower names are in pretrained_bases.
          - For each pretrained base:
            - Checks if every id value on `data['main']` column is in the "towers'" ids.
            - Checks if every id value on "tower" column is in the existing base ids.
        - Else, for each pretrained base:
            - Checks if every id value on `data` column is in the existing base ids.

        Args
        ----
        data (pd.DataFrame): data
        pretrained_bases (List of dicts): list of pretrained bases

        Raises
        ------
        ValueError: On Recomendation System, if ids between bases don't match.
        KeyError: If there are missing ids on the id_name column

        """
        if isinstance(data, dict):
            towers = set(data.keys()) - set([self.name, DEFAULT_NAME])
            pretrained_names = [b["db_parent"] for b in pretrained_bases]
            if not towers <= set(pretrained_names):
                raise ValueError(
                    f"Both {towers} keys must be in pretrained bases:\n"
                    f"db_parents: {pretrained_names}"
                )

        for base in pretrained_bases:
            parent_name = base["db_parent"]
            column = base["id_name"]

            if isinstance(data, pd.DataFrame):
                flat_ids = np.unique(list(flatten_sample(data[column])))
                ids = self._ids(parent_name, mode="complete")
            elif parent_name in data.keys():
                data_parent = data[parent_name]
                df = (
                    data[DEFAULT_NAME]
                    if DEFAULT_NAME in data.keys()
                    else data[self.name]
                )
                flat_ids = np.unique(list(flatten_sample(df[column])))
                ids = (
                    data_parent["id"]
                    if "id" in data_parent.columns
                    else data_parent.index
                )
            else:
                for df in data.values():
                    if column in df.columns:
                        flat_ids = np.unique(list(flatten_sample(df[column])))
                        ids = self._ids(parent_name, mode="complete")
                        break

            inverted_in = np.isin(flat_ids, ids, invert=True)
            if inverted_in.sum() > 0:
                missing = flat_ids[inverted_in].tolist()
                raise KeyError(
                    f"Id values on column `{column}` must belong to the set of Ids from database {parent_name}.\n"
                    f"Missing: {missing}"
                )

    def fit(
        self,
        data,
        *,
        overwrite: bool = False,
        frequency_seconds: int = 1,
        verbose: int = 1,
    ):
        """
        Takes in a dataframe or dictionary of dataframes, and inserts the data into Jai.

        Otherwise, it calls the `wait_setup` function to wait for the model to finish training, and then
        calls the `report` function to print out the model's performance metrics.

        Finally, it returns the `get_query` function, which returns the class to consume the model..

        Args
        ----
        data: pd.DataFrame or dict of pd.DataFrame)
            The data to be inserted into the database. It is required to be an pandas.Dataframe,
            unless it's a RecommendationSystem, then it's a dictionary of pandas.DataFrame.
        overwrite: bool
            If overwrite is True, then deletes previous database with the same name if
            exists. Defaults to False.
        frequency_seconds:int
            How often to check the status of the model. If `frequency_seconds` is
            less than 1, it returns the `insert_responses` and `setup_response` and it won't wait for
            training to finish, allowing to perform other actions, but could cause errors on some scripts
            if the model is expected to be ready for consuming. Defaults to 1.

        Returns
        -------
        Tuple: tuple
            If `frequency_seconds < 1`, the returned value is a tuple of two elements.
            The first element is a list of responses from the insert_data function.
            The second element is a dictionary of the response from the setup function.

        Query: jai.Query class
            If `frequency_seconds >= 1`, then the return will be an Query class of the database trained.
            If the database is a RecommendationSystem type, then it will return a dictionary of Query classes.

        Example
        -------
        >>> from jai import Trainer
        ...
        >>> trainer = Trainer(name)
        >>> trainer.fit(data)

        """
        if self.is_valid():
            if overwrite:
                self.delete_database()
            else:
                raise KeyError(
                    f"Database '{self.name}' already exists in your environment.\
                        Set overwrite=True to overwrite it."
                )
        self._check_pretrained_bases(
            data, self.fit_parameters.get("pretrained_bases", [])
        )

        if isinstance(data, (pd.Series, pd.DataFrame)):
            # make sure our data has the correct type and is free of NAs
            data = check_dtype_and_clean(data=data, db_type=self.db_type)

            # insert data
            self._delete_raw_data(self.name)
            insert_responses = self._insert_data(
                data=data,
                name=self.name,
                db_type=self.fit_parameters["db_type"],
                batch_size=self.insert_parameters["batch_size"],
                has_filter=check_filters(data, self.fit_parameters.get("features", {})),
                max_insert_workers=self.insert_parameters["max_insert_workers"],
                predict=False,
            )

        elif isinstance(data, dict):
            if self.fit_parameters["db_type"] != PossibleDtypes.recommendation_system:
                raise ValueError(
                    "Data as type `dict` is only used for RecommendationSystem databases."
                )

            # loop insert data
            for name, value in data.items():

                # make sure our data has the correct type and is free of NAs
                value = check_dtype_and_clean(data=value, db_type=self.db_type)

                if name == DEFAULT_NAME:
                    name = self.name

                # insert data
                self._delete_raw_data(name)
                insert_responses = self._insert_data(
                    data=value,
                    name=name,
                    db_type=self.fit_parameters["db_type"],
                    batch_size=self.insert_parameters["batch_size"],
                    has_filter=check_filters(
                        value, self.fit_parameters.get("features", {})
                    ),
                    max_insert_workers=self.insert_parameters["max_insert_workers"],
                    predict=False,
                )
        elif self.fit_parameters["db_type"] == PossibleDtypes.recommendation_system:
            raise ValueError("Data must be a dictionary of pd.DataFrames.")
        else:
            raise ValueError("Data must be a pd.Series or pd.Dataframe.")

        # send request to start training
        setup_response = self._setup(
            self.name, self.fit_parameters, overwrite=overwrite
        )

        print_args(
            {k: json.loads(v) for k, v in setup_response["kwargs"].items()},
            self._input_kwargs,
            verbose=verbose,
        )

        if frequency_seconds < 1:
            return insert_responses, setup_response

        self.wait_setup(frequency_seconds=frequency_seconds)
        self.report(verbose)

        if self.fit_parameters["db_type"] == PossibleDtypes.recommendation_system:
            towers = list(set(data.keys()) - set([self.name, DEFAULT_NAME]))
            return {
                towers[0]: self.get_query(name=towers[0]),
                towers[1]: self.get_query(name=towers[1]),
            }

        return self.get_query()

    def append(self, data, *, frequency_seconds: int = 1):
        """
        Insert raw data and extract their latent representation.

        This method should be used when we already setup up a database using `fit()`
        and want to create the vector representations of new data
        using the model we already trained for the given database.

        Args
        ----
        data : pandas.DataFrame
            Data to be inserted and used for training.
        frequency_seconds : int
            Time in between each check of status. If less than 1, it won't wait for setup
            to finish, allowing to perform other actions, but could cause errors on some
            scripts. `Default is 1`.

        Return
        -------
        insert_responses: dict
            Dictionary of responses for each batch. Each response contains
            information of whether or not that particular batch was successfully inserted.

        Example
        -------
        >>> from jai import Trainer
        ...
        >>> trainer = Trainer(name)
        >>> trainer.append(data)
        """
        if not self.is_valid():
            raise KeyError(
                f"Database '{self.name}' does not exist in your environment.\n"
                "Run a `setup` set your database up first."
            )

        # delete data reamains
        self.delete_raw_data()

        # make sure our data has the correct type and is free of NAs
        data = check_dtype_and_clean(data=data, db_type=self.db_type)

        # insert data
        self._delete_raw_data(self.name)
        insert_responses = self._insert_data(
            data=data,
            name=self.name,
            db_type=self.db_type,
            batch_size=self.insert_parameters["batch_size"],
            has_filter=self.describe()["has_filter"],
            max_insert_workers=self.insert_parameters["max_insert_workers"],
            predict=True,
        )

        # add data per se
        add_data_response = self._append(name=self.name)

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
        """
        all_status = self._status()
        if self.name not in all_status.keys():
            raise ValueError(f"No status found for `{self.name}`")
        return all_status[self.name]

    def report(self, verbose: int = 2, return_report: bool = False):
        """
        Get a report about the training model.

        Parameters
        ----------
        verbose : int, optional
            Level of description. The default is 2.
            Use verbose 2 to get the loss graph, verbose 1 to get only the
            metrics result.
        return_report : bool, optional
            Returns the report dictionary and does not print or plot anything. The default is False.


        Returns
        -------
        dict
            Dictionary with the information.

        Example
        -------
        >>> from jai import Trainer
        ...
        >>> trainer = Trainer(name)
        >>> trainer.report()
        """
        if self.db_type not in [
            PossibleDtypes.selfsupervised,
            PossibleDtypes.supervised,
            PossibleDtypes.recommendation_system,
        ]:
            return None

        report = self._report(self.name, verbose)

        if return_report:
            return report

        if "Model Training" in report.keys():
            plots = report["Model Training"]

            plt.plot(*plots["train"], label="train loss")
            plt.plot(*plots["val"], label="val loss")
            plt.title("Training Losses")
            plt.legend()
            plt.xlabel("epoch")
            plt.show()

        print("\nSetup Report:")
        print(
            report["Model Evaluation"]
        ) if "Model Evaluation" in report.keys() else None
        print()
        print(
            report["Loading from checkpoint"].split("\n")[1]
        ) if "Loading from checkpoint" in report.keys() else None

    def wait_setup(self, frequency_seconds: int = 1):
        """
        Wait for the fit (model training) to finish

        Args
        ----
        frequency_seconds : int, optional
            Number of seconds apart from each status check. `Default is 5`.

        Return
        ------
        None.
        """

        end_message = "Task ended successfully."
        error_message = "Something went wrong."

        status = self.status()
        current, max_steps = status["CurrentStep"], status["TotalSteps"]

        step = current
        is_init = True
        sleep_time = frequency_seconds
        try:
            with tqdm(
                total=max_steps,
                desc="JAI is working",
                bar_format="{l_bar}{bar}|{n_fmt}/{total_fmt} [{elapsed}]",
            ) as pbar:
                while status["Status"] != end_message:
                    if status["Status"] == error_message:
                        raise BaseException(status["Description"])

                    iteration, _, max_iterations = get_numbers(status)
                    if iteration:
                        with tqdm(
                            total=max_iterations,
                            desc=f"[{self.name}] Training",
                            leave=False,
                        ) as iteration_bar:
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
                            iteration_bar.update(max_iterations - iteration_bar.n)

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
            raise KeyboardInterrupt(response)

        response = self._delete_status(self.name)
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

        Example
        -------
        >>> from jai import Trainer
        ...
        >>> trainer = Trainer(name)
        >>> trainer.delete_ids([0, 1])
        """
        return self._delete_ids(self.name, ids)

    def delete_raw_data(self):
        """
        Delete raw data. It is good practice to do this after training a model.

        Return
        -------
        response : dict
            Dictionary with the API response.

        Example
        -------
        >>> from jai import Trainer
        ...
        >>> trainer = Trainer(name)
        >>> trainer.delete_raw_data()

        """
        return self._delete_raw_data(self.name)

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
        >>> from jai import Trainer
        ...
        >>> trainer = Trainer(name)
        >>> trainer.delete_database()
        """
        return self._delete_database(self.name)

    def get_query(self, name: str = None):
        """
        This method returns a new `Query` object with the same initial values as the current `Trainer`
        object

        Args
        ----
        name: str
            The name of the query. Defaults to the same name as the current `Trainer` object.

        Returns
        -------
            A Query object with the name and init values.

        Example
        -------
        >>> from jai import Trainer
        ...
        >>> trainer = Trainer(name)
        >>> trainer.get_query()
        """
        if name is None:
            return Query(name=self.name, **self._init_values)
        return Query(name=name, **self._init_values)
