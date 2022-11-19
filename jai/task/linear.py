import time
from fnmatch import fnmatch
from typing import Optional

import pandas as pd
from tqdm import tqdm

from ..types.generic import PossibleDtypes
from ..types.linear import (
    ClassificationHyperparams,
    ClassificationTasks,
    RegressionHyperparams,
    RegressionTasks,
    SchedulerType,
    SGDClassificationHyperparams,
    SGDRegressionHyperparams,
    TrainMode,
)
from .base import TaskBase

__all__ = ["LinearModel"]


def get_numbers(status):
    if fnmatch(status["Description"], "*Iteration:*"):
        curr_step, max_iterations = (
            status["Description"].split("Iteration: ")[1].strip().split(" / ")
        )
        return True, int(curr_step), int(max_iterations)
    return False, 0, 0


class LinearModel(TaskBase):
    """
    Linear Model class.

    An authorization key is needed to use the Jai API.

    Parameters
    ----------
    name: str
        String with the name of a database in your JAI environment.
    task: str
        Task of the linear model. One of {`regression`, `sgd_regression`, `classification`, `sgd_classification`}.
    environment: str
        Jai environment id or name to use. Defaults to "default"
    env_var: str
        The environment variable that contains the JAI authentication token.
        Defaults to "JAI_AUTH".
    verbose: int
        The level of verbosity. Defaults to 1
    safe_mode: bool
        When safe_mode is True, responses from Jai API are validated.
        If the validation fails, the current version you are using is probably incompatible with the current API version.
        We advise updating it to a newer version. If the problem persists and you are on the latest SDK version, please open an issue so we can work on a fix.
        Defaults to False.

    """

    def __init__(
        self,
        name: str,
        task: str,
        auth_key: str = None,
        environment: str = "default",
        env_var: str = "JAI_AUTH",
        verbose: int = 1,
        safe_mode: bool = False,
    ):
        super(LinearModel, self).__init__(
            name=name,
            auth_key=auth_key,
            environment=environment,
            env_var=env_var,
            verbose=verbose,
            safe_mode=safe_mode,
        )
        possible_tasks = [t.value for t in RegressionTasks] + [
            t.value for t in ClassificationTasks
        ]
        if task in possible_tasks:
            self.task = task
        else:
            str_possible = "`, `".join(possible_tasks)
            raise ValueError(
                f"Task `{self.task}` does not exist. Try one of `{str_possible}`."
            )
        self.set_parameters()

    @property
    def model_parameters(self):
        if self._model_parameters is None:
            raise ValueError(
                "No parameter was set, please use `set_parameters` method first."
            )
        return self._model_parameters

    @model_parameters.setter
    def model_parameters(self, value):
        self._model_parameters = value

    def set_parameters(
        self,
        learning_rate: Optional[float] = 0.01,
        l2: float = 0.0,
        scheduler_type: str = "constant",
        scheduler_argument: Optional[float] = None,
        model_parameters: dict = None,
    ):
        if self.task == RegressionTasks.regression:
            p = RegressionHyperparams(
                task=self.task,
                learning_rate=learning_rate,
                l2=l2,
                scheduler_type=scheduler_type,
                scheduler_argument=scheduler_argument,
                model_parameters=model_parameters,
            )
        elif self.task == RegressionTasks.sgd_regression:
            p = SGDRegressionHyperparams(
                task=self.task,
                learning_rate=learning_rate,
                l2=l2,
                scheduler_type=scheduler_type,
                scheduler_argument=scheduler_argument,
                model_parameters=model_parameters,
            )
        elif self.task == ClassificationTasks.classification:
            p = ClassificationHyperparams(
                task=self.task,
                learning_rate=learning_rate,
                l2=l2,
                scheduler_type=scheduler_type,
                scheduler_argument=scheduler_argument,
                model_parameters=model_parameters,
            )
        elif self.task == ClassificationTasks.sgd_classification:
            p = SGDClassificationHyperparams(
                task=self.task,
                learning_rate=learning_rate,
                l2=l2,
                scheduler_type=scheduler_type,
                scheduler_argument=scheduler_argument,
                model_parameters=model_parameters,
            )

        self._model_parameters = p.dict()

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        pretrained_bases: list = None,
        overwrite: bool = False,
        frequency_seconds: int = 1,
    ):
        """
        Train a new linear model.

        Args
        ----
          X: pd.DataFrame)
            dataframe of features.
          y: pd.Series):
            The target variable.
          pretrained_bases: list
            mapping of ids to previously trained databases.
          overwrite: bool
            If True, will overwrite the model if it already exists. Defaults to False.

        Returns
        -------
          A dictionary with information about the training.
          - id_train: List[Any]
          - id_test: List[Any]
          - metrics: Dict[str, Union[float, str]]
        """

        linear_response = self._linear_train(
            self.name,
            X.to_dict(orient="records"),
            y.tolist(),
            task=self.model_parameters["task"],
            learning_rate=self.model_parameters["learning_rate"],
            l2=self.model_parameters["l2"],
            scheduler_type=self.model_parameters["scheduler_type"],
            scheduler_argument=self.model_parameters["scheduler_argument"],
            model_parameters=self.model_parameters["model_parameters"],
            pretrained_bases=pretrained_bases,
            overwrite=overwrite,
        )

        if frequency_seconds < 1:
            return linear_response

        self.wait_setup(frequency_seconds=frequency_seconds)
        return linear_response

    def learn(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        learning_rate: Optional[float] = 0.01,
        l2: Optional[float] = 0.0,
        n_iterations: int = 1,
        scheduler_type: SchedulerType = SchedulerType.constant,
        scheduler_argument: Optional[float] = None,
        train_mode: TrainMode = TrainMode.always,
    ):
        """
        Improves an existing model with informantion from a new data.

        Args
        ----
        X: pd.DataFrame
            dataframe of features.
        y: pd.Series
            The target variable.

        Returns
        -------
        response: dict
            A dictionary with the learning report.
            - before: Dict[str, Union[float, str]]
            - after: Dict[str, Union[float, str]]
            - change: bool
        """
        return self._linear_learn(
            self.name,
            X.to_dict(orient="records"),
            y.tolist(),
            learning_rate=learning_rate,
            l2=l2,
            n_iterations=n_iterations,
            scheduler_type=scheduler_type,
            scheduler_argument=scheduler_argument,
            train_mode=train_mode,
        )

    def predict(
        self, X: pd.DataFrame, predict_proba: bool = False, as_frame: bool = True
    ):
        """
        Makes the prediction using the linear models.

        Args
        ----
        X: pd.DataFrame
            Raw data to be predicted.
        predict_proba:bool):
            If True, the model will return the probability of each class. Defaults to False.
        as_frame: bool
            If True, the result will be returned as a pandas DataFrame. If False, it will be
        returned as a list of dictionaries. Defaults to True.

        Returns
        -------
            A list of dictionaries.
        """
        result = self._linear_predict(
            self.name, X.to_dict(orient="records"), predict_proba=predict_proba
        )

        if as_frame:
            return pd.DataFrame(result).set_index("id")
        return result

    def get_model_weights(self):
        return self._get__linear_model_weights(self.name).json()

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
            PossibleDtypes.linear,
        ]:
            return None

        report = self._report(self.name, verbose)

        if return_report:
            return report
