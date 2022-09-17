import pandas as pd

from ..types.linear import (
    ClassificationHyperparams,
    ClassificationTasks,
    RegressionHyperparams,
    RegressionTasks,
    SGDClassificationHyperparams,
    SGDRegressionHyperparams,
)
from .base import TaskBase

__all__ = ["LinearModel"]


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
        learning_rate: float = None,
        l2: float = 0.1,
        model_parameters: dict = None,
    ):
        if self.task == RegressionTasks.regression:
            p = RegressionHyperparams(
                task=self.task,
                learning_rate=learning_rate,
                l2=l2,
                model_parameters=model_parameters,
            )
        elif self.task == RegressionTasks.sgd_regression:
            p = SGDRegressionHyperparams(
                task=self.task,
                learning_rate=learning_rate,
                l2=l2,
                model_parameters=model_parameters,
            )
        elif self.task == ClassificationTasks.classification:
            p = ClassificationHyperparams(
                task=self.task,
                learning_rate=learning_rate,
                l2=l2,
                model_parameters=model_parameters,
            )
        elif self.task == ClassificationTasks.sgd_classification:
            p = SGDClassificationHyperparams(
                task=self.task,
                learning_rate=learning_rate,
                l2=l2,
                model_parameters=model_parameters,
            )

        self._model_parameters = p.dict()

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        pretrained_bases: list = None,
        overwrite: bool = False,
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
        return self._linear_train(
            self.name,
            X.to_dict(orient="records"),
            y.tolist(),
            task=self.model_parameters["task"],
            learning_rate=self.model_parameters["learning_rate"],
            l2=self.model_parameters["l2"],
            model_parameters=self.model_parameters["model_parameters"],
            pretrained_bases=pretrained_bases,
            overwrite=overwrite,
        )

    def learn(self, X: pd.DataFrame, y: pd.Series):
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
        return self._linear_learn(self.name, X.to_dict(orient="records"), y.tolist())

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
