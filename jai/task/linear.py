import pandas as pd

from .base import TaskBase
from ..types.linear import LinearHyperparams

__all__ = ["LinearModel"]


class LinearModel(TaskBase):
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
        super(LinearModel, self).__init__(name=name,
                                          environment=environment,
                                          env_var=env_var,
                                          verbose=verbose,
                                          safe_mode=safe_mode)
        self._setup_params = None

    @property
    def model_params(self):
        if self._model_params is None:
            raise ValueError(
                "Generic error message.")  #TODO: run set_params first message.
        return self._setup_params

    @model_params.setter
    def model_params(self, value):
        self._model_params = value

    def set_params(
        self,
        task,
        metric,
        learning_rate: float = None,
        l2: float = 0.1,
        model_params: dict = None,
        pretrained_bases: list = None,
    ):

        self._model_params = LinearHyperparams(
            task=task,
            metric=metric,
            learning_rate=learning_rate,
            l2=l2,
            model_params=model_params,
            pretrained_bases=pretrained_bases).dict()

        print(self.model_params)

    def fit(self, X: pd.DataFrame, y: pd.Series, overwrite: bool = False):
        self._linear_train(
            self.name,
            X,
            y,
            task=self.model_params['task'],
            metric=self.model_params['metric'],
            learning_rate=self.model_params['learning_rate'],
            l2=self.model_params['l2'],
            model_params=self.model_params['model_params'],
            pretrained_bases=self.model_params['pretrained_bases'],
            overwrite=overwrite)

    def learn(self, X: pd.DataFrame, y: pd.Series):
        return self._linear_learn(self.name, X, y)

    def predict(self, X: pd.DataFrame, predict_proba: bool = False):
        return self._linear_predict(self.name, X, predict_proba=predict_proba)
