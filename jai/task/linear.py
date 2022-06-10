import pandas as pd
from typing import Any, Dict, List
from .base import TaskBase
from ..types.linear import (
    RegressionTasks,
    ClassificationTasks,
    RegressionHyperparams,
    SGDRegressionHyperparams,
    ClassificationHyperparams,
    SGDClassificationHyperparams,
)
from ..types.responses import (
    LinearPredictResponse,
    LinearFitResponse,
    LinearLearnResponse,
)
from ..core.validations import check_response

__all__ = ["LinearModel"]


class LinearModel(TaskBase):
    """
    Base class for communication with the Mycelia API.

    Used as foundation for more complex applications for data validation such
    as matching tables, resolution of duplicated values, filling missing values
    and more.

    """

    def __init__(
        self,
        name: str,
        task: str,
        environment: str = "default",
        env_var: str = "JAI_AUTH",
        verbose: int = 1,
        safe_mode: bool = False,
    ):
        """
        Initialize the Jai class.

        An authorization key is needed to use the Mycelia API.

        Parameters
        ----------

        Returns
        -------
            None

        """
        super(LinearModel, self).__init__(
            name=name,
            environment=environment,
            env_var=env_var,
            verbose=verbose,
            safe_mode=safe_mode,
        )

        self.task = task
        self.set_parameters()

    @property
    def model_parameters(self):
        if self._model_parameters is None:
            raise ValueError(
                "Generic error message."
            )  # TODO: run set_parameters first message.
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
        else:
            raise ValueError(
                "This task does not exist message."
            )  # TODO: rewrite message

        self._model_parameters = p.dict()

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        pretrained_bases: list = None,
        overwrite: bool = False,
    ):
        response = self._linear_train(
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

        if self.safe_mode:
            return check_response(LinearFitResponse, response).dict(by_alias=True)
        return response

    def learn(self, X: pd.DataFrame, y: pd.Series):
        response = self._linear_learn(
            self.name, X.to_dict(orient="records"), y.tolist()
        )

        if self.safe_mode:
            return check_response(LinearLearnResponse, response).dict()

        return response

    def predict(
        self, X: pd.DataFrame, predict_proba: bool = False, as_frame: bool = True
    ):

        result = self._linear_predict(
            self.name, X.to_dict(orient="records"), predict_proba=predict_proba
        )
        if self.safe_mode:
            if predict_proba:
                result = check_response(List[Dict[Any, Any]], result)
            else:
                result = check_response(LinearPredictResponse, result, list_of=True)

        if as_frame:
            return pd.DataFrame(result).set_index("id")
        return result
