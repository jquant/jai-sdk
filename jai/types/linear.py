import sys
from enum import Enum
from typing import Dict, List, Optional, Union

from pydantic import BaseModel, Field

if sys.version < "3.8":
    from typing_extensions import Literal
else:
    from typing import Literal


class RegressionTasks(str, Enum):
    sgd_regression = "sgd_regression"
    regression = "regression"


class ClassificationTasks(str, Enum):
    sgd_classification = "sgd_classification"
    classification = "classification"


class RegressionMetrics(str, Enum):
    mae = "MAE"
    mse = "MSE"
    mape = "MAPE"
    r2_score = "R2_Score"


class ClassificationMetrics(str, Enum):
    report = "Report"


class LinearRegressionParams(BaseModel):
    fit_intercept: bool = True
    copy_X: bool = True
    n_jobs: Union[int, None] = None
    positive: bool = False


class SGDRegressorParams(BaseModel):
    loss: Literal[
        "squared_error", "huber", "epsilon_insensitive", "squared_epsilon_insensitive"
    ] = "squared_error"
    penalty: Literal["l2", "l1", "elasticnet"] = "l2"
    alpha: float = 0.0001
    l1_ratio: float = 0.15
    fit_intercept: bool = True
    max_iter: int = 1000
    tol: float = 0.001
    shuffle: bool = True
    # verbose: int = 0
    epsilon: float = 0.1
    random_state: Union[int, None] = None
    learning_rate: Literal["constant", "optimal", "invscaling", "adaptive"] = "optimal"
    eta0: float = 0.01
    power_t: float = 0.5
    early_stopping: bool = False
    validation_fraction: float = 0.1
    n_iter_no_change: int = 5
    # warm_start: bool = False
    average: Union[int, bool] = False


class LogisticRegressionParams(BaseModel):
    penalty: Literal["none", "l2", "l1", "elasticnet"] = "l2"
    dual: bool = False
    tol: float = 1e-4
    C: float = 1
    fit_intercept: bool = True
    intercept_scaling: float = 1
    # class_weight: Union[Dict, Literal['balanced']]= None
    random_state: Union[int, None] = None
    solver: Literal["newton-cg", "lbfgs", "liblinear", "sag", "saga"] = "lbfgs"
    max_iter: int = 100
    multi_class: Literal["auto", "ovr", "mutinomial"] = "auto"
    # verbose: int = 0
    # warm_start: bool = False
    n_jobs: Union[int, None] = None
    l1_ratio: Union[float, None] = None


class SGDClassifierParams(BaseModel):
    loss: Literal[
        "hinge",
        "log_loss",
        "log",
        "modified_huber",
        "squared_hinge",
        "perceptron",
        "squared_error",
        "huber",
        "epsilon_insensitive",
        "squared_epsilon_insensitive",
    ] = "hinge"
    penalty: Literal["l2", "l1", "elasticnet"] = "l2"
    alpha: float = 0.0001
    l1_ratio: float = 0.15
    fit_intercept: bool = True
    max_iter: int = 1000
    tol: float = 0.001
    shuffle: bool = True
    # verbose: int = 0
    epsilon: float = 0.1
    n_jobs: Union[int, None] = None
    random_state: Union[int, None] = None
    learning_rate: Literal["constant", "optimal", "invscaling", "adaptive"] = "optimal"
    eta0: float = 0.0
    power_t: float = 0.5
    early_stopping: bool = False
    validation_fraction: float = 0.1
    n_iter_no_change: int = 5
    # class_weight: Union[Dict, Literal['balanced']]= None
    # warm_start: bool = False
    average: Union[int, bool] = False


class LinearBase(BaseModel):
    learning_rate: Optional[float]
    l2: float


class RegressionHyperparams(LinearBase):
    task: Literal[RegressionTasks.regression]
    model_parameters: Optional[LinearRegressionParams] = LinearRegressionParams()


class SGDRegressionHyperparams(LinearBase):
    task: Literal[RegressionTasks.sgd_regression]
    model_parameters: Optional[SGDRegressorParams] = SGDRegressorParams()


class ClassificationHyperparams(LinearBase):
    task: Literal[ClassificationTasks.classification]
    model_parameters: Optional[LogisticRegressionParams] = LogisticRegressionParams()


class SGDClassificationHyperparams(LinearBase):
    task: Literal[ClassificationTasks.sgd_classification]
    model_parameters: Optional[SGDClassifierParams] = SGDClassifierParams()
