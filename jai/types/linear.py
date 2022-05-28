from typing import Dict, List, Literal, Union

from pydantic import BaseModel, Field
from typing_extensions import Annotated
from enum import Enum


class RegressionTasks(Enum):
    sgd_regression = "sgd_regression"
    regression = "regression"


class ClassificationTasks(Enum):
    sgd_classification = "sgd_classification"
    classification = "classification"


class RegressionMetrics(Enum):
    mae = "MAE"
    mse = "MSE"
    mape = "MAPE"
    r2_score = "R2_Score"


class ClassificationMetrics(Enum):
    report = "Report"


class LinearRegressionParams(BaseModel):
    fit_intercept: bool = True
    normalize: bool = False
    copy_X: bool = True
    n_jobs: Union[int, None] = None
    positive: bool = False


class SGDRegressorParams(BaseModel):
    loss: Literal["squared_error", "huber", "epsilon_insensitive",
                  "squared_epsilon_insensitive"] = "squared_error"
    penalty: Literal['l2', 'l1', 'elasticnet'] = 'l2'
    alpha: float = 0.0001
    l1_ratio: float = 0.15
    fit_intercept: bool = True
    max_iter: int = 1000
    tol: float = 0.001
    shuffle: bool = True
    # verbose: int = 0
    epsilon: float = 0.1
    random_state: Union[int, None] = None
    learning_rate: Literal["constant", "optimal", "invscaling",
                           "adaptive"] = 'optimal'
    eta0: float = 0.01
    power_t: float = 0.5
    early_stopping: bool = False
    validation_fraction: float = 0.1
    n_iter_no_change: int = 5
    # warm_start: bool = False
    average: Union[int, bool] = False


class LogisticRegressionParams(BaseModel):
    penalty: Literal['none', 'l2', 'l1', 'elasticnet'] = 'l2'
    dual: bool = False
    tol: float = 1e-4
    C: float = 1
    fit_intercept: bool = True
    intercept_scaling: float = 1
    # class_weight: Union[Dict, Literal['balanced']]= None
    random_state: Union[int, None] = None
    solver: Literal['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'] = "lbfgs"
    max_iter: int = 100
    multi_class: Literal['auto', 'ovr', 'mutinomial'] = 'auto'
    # verbose: int = 0
    # warm_start: bool = False
    n_jobs: Union[int, None] = None
    l1_ratio: Union[float, None] = None


class SGDClassifierParams(BaseModel):
    loss: Literal["hinge", "log_loss", "log", "modified_huber",
                  "squared_hinge", "perceptron", "squared_error", "huber",
                  "epsilon_insensitive",
                  "squared_epsilon_insensitive"] = "hinge"
    penalty: Literal['l2', 'l1', 'elasticnet'] = 'l2'
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
    learning_rate: Literal["constant", "optimal", "invscaling",
                           "adaptive"] = 'optimal'
    eta0: float = 0.0
    power_t: float = 0.5
    early_stopping: bool = False
    validation_fraction: float = 0.1
    n_iter_no_change: int = 5
    # class_weight: Union[Dict, Literal['balanced']]= None
    # warm_start: bool = False
    average: Union[int, bool] = False


class LinearBase(BaseModel):
    learning_rate: float
    l2: float


class RegressionHyperparams(LinearBase):
    task: Literal[RegressionTasks.regression]
    metric: RegressionMetrics = RegressionMetrics.mse
    model: LinearRegressionParams = LinearRegressionParams()


class SGDRegressionHyperparams(LinearBase):
    task: Literal[RegressionTasks.sgd_regression]
    metric: RegressionMetrics = RegressionMetrics.mse
    model: SGDRegressorParams = SGDRegressorParams()


class ClassificationHyperparams(LinearBase):
    task: Literal[ClassificationTasks.classification]
    metric: ClassificationMetrics = ClassificationMetrics.report
    model: LogisticRegressionParams = LogisticRegressionParams()


class SGDClassificationHyperparams(LinearBase):
    task: Literal[ClassificationTasks.sgd_classification]
    metric: ClassificationMetrics = ClassificationMetrics.report
    model: SGDClassifierParams = SGDClassifierParams()


LinearHyperparams = Annotated[Union[RegressionHyperparams,
                                    SGDRegressionHyperparams,
                                    ClassificationHyperparams,
                                    SGDClassificationHyperparams],
                              Field(discriminator='task')  # noqa: F821
                              ]
