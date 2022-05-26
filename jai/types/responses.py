from pydantic import BaseModel, Field
from typing import Any, Optional, Dict, List, Union
from enum import Enum
import sys

if sys.version < '3.8':
    from typing_extensions import Literal
else:
    from typing import Literal


class UserResponse(BaseModel):
    userId: str
    email: str
    firstName: str
    lastName: str
    memberRole: str
    namespace: Union[None, str]


class EnvironmentsResponse(BaseModel):
    key: Optional[str]
    id: str
    name: str


class StatusResponse(BaseModel):
    Task: str
    Status: str
    Description: str
    Interrupted: bool
    CurrentStep: int
    TotalSteps: int


class InfoResponse(BaseModel):
    db_name: str
    db_type: str
    db_version: str
    db_parents: List[str]


class InfoSizeResponse(BaseModel):
    db_name: str
    db_type: str
    db_version: str
    db_parents: List[str]
    size: int
    embedding_dimension: int


class FieldsResponse(BaseModel):
    key: Optional[str]
    id: str
    name: str


class ResultItem(BaseModel):
    id: Any
    distance: float


class QueryItem(BaseModel):
    query_id: Any
    results: List[ResultItem]


class SimilarNestedResponse(BaseModel):
    similarity: List[QueryItem]


class RecNestedResponse(BaseModel):
    recommendation: List[QueryItem]


class FlatResponse(BaseModel):
    query_id: Any
    id: Any
    distance: float


class PredictResponse(BaseModel):
    id: Any
    predict: Union[int, float, str, Dict[Any, float]]


class ValidResponse(BaseModel):
    value: bool
    message: str


class InsertDataResponse(BaseModel):
    Task: str
    Status: str
    Description: str
    Interrupted: bool


class SetupResponse(BaseModel):
    Task: str
    Status: str
    Description: str
    kwargs: Dict


class ReportResponse(BaseModel):
    train_ids: List[Any] = Field(..., alias="Train Ids")
    val_ids: List[Any] = Field(..., alias="Validation Ids")
    test_ids: Optional[List[Any]] = Field(..., alias="Evaluation Ids")
    auth_batch_size: Optional[str] = Field(..., alias="Auto scale batch size")
    auth_lr_finder: Optional[str] = Field(..., alias="Auto lr finder")
    model_training: Dict[str, List[List[Any]]] = Field(...,
                                                       alias="Model Training")
    metrics_train: Optional[str] = Field(..., alias="Metrics Train")
    metrics_val: Optional[str] = Field(..., alias="Metrics Validation")
    metrics_test: Optional[str] = Field(..., alias="Model Evaluation")
    thresholds: Optional[Dict[Any, float]] = Field(...,
                                                   alias="Optimal Thresholds")
    baseline_models: Optional[str] = Field(..., alias="Baseline Model")
