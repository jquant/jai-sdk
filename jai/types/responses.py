from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field


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
    name: str
    displayName: str
    owner: str
    project: str
    type: str
    version: str
    parents: List[str]


class InfoSizeResponse(BaseModel):
    name: str
    displayName: str
    owner: str
    project: str
    type: str
    version: str
    parents: List[str]
    size: int
    embedding_dimension: int


class DescribeResponse(BaseModel):
    name: str
    displayName: str
    owner: str
    project: str
    dtype: str
    state: str
    version: str
    has_filter: bool
    twin_base: Optional[str]
    dimension: Optional[int]
    features: List[Dict]  # TODO: future improvement
    model_hyperparams: Optional[Dict]  # TODO: future improvement
    trainer_hyperparams: Optional[Dict]  # TODO: future improvement


class FieldsDescription(BaseModel):
    name: str
    type: str


class FieldsResponse(BaseModel):
    database: str
    mapping: str
    fields: List[FieldsDescription]


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
    kwargs: Dict  # TODO: future improvement


class AddDataResponse(BaseModel):
    Task: str
    Status: str
    Description: str
    Interrupted: bool


class Report1Response(BaseModel):
    auth_batch_size: Optional[str] = Field(None, alias="Auto scale batch size")
    auth_lr_finder: Optional[str] = Field(None, alias="Auto lr finder")
    metrics_test: Optional[str] = Field(None, alias="Model Evaluation")
    thresholds: Optional[Dict[Any, float]] = Field(None, alias="Optimal Thresholds")
    baseline_models: Optional[str] = Field(None, alias="Baseline Model")
    loading_from_checkpoint: Optional[str] = Field(
        None, alias="Loading from checkpoint"
    )


class Report2Response(BaseModel):
    auth_batch_size: Optional[str] = Field(None, alias="Auto scale batch size")
    auth_lr_finder: Optional[str] = Field(None, alias="Auto lr finder")
    model_training: Dict[str, List[List[Any]]] = Field(..., alias="Model Training")
    metrics_train: Optional[str] = Field(None, alias="Metrics Train")
    metrics_val: Optional[str] = Field(None, alias="Metrics Validation")
    metrics_test: Optional[str] = Field(None, alias="Model Evaluation")
    thresholds: Optional[Dict[Any, float]] = Field(None, alias="Optimal Thresholds")
    baseline_models: Optional[str] = Field(None, alias="Baseline Model")
    loading_from_checkpoint: Optional[str] = Field(
        None, alias="Loading from checkpoint"
    )


class InsertVectorResponse(BaseModel):
    collection_name: str = Field(..., alias="Collection Name")
    vector_length: int = Field(..., alias="Vector Length")
    vector_dimension: int = Field(..., alias="Vector Dimension")
    message: str = Field(..., alias="Message")


class LinearFitResponse(BaseModel):
    id_train: List[Any] = Field(..., alias="Train Ids")
    id_test: List[Any] = Field(..., alias="Evaluation Ids")
    metrics: Dict[str, Union[float, str]] = Field(..., alias="Metrics")


class LinearLearnResponse(BaseModel):
    before: Dict[str, Union[float, str]]
    after: Dict[str, Union[float, str]]
    change: bool


class LinearPredictResponse(BaseModel):
    id: Any
    predict: Union[float, str]
