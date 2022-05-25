from pydantic import BaseModel, Field
from typing import Any, Optional, Dict, List, Literal, Union


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
