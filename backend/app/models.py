from pydantic import BaseModel, Field
from typing import List

class TimeSeriesData(BaseModel):
    t: int
    val: float
    raw: float


class GroupData(BaseModel):
    iid: int
    label: str
    ts_data: list[TimeSeriesData]

class ClusterData(BaseModel):
    id: int
    x: int
    y: int

class CounterData(BaseModel):
    start: int
    end: int


class ShapeData(BaseModel):
    id: int
    vals: List[float]
    len: int
    gain: float

class MatchUnit(BaseModel):
    s: int
    e: int
    dist: float

class TimeSeriesReturnModel(BaseModel):
    
    data: List[GroupData]

class ClusterReturnModel(BaseModel):
    clusters: List[ClusterData]
    belongings: List[int]

class ShapeReturnModel(BaseModel):
    shapes: List[ShapeData]

class TransformReturnModel(BaseModel):
    data: List[List[float]] = Field(
        ...,
        description="""A 2D matrix with shape [num_instances, num_shapelets] indicating the distance from each instance to each shapelet. The value is normalized to [0, 1] by min-max normalization.
        """,
    )
    max: float = Field(..., description="The maximum value in the matrix (before normalization).")
    min: float = Field(..., description="The minimum value in the matrix (before normalization).")

class MatchingLocation(BaseModel):
    data: List[MatchUnit]