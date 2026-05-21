from pydantic import BaseModel
from typing import List, Optional


class StateVariable(BaseModel):
    name: str
    description: Optional[str] = None
    unit: Optional[str] = None
    evidence: Optional[str] = None
    page: Optional[str] = None


class Parameter(BaseModel):
    name: str
    description: Optional[str] = None
    value: Optional[str] = None
    unit: Optional[str] = None
    status: Optional[str] = None
    evidence: Optional[str] = None
    page: Optional[str] = None


class Mechanism(BaseModel):
    source: str
    target: str
    effect: str
    evidence: Optional[str] = None
    page: Optional[str] = None


class Equation(BaseModel):
    name: Optional[str] = None
    equation: str
    description: Optional[str] = None
    evidence: Optional[str] = None
    page: Optional[str] = None


class ExtractedModelInfo(BaseModel):
    state_variables: List[StateVariable]
    parameters: List[Parameter]
    mechanisms: List[Mechanism]
    equations: List[Equation]
    assumptions: List[str]
    missing_information: List[str]
    candidate_model_description: str