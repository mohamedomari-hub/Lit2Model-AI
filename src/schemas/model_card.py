"""
Schema for the full extracted model card.
"""

from typing import List

from pydantic import BaseModel

from src.schemas.equations import Equation
from src.schemas.mechanisms import Mechanism
from src.schemas.parameters import Parameter
from src.schemas.states import StateVariable


class ExtractedModelInfo(BaseModel):
    state_variables: List[StateVariable]
    parameters: List[Parameter]
    mechanisms: List[Mechanism]
    equations: List[Equation]
    assumptions: List[str]
    missing_information: List[str]
    candidate_model_description: str
