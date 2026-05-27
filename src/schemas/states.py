"""
Schema for model state variables.
"""

from typing import Optional

from pydantic import BaseModel


class StateVariable(BaseModel):
    name: str
    description: Optional[str] = None
    unit: Optional[str] = None
    evidence: Optional[str] = None
    page: Optional[str] = None
