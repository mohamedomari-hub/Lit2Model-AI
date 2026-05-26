from typing import Optional

from pydantic import BaseModel


class Observation(BaseModel):
    name: str
    description: Optional[str] = None
    unit: Optional[str] = None
    mapping: Optional[str] = None
    evidence: Optional[str] = None
    page: Optional[str] = None
