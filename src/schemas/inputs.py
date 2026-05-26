from typing import Optional

from pydantic import BaseModel


class ModelInput(BaseModel):
    name: str
    description: Optional[str] = None
    target_state: Optional[str] = None
    timing: Optional[str] = None
    magnitude: Optional[str] = None
    unit: Optional[str] = None
    evidence: Optional[str] = None
    page: Optional[str] = None
