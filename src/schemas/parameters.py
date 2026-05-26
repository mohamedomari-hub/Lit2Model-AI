from typing import Optional

from pydantic import BaseModel


class Parameter(BaseModel):
    name: str
    description: Optional[str] = None
    value: Optional[str] = None
    unit: Optional[str] = None
    status: Optional[str] = None
    evidence: Optional[str] = None
    page: Optional[str] = None
