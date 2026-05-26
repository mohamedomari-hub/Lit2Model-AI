from typing import Optional

from pydantic import BaseModel


class Equation(BaseModel):
    name: Optional[str] = None
    equation: str
    description: Optional[str] = None
    evidence: Optional[str] = None
    page: Optional[str] = None
