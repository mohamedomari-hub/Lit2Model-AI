"""
Schema for model mechanisms and relationships.
"""

from typing import Optional

from pydantic import BaseModel


class Mechanism(BaseModel):
    source: str
    target: str
    effect: str
    evidence: Optional[str] = None
    page: Optional[str] = None
