from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class EquationEvidence(BaseModel):
    equation_id: str | None = None
    equation_type: Literal[
        "state_equation",
        "coupling_equation",
        "regulatory_function",
        "derived_definition",
        "unknown",
    ] = "unknown"
    raw_text: str
    variables: list[str] = Field(default_factory=list)
    page: int | None = None
    confidence: Literal["high", "medium", "low"] = "low"
    needs_ocr: bool = False
    review_note: str = ""


class ParameterEvidence(BaseModel):
    symbol: str | None = None
    name: str | None = None
    value: str | None = None
    unit: str | None = None
    meaning: str | None = None
    source: str | None = None
    page: int | None = None
    confidence: Literal["high", "medium", "low"] = "low"
    review_note: str = ""


class StateVariableEvidence(BaseModel):
    symbol: str | None = None
    name: str | None = None
    meaning: str | None = None
    unit: str | None = None
    page: int | None = None
    confidence: Literal["high", "medium", "low"] = "low"


class MechanismEvidence(BaseModel):
    source_entity: str | None = None
    relation: str | None = None
    target_entity: str | None = None
    evidence_text: str
    page: int | None = None
    confidence: Literal["high", "medium", "low"] = "low"
    review_note: str = ""


class TableEvidence(BaseModel):
    table_id: str | None = None
    caption_or_title: str | None = None
    purpose: str | None = None
    page: int | None = None
    likely_contains_parameters: bool = False
    needs_table_extraction: bool = True
    confidence: Literal["high", "medium", "low"] = "low"


class FigureEvidence(BaseModel):
    figure_id: str | None = None
    caption_or_title: str | None = None
    purpose: str | None = None
    page: int | None = None
    likely_contains_mechanism_graph: bool = False
    needs_vision: bool = True
    confidence: Literal["high", "medium", "low"] = "low"


class ObservationEvidence(BaseModel):
    observed_quantity: str | None = None
    description: str
    page: int | None = None
    confidence: Literal["high", "medium", "low"] = "low"


class ChunkEvidence(BaseModel):
    chunk_id: str
    page_start: int | None = None
    page_end: int | None = None

    equations: list[EquationEvidence] = Field(default_factory=list)
    parameters: list[ParameterEvidence] = Field(default_factory=list)
    state_variables: list[StateVariableEvidence] = Field(default_factory=list)
    mechanisms: list[MechanismEvidence] = Field(default_factory=list)
    tables: list[TableEvidence] = Field(default_factory=list)
    figures: list[FigureEvidence] = Field(default_factory=list)
    observations: list[ObservationEvidence] = Field(default_factory=list)

    missing_or_uncertain: list[str] = Field(default_factory=list)