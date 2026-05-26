from typing import List, Optional

from pydantic import BaseModel


class SimulationRequirements(BaseModel):
    solver: Optional[str] = None
    time_grid: Optional[str] = None
    initial_conditions: List[str] = []
    scenarios: List[str] = []
    missing_information: List[str] = []
