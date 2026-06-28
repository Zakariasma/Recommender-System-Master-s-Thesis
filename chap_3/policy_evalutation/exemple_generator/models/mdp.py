from pydantic import BaseModel

from chap_3.policy_evalutation.exemple_generator.models.state import StateModel


class MDPModel(BaseModel):
    description: str
    initial_state: str
    rule_if_hole: str
    gamma: float
    states: dict[str, StateModel]