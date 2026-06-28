from pydantic import BaseModel

from chap_3.policy_evalutation.exemple_generator.models.action import ActionModel


class StateModel(BaseModel):
    level: int | None = None
    type: str | None = None
    description: str | None = None
    actions: dict[str, ActionModel] | None = None