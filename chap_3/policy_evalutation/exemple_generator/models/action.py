from pydantic import BaseModel

from chap_3.policy_evalutation.exemple_generator.models.transition import Transition


class ActionModel(BaseModel):
    safe_transition: Transition
    hole_transition: Transition