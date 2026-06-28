from pydantic import BaseModel

from chap_3.policy_evalutation.exemple_generator.models.reward_prob import RewardProb


class Transition(BaseModel):
    prob: float
    next_state: str
    reward: int | None = None
    reward_distribution: list[RewardProb] | None = None