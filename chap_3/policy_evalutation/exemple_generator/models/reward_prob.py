from pydantic import BaseModel


class RewardProb(BaseModel):
    reward: int
    prob: float