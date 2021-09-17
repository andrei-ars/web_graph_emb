from dataclasses import dataclass
from typing import List

@dataclass
class Config:
    action_types: List[str]
    discretization: int
    pos_reward: int
    neg_reward: int
    use_fasttext: bool
    use_db: bool