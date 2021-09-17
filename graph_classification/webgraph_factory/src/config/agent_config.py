from dataclasses import dataclass

@dataclass
class Config:
    total_episodes: int
    episodes_before_update: int
    max_steps: int
    solved_repeat: int