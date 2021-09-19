from dataclasses import dataclass
from typing import Optional

@dataclass
class Config:
    batch_size: int
    optimizer: str
    loss_fn: str # ["mse"]
    policy: str #['categorical','boltzmann','epsilon_greedy', 'gumbel_softmax']
    learning_rate: float
    update_rate: float
    discount: float
    epsilon_decay: float
    lr_scheduler: Optional[str] #[null,"step_decay"]