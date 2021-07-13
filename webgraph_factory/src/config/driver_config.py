from dataclasses import dataclass
from typing import List

@dataclass
class Config:
    mock: bool
    dump_path: str
    browser: str # ["Chrome"]
    arguments: List[str] # without opening a browser window

