from dataclasses import dataclass
from typing import Optional
from pathlib import Path
from datetime import datetime

@dataclass
class Config:
    model_dir: Path
    logs_dir: Path
    targets: Path
    datagen: Path
    ui_parser: Optional[Path] 

    def set_logs_path(self,resume=False):
        if resume:
            self.logs_dir = sorted(list(self.logs_dir.iterdir()))[-1]
        else:
            now = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            self.logs_dir = self.logs_dir / f'{now}/'
            self.logs_dir.mkdir(parents=True, exist_ok=True)

        model_dir = self.model_dir 
        model_dir.mkdir(parents=True, exist_ok=True)