import yaml
import logging
from dacite import from_dict, Config as DaciteConfig
from dataclasses import dataclass
from typing import Optional
from pathlib import Path


from src.state.feature_extraction.extractor import FeatureExtractor
from .agent_config import Config as AgentConfig
from .path_config import Config as PathConfig
from .nn_config import Config as NNConfig
from .env_config import Config as EnvConfig
from .driver_config import Config as DriverConfig
from .externals_config import Config as ExternalsConfig 

logger = logging.getLogger(__name__)

@dataclass
class MainConfig:
    agent: Optional[AgentConfig] = None
    paths: Optional[PathConfig] = None
    nn: Optional[NNConfig] = None
    env: Optional[EnvConfig] = None
    webdriver: Optional[DriverConfig] = None
    externals: Optional[ExternalsConfig] = None # not populated from yaml

    def init_config(self,yaml_path):
        with open(yaml_path) as f:
            data = yaml.load(f, Loader=yaml.SafeLoader)
        new_config = from_dict(data_class=MainConfig, data=data,config=DaciteConfig(cast=[Path]))
        self.__dict__.update(new_config.__dict__)
        FeatureExtractor.set_config(self.env)
        base_path = "base" # TODO: Change
        self.paths.logs_dir = Path(base_path) / self.paths.logs_dir
        self.paths.model_dir = Path(base_path) / self.paths.model_dir
        self.paths.datagen = Path(base_path) / self.paths.datagen


    def set_paths(self, base_path=None,resume=False):
        if base_path is not None:
            self.paths.logs_dir = Path(base_path,*self.paths.logs_dir.parts[1:]) 
            self.paths.model_dir = Path(base_path,*self.paths.model_dir.parts[1:])
            self.paths.datagen = Path(base_path,*self.paths.datagen.parts[1:])
            logger.info(f'Current logs dir : {self.paths.logs_dir.resolve()}')
        self.paths.set_logs_path(resume)

        self.externals = ExternalsConfig()
        self.externals.set_parser_module(self.paths.ui_parser)



config = MainConfig()