import importlib
from dataclasses import dataclass
from typing import Any

import src.state.ui_parser

@dataclass
class Config:
    parser_module: Any = None

    def set_parser_module(self, parser_path):
        if parser_path is None or str(parser_path) == ".":
            self.parser_module = src.state.ui_parser
        else:
            spec = importlib.util.spec_from_file_location("uiparser", parser_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            self.parser_module = module
