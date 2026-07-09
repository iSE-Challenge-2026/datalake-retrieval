from pathlib import Path
import json
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, asdict, field
import uuid

@dataclass
class DSConfig:
    """Centralized configuration for the entire pipeline."""
    run_id: str = None
    query: Optional[str] = None
    data_sources_gt: Optional[List[str]] = None
    data_files: Optional[List[str]] = None
    data_dir: Optional[str] = None
    resume: Optional[str] = None
    edit_last: bool = False

    analysis_attempts: int = 3
    max_refinement_rounds: int = 3
    api_key: Optional[str] = None
    provider: str = "openrouter"
    model_name: str = None
    interactive: bool = False
    auto_debug: bool = True
    debug_attempts: float = float("inf")
    execution_timeout: int = 60
    preserve_artifacts: bool = True
    runs_dir: str = "runs"
    data_dir_default: str = "data"
    code_library_dir: str = "code_library"
    agent_models: Dict[str, str] = field(default_factory=dict)
    enable_direct_route: bool = True
    direct_context_char_limit: int = 60000
    direct_file_char_limit: int = 20000
    direct_attachment_limit: int = 20
    direct_attachment_max_bytes: int = 20000000

    def __post_init__(self):
        if self.resume:
            self.run_id = self.resume

        if self.run_id is None:
            self.run_id = (
                datetime.now().strftime("%Y%m%d_%H%M%S")
                + f"_{uuid.uuid4().hex[:6]}"
            )

        if self.agent_models is None:
            self.agent_models = {}
