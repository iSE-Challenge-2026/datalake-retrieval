import os
import json
import subprocess
import re
import uuid
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path
import sys
import atexit
import time
import yaml
from datetime import datetime
from cache import GlobalCacheManager
from config import DSConfig
from provider import GeminiProvider, OpenAIProvider, OpenRouterProvider

# =============================================================================
# CONFIGURATION & PROMPT TEMPLATES
# =============================================================================

with open("./config/prompt.yaml", "r") as f:
    PROMPT_TEMPLATES = yaml.safe_load(f)

# =============================================================================
# ARTIFACT STORAGE SYSTEM
# =============================================================================

class ArtifactStorage:
    """Persistently stores every step of the pipeline for reproducibility."""
    
    def __init__(self, config: DSConfig):
        self.config = config
        self.run_dir = Path(config.runs_dir) / config.run_id
        self._setup_directories()
        
    def _setup_directories(self):
        """Create directory structure for this run."""
        dirs = [
            self.run_dir,
            self.run_dir / "steps",
            self.run_dir / "data_cache",
            self.run_dir / "logs",
            self.run_dir / "final_output"
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)
            
    def save_step(self, step_id: str, step_type: str, prompt: str, 
                  code: Optional[str], result: str, metadata: Dict[str, Any]):
        """Save all artifacts for a single pipeline step."""
        step_dir = self.run_dir / "steps" / step_id
        step_dir.mkdir(exist_ok=True)
        
        # Save prompt
        (step_dir / "prompt.md").write_text(prompt, encoding='utf-8')
        
        # Save generated code
        if code:
            (step_dir / "code.py").write_text(code, encoding='utf-8')
            
        # Save execution result
        (step_dir / "result.txt").write_text(result, encoding='utf-8')
        
        # Save metadata
        metadata.update({
            "timestamp": datetime.now().isoformat(),
            "step_type": step_type,
            "step_id": step_id
        })
        with open(step_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
            
    def get_step(self, step_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a previous step's artifacts."""
        step_dirs = list(self.run_dir.glob(f"steps/{step_id}_*"))
        if not step_dirs:
            return None
        
        step_dir = step_dirs[0]
        return {
            "prompt": (step_dir / "prompt.md").read_text(encoding='utf-8'),
            "code": (step_dir / "code.py").read_text(encoding='utf-8') 
                   if (step_dir / "code.py").exists() else None,
            "result": (step_dir / "result.txt").read_text(encoding='utf-8'),
            "metadata": json.loads((step_dir / "metadata.json").read_text())
        }
    
    def list_steps(self) -> List[Dict[str, Any]]:
        """List all steps in chronological order."""
        steps = []
        for step_path in sorted(self.run_dir.glob("steps/*")):
            with open(step_path / "metadata.json") as f:
                metadata = json.load(f)
                steps.append(metadata)
        return steps
    
    def get_current_state(self) -> Dict[str, Any]:
        """Load the pipeline state."""
        state_file = self.run_dir / "pipeline_state.json"
        if state_file.exists():
            return json.loads(state_file.read_text())
        return {"current_step": 0, "completed_steps": [], "plan": [], "data_descriptions": {}}
    
    def save_state(self, state: Dict[str, Any]):
        """Save the pipeline state."""
        state_file = self.run_dir / "pipeline_state.json"
        state_file.write_text(json.dumps(state, indent=2))

# =============================================================================
# STATE MANAGEMENT & EXECUTION CONTROL
# =============================================================================

class PipelineController:
    """Manages pipeline execution with resume and editing capabilities."""
    
    def __init__(self, config: DSConfig, storage: ArtifactStorage, agent: 'Agent'):
        self.config = config
        self.storage = storage
        self.agent = agent
        self.logger = self._setup_logger()
        
    def _setup_logger(self):
        """Setup structured logging."""
        import logging
        log_file = self.storage.run_dir / "logs" / "pipeline.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        return logging.getLogger(__name__)
    
    def should_execute_step(self, step_index: int) -> bool:
        """Check if step should be executed (for resuming)."""
        state = self.storage.get_current_state()
        return step_index >= state["current_step"]
    
    def execute_step(self, step_name: str, step_func, **kwargs) -> Any:
        """Execute a single step with full artifact preservation."""
        step_id = f"{self._get_next_step_index():03d}_{step_name}"
        
        self.logger.info(f"{'='*50}")
        self.logger.info(f"STEP {step_id}")
        self.logger.info(f"{'='*50}")
        
        # Execute the step
        result = step_func(**kwargs)
        
        # Save artifacts
        if self.config.preserve_artifacts:
            metadata = kwargs.copy()
            code = result.get("code") if isinstance(result, dict) else None
            self.storage.save_step(
                step_id=step_id,
                step_type=step_name,
                prompt=kwargs.get("prompt", ""),
                code=code,
                result=str(result.get("result") if isinstance(result, dict) else result),
                metadata=metadata
            )
        
        # Update state
        state = self.storage.get_current_state()
        state["current_step"] = self._get_next_step_index()
        state["completed_steps"].append(step_id)
        self.storage.save_state(state)
        
        # Interactive mode
        if self.config.interactive:
            input(f"Step {step_id} complete. Press Enter to continue...")
            
        return result
    
    def _get_next_step_index(self) -> int:
        """Get the next step index."""
        steps = self.storage.list_steps()
        return len(steps)
    
    def edit_last_step_code(self):
        """Allow manual editing of the last generated code."""
        steps = self.storage.list_steps()
        if not steps:
            return
        
        last_step = steps[-1]
        step_dir = self.storage.run_dir / "steps" / f"{last_step['step_id']}_{last_step['step_type']}"
        code_file = step_dir / "code.py"
        
        if code_file.exists():
            self.logger.info(f"Opening {code_file} for editing...")
            # Use system editor
            editor = os.environ.get('EDITOR', 'nano')
            subprocess.run([editor, str(code_file)])
            self.logger.info("Code updated. Re-executing step...")
            # Re-execute the modified code
            code = code_file.read_text()
            result, error = self.agent._execute_code(code)
            (step_dir / "result.txt").write_text(result)
            (step_dir / "metadata.json").write_text(json.dumps({**last_step, "edited": True}, indent=2))

# =============================================================================
# CORE AGENT (Refactored)
# =============================================================================



class Agent:
    """DS-STAR agent with persistent artifact storage."""

    def __init__(self, config: DSConfig):
        self.config = config
        self.storage = ArtifactStorage(config)
        self.controller = PipelineController(config, self.storage, self)
        self.providers = {}
        canonical_cache_path = Path(__file__).resolve().parents[1] / "data" / "output" / "Data-Lake" / "canonical_context_cache.json"
        self.global_cache = GlobalCacheManager(str(canonical_cache_path))
        self._canonical_context_indexes: Optional[Dict[str, Dict[str, List[dict]]]] = None
        
        default_model = config.model_name
        agents = [
            "ANALYZER", "PLANNER", "CODER", "VERIFIER", "ROUTER",
            "DIRECT", "DEBUGGER", "FINALYZER", "ANSWER_FORMATTER"
        ]
        
        def get_provider_for_model(model_name: str, config: DSConfig) -> Any:
            provider_cls = None
            for provider in [OpenRouterProvider, OpenAIProvider, GeminiProvider]:
                if provider.provider_instance(model_name):
                    provider_cls = provider
                    break
            if not provider_cls:
                raise ValueError(f"No provider found for model {model_name}")
            return provider_cls(config.api_key, model_name)

        for agent in agents:
            model_name = config.agent_models.get(agent, default_model)
            self.providers[agent] = get_provider_for_model(model_name, config)
            self.controller.logger.info(f"Initialized {agent} with model: {model_name}")
        
        self.exec_dir = Path(config.runs_dir) / config.run_id / "exec_env"
        self.exec_dir.mkdir(exist_ok=True)
        self._setup_tee_logging()
        
    def _setup_tee_logging(self):
        log_path = self.storage.run_dir / "logs" / "execution.log"
        self.log_file = open(log_path, "a", encoding="utf-8")
    
        self._old_stdout = sys.stdout
        self._old_stderr = sys.stderr
    
        class _Tee:
            def __init__(self, *writers):
                self.writers = writers
    
            def write(self, data):
                for w in self.writers:
                    try:
                        w.write(data)
                        w.flush()
                    except Exception:
                        pass
    
            def flush(self):
                for w in self.writers:
                    try:
                        w.flush()
                    except Exception:
                        pass
    
        sys.stdout = _Tee(self._old_stdout, self.log_file)
        sys.stderr = _Tee(self._old_stderr, self.log_file)
    
    
    def restore_logging(self):
        if hasattr(self, "_old_stdout"):
            sys.stdout = self._old_stdout
        if hasattr(self, "_old_stderr"):
            sys.stderr = self._old_stderr
        if hasattr(self, "log_file") and not self.log_file.closed:
            self.log_file.close()
    
    def _call_model(self, agent_name: str, prompt: str, file_paths: Optional[List[str]] = None) -> str:
        """Call the appropriate model provider for the agent."""
        try:
            provider = self.providers[agent_name]
            
            start_time = time.time()
            try:
                res_dict = provider.generate_content(prompt, file_paths=file_paths)
            except Exception as e:
                if not file_paths:
                    raise
                self.controller.logger.warning(
                    f"[{agent_name}] File-aware call failed ({e}); retrying with prompt context only."
                )
                res_dict = provider.generate_content(prompt)
            end_time = time.time()
            
            run_time_ms = int((end_time - start_time) * 1000)
            
            response_text = res_dict.get("text", "") if isinstance(res_dict, dict) else res_dict
            usage = res_dict.get("usage", {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}) if isinstance(res_dict, dict) else {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
            
            self.controller.current_step_usage = usage
            self.controller.current_step_usage["run_time_ms"] = run_time_ms
            
            self.controller.logger.info(f"[{agent_name}] Response received ({len(response_text)} chars)")
            return response_text
        except Exception as e:
            error_msg = f"Error calling model for {agent_name}: {str(e)}"
            self.controller.logger.error(error_msg)
            raise
    
    def _extract_code_block(self, response: str) -> str:
        """Extract Python code from markdown blocks."""
        triple_backticks = '`' * 3
        pattern = re.escape(triple_backticks) + r'(?:python)?\n(.*?)\n' + re.escape(triple_backticks)
        code_blocks = re.findall(pattern, response, re.DOTALL)
        return code_blocks[0] if code_blocks else response.strip()

    def _extract_direct_final_answer(self, response: str) -> str:
        """Extract the final answer from the direct-answer response format."""
        text = response.strip()
        xml_match = re.search(r"(?is)<final_answer>\s*(.*?)\s*</final_answer>", text)
        if xml_match:
            return xml_match.group(1).strip()

        label_match = re.search(r"(?is)(?:^|\n)\s*FINAL_ANSWER\s*:\s*(.+?)\s*$", text)
        if label_match:
            return label_match.group(1).strip()

        return text

    def _format_file_manifest(self, data_files: List[str]) -> str:
        if not data_files:
            return "No files were retrieved."

        lines = []
        for idx, file_path in enumerate(data_files, start=1):
            path = Path(file_path)
            if path.exists():
                lines.append(
                    f"{idx}. {path.name} | path={path} | extension={path.suffix or '(none)'} "
                )
            else:
                lines.append(f"{idx}. {file_path} | missing")
        return "\n".join(lines)

    def _canonical_context_root(self) -> Path:
        return Path(__file__).resolve().parents[1] / "data" / "processed" / "Data-Lake" / "canonical"

    def _canonical_source_keys(self, file_path: str) -> List[str]:
        path = Path(str(file_path).replace("\\", "/"))
        keys: List[str] = []

        normalized = path.as_posix().lstrip("./")
        if normalized:
            keys.append(normalized)

        parts = path.parts
        if "Data-Lake" in parts:
            data_lake_index = parts.index("Data-Lake")
            relative_path = Path(*parts[data_lake_index + 1 :]).as_posix()
            if relative_path:
                keys.append(relative_path)

        if path.name:
            keys.append(path.name)

        deduped: List[str] = []
        seen = set()
        for key in keys:
            if key in seen:
                continue
            seen.add(key)
            deduped.append(key)
        return deduped

    def _load_jsonl_records(self, jsonl_path: Path) -> List[dict]:
        if not jsonl_path.exists():
            return []

        records: List[dict] = []
        with jsonl_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(record, dict):
                    records.append(record)
        return records

    def _load_canonical_context_indexes(self) -> Dict[str, Dict[str, List[dict]]]:
        if self._canonical_context_indexes is not None:
            return self._canonical_context_indexes

        canonical_root = self._canonical_context_root()
        index: Dict[str, Dict[str, List[dict]]] = {"text": {}, "table": {}, "image": {}}
        source_files = {
            "text": canonical_root / "texts.jsonl",
            "table": canonical_root / "tables.jsonl",
            "image": canonical_root / "images.jsonl",
        }

        for kind, jsonl_path in source_files.items():
            for record in self._load_jsonl_records(jsonl_path):
                source_path = str(record.get("source_path", "")).strip()
                if not source_path:
                    continue
                for key in self._canonical_source_keys(source_path):
                    index[kind].setdefault(key, []).append(record)

        self._canonical_context_indexes = index
        return index

    def _truncate_text(self, value: Optional[str], limit: int) -> str:
        text = (value or "").strip()
        if len(text) <= limit:
            return text
        return text[: max(0, limit - 20)].rstrip() + "\n[... truncated ...]"

    def _render_canonical_record(self, kind: str, record: dict) -> str:
        lines: List[str] = []
        source_path = record.get("source_path") or record.get("table_path") or record.get("image_path") or ""
        # if source_path:
        #     lines.append(f"Source path: {source_path}")
        if record.get("role"):
            lines.append(f"Role: {record['role']}")
        if record.get("parser"):
            lines.append(f"Parser: {record['parser']}")

        if kind == "text":
            text_value = record.get("description") or record.get("text") or record.get("llm_description") or ""
            if text_value:
                lines.append("Text context:")
                lines.append(self._truncate_text(str(text_value), self.config.direct_file_char_limit))
        elif kind == "table":
            description = record.get("description")
            preview_text = record.get("preview_text")
            llm_description = record.get("llm_description")
            columns = record.get("columns") or []
            sample_rows = record.get("sample_rows") or []
            tail_rows = record.get("tail_sample_rows") or []

            if description:
                lines.append("Description:")
                lines.append(self._truncate_text(str(description), self.config.direct_file_char_limit))
            if preview_text:
                lines.append("Preview:")
                lines.append(self._truncate_text(str(preview_text), self.config.direct_file_char_limit))
            if columns:
                lines.append("Columns:")
                lines.append(", ".join(str(column) for column in columns))
            if sample_rows:
                lines.append("Sample rows:")
                lines.append(self._truncate_text(json.dumps(sample_rows[:3], ensure_ascii=False, indent=2), self.config.direct_file_char_limit))
            if tail_rows and not sample_rows:
                lines.append("Tail sample rows:")
                lines.append(self._truncate_text(json.dumps(tail_rows[:3], ensure_ascii=False, indent=2), self.config.direct_file_char_limit))
            if llm_description:
                lines.append("Enhanced description:")
                lines.append(self._truncate_text(str(llm_description), self.config.direct_file_char_limit))
        elif kind == "image":
            for label, field_name in (
                ("Description", "description"),
                ("Text", "text"),
                ("OCR text", "ocr_text"),
                ("Caption", "caption"),
                ("Enhanced description", "llm_description"),
            ):
                value = record.get(field_name)
                if value:
                    lines.append(f"{label}:")
                    lines.append(self._truncate_text(str(value), self.config.direct_file_char_limit))

            metadata = record.get("metadata")
            # if metadata:
            #     lines.append("Metadata:")
            #     lines.append(self._truncate_text(json.dumps(metadata, ensure_ascii=False, indent=2), self.config.direct_file_char_limit))

        return "\n".join(line for line in lines if line)

    def _lookup_canonical_context(self, file_path: str) -> str:
        indexes = self._load_canonical_context_indexes()
        lookup_keys = self._canonical_source_keys(file_path)
        sections: List[str] = []

        for kind in ("text", "table", "image"):
            matched_records: List[dict] = []
            seen_signatures = set()
            for key in lookup_keys:
                for record in indexes[kind].get(key, []):
                    signature = json.dumps(record, sort_keys=True, ensure_ascii=False)
                    if signature in seen_signatures:
                        continue
                    seen_signatures.add(signature)
                    matched_records.append(record)

            if not matched_records:
                continue

            rendered_records = [self._render_canonical_record(kind, record) for record in matched_records]
            sections.append(f"[{kind.upper()} CONTEXT]\n" + "\n\n".join(rendered_records))

        if not sections:
            return f"No canonical context found for {file_path}."

        return "\n\n\n".join(sections)

    def _build_direct_file_context(self, query: str, data_files: List[str]) -> str:
        if not data_files:
            return "No files were retrieved."

        remaining = self.config.direct_context_char_limit
        sections: List[str] = []

        for file_path in data_files:
            if remaining <= 0:
                sections.append("[Context truncated because the configured direct context limit was reached.]")
                break

            canonical_context = self._lookup_canonical_context(file_path)
            header = f"Source file: {file_path}"
            section = f"{header}\n{canonical_context}"
            section = section[:remaining]
            sections.append(section)
            remaining -= len(section)

        return "\n\n-----\n\n".join(sections)

    def _direct_attachment_paths(self, data_files: List[str]) -> List[str]:
        return []

    def decide_answer_route(self, query: str, data_files: List[str]) -> str:
        if not self.config.enable_direct_route:
            return "CODE"

        file_context = self._build_direct_file_context(query, data_files)
        prompt = PROMPT_TEMPLATES["branch_router"].format(
            question=query,
            file_context=file_context
        )
        decision = self.controller.execute_step(
            "answer_router",
            step_func=lambda prompt=prompt, **kwargs: self._call_model("ROUTER", prompt),
            prompt=prompt,
            file_count=len(data_files)
        )

        match = re.search(r"\b(DIRECT|CODE)\b", decision.strip().upper())
        if match:
            route = match.group(1)
        else:
            route = "DIRECT" if not data_files else "CODE"
            self.controller.logger.warning(
                f"Could not parse route decision {decision!r}; falling back to {route}."
            )

        self.controller.logger.info(f"Answer route selected: {route}")
        return route

    def direct_answer(self, query: str, data_files: List[str]) -> Dict[str, str]:
        file_context = self._build_direct_file_context(query, data_files)
        prompt = PROMPT_TEMPLATES["direct_answer"].format(
            question=query,
            file_context=file_context
        )
        result = self.controller.execute_step(
            "direct_answer",
            step_func=lambda prompt=prompt, **kwargs: self._call_model("DIRECT", prompt),
            prompt=prompt,
            file_count=len(data_files),
            attached_files=[]
        )
        raw_result = result.strip()
        return {
            "raw_result": raw_result,
            "final_answer": self._extract_direct_final_answer(raw_result)
        }
    
    def _execute_code(self, code_script: str, data_files: Optional[List[str]] = None) -> Tuple[str, Optional[str]]:
        """Execute code in isolated environment."""
        self.controller.logger.info("Executing code...")
        if data_files:
            missing = []
            for f in data_files:
                p = Path(f)
                if p.is_absolute():
                    if not p.exists(): missing.append(f)
                else:
                    if not (Path(self.config.data_dir) / f).exists(): missing.append(f)
            if missing:
                return "", f"Missing data files: {missing}"
        
        exec_id = uuid.uuid4().hex[:8]
        exec_path = self.exec_dir / f"exec_{exec_id}.py"
        exec_path.write_text(code_script, encoding='utf-8')
        
        try:
            result = subprocess.run(
                [sys.executable, str(exec_path)],
                capture_output=True,
                text=True,
                timeout=self.config.execution_timeout,
                cwd=Path.cwd()
            )
            if result.returncode == 0:
                self.controller.logger.info("Execution successful")
                return result.stdout, None
            else:
                error_msg = result.stderr or "Unknown execution error"
                self.controller.logger.error(f"Execution failed: {error_msg}")
                return "", error_msg
        except subprocess.TimeoutExpired:
            return "", f"Timeout after {self.config.execution_timeout}s"
        except Exception as e:
            return "", f"Execution error: {str(e)}"

    def _execute_and_debug_code(self, code: str, data_files: List[str], data_desc: str) -> Tuple[str, str]:
        exec_result, error = self._execute_code(code, data_files)
        attempts = 0
        while error and self.config.auto_debug and attempts < self.config.debug_attempts:
            self.controller.logger.warning("Debugging...")
            code = self._debug_code(code, error, data_desc, data_files)
            exec_result, error = self._execute_code(code, data_files)
            attempts += 1
        if error:
            self.controller.logger.fatal(f"Execution error: {error}")
        return code, exec_result

    def analyze_data(self, filename: str) -> Dict[str, str]:
        context = self._lookup_canonical_context(filename)
        self.global_cache.set(filename, "canonical_context", context)
        return {
            "code": "# Loaded canonical context from texts.jsonl, tables.jsonl, and images.jsonl",
            "result": context,
            "filename": filename,
        }



    def plan_next_step(self, query: str, data_desc: str, current_plan: List[str], last_result: Optional[str]) -> str:
        if not current_plan:
            prompt = PROMPT_TEMPLATES["planner_init"].format(question=query, summaries=data_desc)
            step_type = "planner_init"
        else:
            plan_str = "\n".join(f"{i+1}. {step}" for i, step in enumerate(current_plan))
            prompt = PROMPT_TEMPLATES["planner_next"].format(
                question=query, summaries=data_desc,
                plan=plan_str, result=last_result, current_step=current_plan[-1]
            )
            step_type = "planner_next"
        
        return self.controller.execute_step(
            step_type,
            step_func=lambda prompt=prompt, **kwargs: self._call_model("PLANNER", prompt),
            prompt=prompt,
            query=query,
            plan_length=len(current_plan)
        )
    
    def generate_code(self, plan: List[str], data_desc: str, base_code: Optional[str] = None) -> str:
        plan_str = "\n".join(f"{i+1}. {step}" for i, step in enumerate(plan))
        
        if not base_code:
            prompt = PROMPT_TEMPLATES["coder_init"].format(summaries=data_desc, plan=plan_str)
        else:
            prompt = PROMPT_TEMPLATES["coder_next"].format(
                summaries=data_desc, base_code=base_code,
                plan=plan_str, current_plan=plan[-1]
            )
            
        result = self.controller.execute_step(
            "coder",
            step_func=lambda prompt=prompt, **kwargs: self._call_model("CODER", prompt),
            prompt=prompt,
            plan_length=len(plan),
            has_base_code=base_code is not None
        )
        return self._extract_code_block(result)

    def verify_plan(self, plan: List[str], code: str, result: str, query: str, data_desc: str) -> str:
        plan_str = "\n".join(f"{i+1}. {step}" for i, step in enumerate(plan))
        prompt = PROMPT_TEMPLATES["verifier"].format(
            plan=plan_str, code=code, result=result, question=query, summaries=data_desc, current_step=plan[-1]
        )
        return self.controller.execute_step(
            "verifier",
            step_func=lambda prompt=prompt, **kwargs: self._call_model("VERIFIER", prompt),
            prompt=prompt,
            plan_length=len(plan)
        ).strip()

    def route_plan(self, plan: List[str], query: str, result: str, data_desc: str) -> str:
        plan_str = "\n".join(f"{i+1}. {step}" for i, step in enumerate(plan))
        prompt = PROMPT_TEMPLATES["router"].format(
            question=query, summaries=data_desc,
            plan=plan_str, result=result, current_step=plan[-1]
        )
        return self.controller.execute_step(
            "router",
            step_func=lambda prompt=prompt, **kwargs: self._call_model("ROUTER", prompt),
            prompt=prompt,
            plan_length=len(plan)
        ).strip()

    def _debug_code(self, code: str, error: str, data_desc: str, filenames: List[str]) -> str:
        prompt = PROMPT_TEMPLATES["debugger"].format(
            summaries=data_desc, code=code, bug=error, filenames=", ".join(filenames)
        )
        result = self.controller.execute_step(
            "debugger",
            step_func=lambda prompt=prompt, **kwargs: self._call_model("DEBUGGER", prompt),
            prompt=prompt,
            error_type=error.split(":")[0]
        )
        return self._extract_code_block(result)

    def finalize_solution(self, code: str, result: str, query: str, data_desc: str) -> str:
        prompt = PROMPT_TEMPLATES["finalyzer"].format(
            summaries=data_desc, code=code, result=result, question=query
        )
        result = self.controller.execute_step(
            "finalyzer",
            step_func=lambda prompt=prompt, **kwargs: self._call_model("FINALYZER", prompt),
            prompt=prompt
        )
        return self._extract_code_block(result)

    def format_answer(
        self,
        query: str,
        raw_answer: str,
        data_desc: str = "",
        source: str = "unknown",
    ) -> Dict[str, str]:
        prompt = PROMPT_TEMPLATES["answer_formatter"].format(
            question=query,
            raw_answer=raw_answer,
            summaries=data_desc or "(No additional data context.)"
        )

        try:
            result = self.controller.execute_step(
                "answer_formatter",
                step_func=lambda prompt=prompt, **kwargs: self._call_model("ANSWER_FORMATTER", prompt),
                prompt=prompt,
                source=source
            )
        except Exception as e:
            self.controller.logger.warning(f"Answer formatter failed; using raw answer. Error: {e}")
            raw_answer = str(raw_answer).strip()
            return {"raw_result": raw_answer, "final_answer": raw_answer}

        raw_result = result.strip()
        return {
            "raw_result": raw_result,
            "final_answer": self._extract_direct_final_answer(raw_result)
        }

    def run_pipeline(self, query: str, data_files: List[str]) -> Dict[str, Any]:
        """Main pipeline with full persistence and resume capability."""
        self.controller.logger.info(f"Starting pipeline: {self.config.run_id}")
        state = self.storage.get_current_state()
        Path(self.config.data_dir).mkdir(exist_ok=True)
        
        code = None
        exec_result = None

        answer_route = state.get("answer_route")
        if not answer_route:
            answer_route = "CODE" if state.get("phase2_done") else self.decide_answer_route(query, data_files)
            state = self.storage.get_current_state()
            state["answer_route"] = answer_route
            self.storage.save_state(state)

        if answer_route == "DIRECT":
            self.controller.logger.info("=== DIRECT ANSWER BRANCH ===")
            state = self.storage.get_current_state()
            if state.get("direct_done", False):
                raw_final_result = state.get("direct_result", "")
            else:
                direct_result = self.direct_answer(query, data_files)
                raw_final_result = direct_result["final_answer"]
                state = self.storage.get_current_state()
                state["direct_done"] = True
                state["direct_result"] = raw_final_result
                state["direct_raw_result"] = direct_result["raw_result"]
                self.storage.save_state(state)

            state = self.storage.get_current_state()
            if state.get("answer_formatter_input") == raw_final_result:
                final_result = state.get("formatted_result", raw_final_result)
            else:
                formatted = self.format_answer(query, raw_final_result, source="direct")
                final_result = formatted["final_answer"]
                state = self.storage.get_current_state()
                state["answer_formatter_input"] = raw_final_result
                state["formatted_result"] = final_result
                state["formatter_raw_result"] = formatted["raw_result"]
                self.storage.save_state(state)

            output_file = self.storage.run_dir / "final_output" / "result.json"
            output_file.write_text(
                json.dumps({"final_answer": final_result}, indent=2, ensure_ascii=False),
                encoding="utf-8"
            )

            return {
                "run_id": self.config.run_id,
                "answer_route": answer_route,
                "final_result": final_result,
                "output_file": str(output_file),
                "total_steps": len(self.storage.list_steps())
            }
        
        # PHASE 1: Data Analysis
        state = self.storage.get_current_state()
        has_prior_analysis = bool(state.get("data_descriptions")) or state.get("analysis_done", False)
        if not has_prior_analysis:
            self.controller.logger.info("=== PHASE 1: ANALYZING DATA FILES ===")
        
            data_descriptions = {}
        
            for file_path in data_files:
                cached_result = self.global_cache.get(file_path)
        
                if cached_result:
                    self.controller.logger.info(
                        f"[CACHE HIT] Using cached data analysis for: {file_path}"
                    )
                    data_descriptions[file_path] = {
                        "code": cached_result["code"],
                        "result": cached_result["result"]
                    }
                else:
                    self.controller.logger.info(
                        f"[CACHE MISS] Analyzing data file: {file_path}"
                    )
                    analysis = self.analyze_data(file_path)
                    data_descriptions[file_path] = {
                        "code": analysis["code"],
                        "result": analysis["result"]
                    }

            state = self.storage.get_current_state()
            state["data_descriptions"] = data_descriptions
            state["analysis_done"] = True
            self.storage.save_state(state)
        else:
            self.controller.logger.info("Using saved data descriptions.")
            data_descriptions = state.get("data_descriptions", {})
    
        data_desc_str = "\n".join(
            (f"File: {k}\n"
             f"Code to load data: {v['code']}\n"
             f"Execution result: {v['result']}")
            for k, v in data_descriptions.items()
        ) 
        
        # PHASE 2: Iterative Planning & Execution
        state = self.storage.get_current_state()
        if not state.get("phase2_done", False):
            self.controller.logger.info("=== PHASE 2: ITERATIVE PLANNING & VERIFICATION ===")
            plan = []
            plan.append(self.plan_next_step(query, data_desc_str, plan, ""))
            code = self.generate_code(plan, data_desc_str)
            code, exec_result = self._execute_and_debug_code(code, data_files, data_desc_str)
            
            for round_idx in range(self.config.max_refinement_rounds):
                self.controller.logger.info(f"--- Refinement Round {round_idx+1} ---")
                verdict = self.verify_plan(plan, code, exec_result, query, data_desc_str)
                if verdict.lower() == "yes":
                    self.controller.logger.info("Plan verified as sufficient!")
                    break
                # routing = self.route_plan(plan, query, exec_result, data_desc_str)
                # if routing.lower().startswith("step"):
                #     try:
                #         step_to_remove = int(routing.split()[1]) - 1
                #         plan = plan[:step_to_remove]
                #     except:
                #         plan = []

                next_plan = self.plan_next_step(query, data_desc_str, plan, exec_result)
                plan.append(next_plan)
                code = self.generate_code(plan, data_desc_str, base_code=code)
                code, exec_result = self._execute_and_debug_code(code, data_files, data_desc_str)

            state = self.storage.get_current_state()
            state["phase2_done"] = True
            state["phase2_code"] = code
            state["phase2_result"] = exec_result
            self.storage.save_state(state)

        if code is None or exec_result is None:
            state = self.storage.get_current_state()
            code = state.get("phase2_code")
            exec_result = state.get("phase2_result", "")
            if code is None:
                raise ValueError("Could not load code from previous run. Re-run without --resume.")

        # PHASE 3: Finalization
        self.controller.logger.info("=== PHASE 3: FINALIZING ===")
        final_code = self.finalize_solution(
            code, exec_result, query,
            data_desc_str
        )
        final_code, raw_final_result = self._execute_and_debug_code(final_code, data_files, data_desc_str)
        formatted = self.format_answer(query, raw_final_result, data_desc_str, source="code")
        final_result = formatted["final_answer"]

        state = self.storage.get_current_state()
        state["phase3_raw_result"] = raw_final_result
        state["answer_formatter_input"] = raw_final_result
        state["formatted_result"] = final_result
        state["formatter_raw_result"] = formatted["raw_result"]
        self.storage.save_state(state)
        
        output_file = self.storage.run_dir / "final_output" / "result.json"
        output_file.write_text(
            json.dumps({"final_answer": final_result}, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )
        
        return {
            "run_id": self.config.run_id,
            "answer_route": answer_route,
            "final_result": final_result,
            "output_file": str(output_file),
            "total_steps": len(self.storage.list_steps())
        }
