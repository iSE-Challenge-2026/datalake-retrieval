from pathlib import Path
import json
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Any

class GlobalCacheManager:
    def __init__(self, cache_file: str = "src/cache/global_cache.json"):
        self.cache_file = Path(cache_file)
        self.cache = self._load_cache()

    def _load_cache(self) -> Dict[str, Any]:
        if not self.cache_file.exists():
            print(f"[CACHE] Cache file does not exist: {self.cache_file}")
            return {}
    
        try:
            text = self.cache_file.read_text(encoding="utf-8")
            loaded = json.loads(text)
    
            if not isinstance(loaded, dict):
                print(
                    f"[CACHE WARNING] Cache file is not a dict. "
                    f"Got {type(loaded).__name__}: {self.cache_file}"
                )
                return {}
    
            print(f"[CACHE] Loaded {len(loaded)} entries from {self.cache_file}")
            return loaded
    
        except json.JSONDecodeError as e:
            print(f"[CACHE ERROR] Invalid JSON in {self.cache_file}: {e}")
            return {}
    
        except PermissionError as e:
            print(f"[CACHE ERROR] Permission denied reading {self.cache_file}: {e}")
            return {}
    
        except OSError as e:
            print(f"[CACHE ERROR] OS error reading {self.cache_file}: {e}")
            return {}

    def _get_file_key(self, filepath: str) -> str:
        p = Path(filepath)
        if not p.exists():
            return filepath
        stat = p.stat()
        # return f"{p.resolve()}_{stat.st_size}_{stat.st_mtime}"
        return f"{p.resolve()}"
    
    def get(self, filepath: str) -> Optional[Dict[str, Any]]:
        key = self._get_file_key(filepath)
        return self.cache.get(key)

    def set(self, filepath: str, code: str, result: str):
        key = self._get_file_key(filepath)
        self.cache[key] = {
            "code": code,
            "result": result,
            "cached_at": datetime.now().isoformat()
        }
        self._save_cache()

    def _save_cache(self):
        try:
            self.cache_file.write_text(json.dumps(self.cache, indent=2, ensure_ascii=False), encoding='utf-8')
        except Exception as e:
            print(f"[CACHE WARNING] Can not write: {e}")