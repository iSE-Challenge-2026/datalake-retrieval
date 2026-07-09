import os
import json
import base64
import mimetypes
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional
import openai
from fnmatch import fnmatch

try:
    from google import generativeai
except ImportError:
    generativeai = None


class ModelProvider(ABC):
    """Abstract base class for model providers."""

    @classmethod
    @abstractmethod
    def provider_instance(cls, model_name: str) -> bool:
        """Check if this provider can handle the given model name."""
        pass

    @property
    @abstractmethod
    def env_var_name(self) -> str:
        """The name of the environment variable required for the API key."""
        pass

    @abstractmethod
    def generate_content(self, prompt: str, file_paths: Optional[List[str]] = None) -> dict:
        """Generates content and returns a dict containing text and usage stats."""
        pass


def _encode_image_for_chat(file_path: str) -> Optional[dict]:
    path = Path(file_path)
    mime_type, _ = mimetypes.guess_type(path.name)
    if not mime_type or not mime_type.startswith("image/"):
        return None

    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return {
        "type": "image_url",
        "image_url": {
            "url": f"data:{mime_type};base64,{encoded}"
        }
    }


class GeminiProvider(ModelProvider):
    """Provider for Google's Gemini models."""
    
    def __init__(self, config_api_key: str, model_name: str):
        if generativeai is None:
            raise ImportError("google-generativeai is required to use Gemini models.")

        self.api_key = config_api_key or os.getenv(self.env_var_name)
        if not self.api_key:
            raise ValueError(f"Missing API key for {model_name}. Env var = {self.env_var_name}.")

        self.model_name = model_name
        generativeai.configure(api_key=self.api_key)
        self.model = generativeai.GenerativeModel(self.model_name)

    @classmethod
    def provider_instance(cls, model_name: str) -> bool:
        return True

    @property
    def env_var_name(self) -> str:
        return "GEMINI_API_KEY"
        
    def generate_content(self, prompt: str, file_paths: Optional[List[str]] = None) -> dict:
        contents = [prompt]
        uploaded_files = []

        for file_path in file_paths or []:
            path = Path(file_path)
            if not path.exists():
                continue
            uploaded = generativeai.upload_file(path=str(path))
            uploaded_files.append(uploaded)
            contents.append(uploaded)

        response = self.model.generate_content(contents if uploaded_files else prompt)
        
        metadata = getattr(response, 'usage_metadata', None)
        return {
            "text": response.text,
            "usage": {
                "input_tokens": getattr(metadata, 'prompt_token_count', 0),
                "output_tokens": getattr(metadata, 'candidates_token_count', 0),
                "total_tokens": getattr(metadata, 'total_token_count', 0)
            }
        }


class OpenAIProvider(ModelProvider):
    """Provider for OpenAI models."""
    
    def __init__(self, config_api_key: str, model_name: str):
        self.api_key = os.getenv(self.env_var_name, config_api_key)
        if not self.api_key:
            raise ValueError(f"Missing API key for {model_name}. Env var = {self.env_var_name}.")

        self.model_name = model_name
        self.client = openai.OpenAI(api_key=self.api_key)

    @classmethod
    def provider_instance(cls, model_name: str) -> bool:
        return model_name.startswith("gpt") or model_name.startswith("o1")

    @property
    def env_var_name(self) -> str:
        return "OPENAI_API_KEY"
        
    def generate_content(self, prompt: str, file_paths: Optional[List[str]] = None) -> dict:
        content = prompt
        if file_paths:
            content_parts = [{"type": "text", "text": prompt}]
            for file_path in file_paths:
                image_part = _encode_image_for_chat(file_path)
                if image_part:
                    content_parts.append(image_part)
            content = content_parts if len(content_parts) > 1 else prompt

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": content}],
        )
        usage_data = getattr(response, 'usage', None)
        return {
            "text": response.choices[0].message.content,
            "usage": {
                "input_tokens": getattr(usage_data, 'prompt_tokens', 0),
                "output_tokens": getattr(usage_data, 'completion_tokens', 0),
                "total_tokens": getattr(usage_data, 'total_tokens', 0)
            }
        }


class OpenRouterProvider(ModelProvider):
    """Provider for OpenRouter models."""
    
    def __init__(self, config_api_key: str, model_name: str):
        self.api_key = os.getenv(self.env_var_name, config_api_key)
        if not self.api_key:
            raise ValueError(f"Missing API key for {model_name}. Env var = {self.env_var_name}.")

        self.model_name = model_name.replace("openrouter/", "")
        self.client = openai.OpenAI(base_url="https://openrouter.ai/api/v1", api_key=self.api_key)

    @classmethod
    def provider_instance(cls, model_name: str) -> bool:
        return model_name.startswith("openrouter/")

    @property
    def env_var_name(self) -> str:
        return "OPENROUTER_API_KEY"
        
    def generate_content(self, prompt: str, file_paths: Optional[List[str]] = None) -> dict:
        content = prompt
        if file_paths:
            content_parts = [{"type": "text", "text": prompt}]
            for file_path in file_paths:
                image_part = _encode_image_for_chat(file_path)
                if image_part:
                    content_parts.append(image_part)
            content = content_parts if len(content_parts) > 1 else prompt

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": content}],
        )
        usage_data = getattr(response, 'usage', None)
        return {
            "text": response.choices[0].message.content,
            "usage": {
                "input_tokens": getattr(usage_data, 'prompt_tokens', 0),
                "output_tokens": getattr(usage_data, 'completion_tokens', 0),
                "total_tokens": getattr(usage_data, 'total_tokens', 0)
            }
        }
