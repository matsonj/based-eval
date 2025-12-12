"""OpenRouter API adapter for LLM calls.

This is the unified adapter for all BASED Eval games. It combines:
- Class-based API (from Codenames) for games that prefer stateful adapters
- Function-based API (from Connections) for simpler direct calls
- Thinking model detection from model_mappings.yml
- Retry logic with exponential backoff
"""

import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import requests
import yaml

from ..utils.retry import retry_with_backoff

logger = logging.getLogger(__name__)


def _get_shared_inputs_path() -> Path:
    """Get path to shared/inputs directory."""
    return Path(__file__).parent.parent / "inputs"


def _load_model_mappings(mappings_file: Optional[Path] = None) -> Dict[str, Any]:
    """Load model mappings from YAML configuration file."""
    if mappings_file is None:
        mappings_file = _get_shared_inputs_path() / "model_mappings.yml"
    
    try:
        with open(mappings_file, "r") as f:
            data = yaml.safe_load(f)
        return data.get("models", {})
    except FileNotFoundError:
        logger.warning(f"Model mappings file not found: {mappings_file}")
        return {}
    except Exception as e:
        logger.error(f"Error loading model mappings: {e}")
        return {}


def _load_thinking_models(mappings_file: Optional[Path] = None) -> Set[str]:
    """Load the set of thinking model IDs from model_mappings.yml."""
    mappings = _load_model_mappings(mappings_file)
    thinking_models = mappings.get("thinking", {})
    return set(thinking_models.values())


# Cache the thinking models set (loaded once)
_THINKING_MODELS: Optional[Set[str]] = None


def _get_thinking_models() -> Set[str]:
    """Get cached thinking models set."""
    global _THINKING_MODELS
    if _THINKING_MODELS is None:
        _THINKING_MODELS = _load_thinking_models()
    return _THINKING_MODELS


def _get_api_key() -> str:
    """Get OpenRouter API key from environment."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable not set")
    return api_key


def resolve_model_id(model_name: str, mappings: Optional[Dict[str, Any]] = None) -> str:
    """Resolve a CLI model name to OpenRouter model ID.
    
    Args:
        model_name: CLI model name (e.g., "gemini-flash") or full model ID
        mappings: Optional pre-loaded mappings dict
        
    Returns:
        OpenRouter model ID (e.g., "google/gemini-2.5-flash")
    """
    if mappings is None:
        mappings = _load_model_mappings()
    
    # Check thinking models first
    if "thinking" in mappings and model_name in mappings["thinking"]:
        return mappings["thinking"][model_name]
    
    # Check non-thinking models
    if "non_thinking" in mappings and model_name in mappings["non_thinking"]:
        return mappings["non_thinking"][model_name]
    
    # Flat structure fallback (legacy format)
    if model_name in mappings:
        return mappings[model_name]
    
    # If not found in mappings, assume it's already a full model ID
    return model_name


@retry_with_backoff(max_retries=5, base_delay=2.0, exceptions=(requests.RequestException,))
def chat(messages: List[Dict], model: str, timeout: int = 300) -> Dict:
    """
    Call OpenRouter Chat Completions API (function-based API).
    
    Args:
        messages: List of message objects with 'role' and 'content'
        model: OpenRouter model ID (e.g., 'openai/o3', 'google/gemini-2.5-flash')
        timeout: Request timeout in seconds
        
    Returns:
        Raw API response JSON including usage and cost info
        
    Raises:
        requests.RequestException: On API errors
    """
    openrouter_model = model
    
    url = "https://openrouter.ai/api/v1/chat/completions"
    
    headers = {
        "Authorization": f"Bearer {_get_api_key()}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/matsonj/based-eval",
        "X-Title": "BASED Eval"
    }
    
    # Check if this is a thinking model
    thinking_models = _get_thinking_models()
    is_thinking_model = openrouter_model in thinking_models
    
    # Special handling for Gemini reasoning models (keeps temperature)
    is_gemini_thinking_model = openrouter_model == 'google/gemini-2.5-pro'
    
    # Check if this is a DeepSeek model that supports reasoning parameter
    is_deepseek_reasoning_model = (
        openrouter_model.startswith('deepseek/') and 
        is_thinking_model
    )
    
    payload = {
        "model": openrouter_model,
        "messages": messages,
        "usage": {
            "include": True  # Request cost and usage information
        }
    }
    
    # Handle different model types
    if is_thinking_model:
        # Thinking models don't support max_tokens or temperature and need longer timeout
        if timeout < 600:
            timeout = 600
            
        # Special case: Gemini thinking models keep temperature
        if is_gemini_thinking_model:
            payload.update({
                "temperature": 0.0,
            })
        
        # Special case: DeepSeek models support reasoning parameter
        if is_deepseek_reasoning_model:
            payload.update({
                "reasoning": {
                    "enabled": True
                }
            })
    else:
        # Standard models
        payload.update({
            "max_tokens": 25000,
            "temperature": 0.0,
        })
    
    response = requests.post(url, json=payload, headers=headers, timeout=timeout)
    
    # Check for OpenRouter-specific errors before raising
    if not response.ok:
        try:
            error_data = response.json()
            error_msg = error_data.get("error", {}).get("message", "")
            
            # Check for data policy configuration error
            if "data policy" in error_msg.lower() and response.status_code == 404:
                logger.error(f"[OpenRouter] Data policy configuration required for model: {openrouter_model}")
                logger.error(f"[OpenRouter] Error details: {error_msg}")
                detailed_msg = (
                    f"OpenRouter data policy error for model '{openrouter_model}': {error_msg}\n"
                    f"Configure your data policy settings at: https://openrouter.ai/settings/privacy"
                )
                error = requests.HTTPError(detailed_msg)
                error.response = response
                raise error
        except requests.HTTPError:
            # Re-raise our custom error
            raise
        except (ValueError, KeyError):
            # If we can't parse the error JSON, fall through to default handling
            pass
    
    response.raise_for_status()
    
    response_data = response.json()
    
    # DEBUG: Log if content is missing but tokens were used
    if response_data.get("choices") and len(response_data["choices"]) > 0:
        choice = response_data["choices"][0]
        message = choice.get("message", {})
        content = message.get("content", "")
        usage = response_data.get("usage", {})
        completion_tokens = usage.get("completion_tokens", 0)
        
        if (not content or content.strip() == "") and completion_tokens > 0:
            logger.warning(f"[OpenRouter] Model generated {completion_tokens} tokens but content is empty!")
            logger.warning(f"[OpenRouter] finish_reason: {choice.get('finish_reason')}")
            logger.warning(f"[OpenRouter] Message keys: {list(message.keys())}")
    
    return response_data


class OpenRouterAdapter:
    """Class-based adapter for calling AI models through OpenRouter.
    
    This provides a stateful interface that caches model mappings and
    provides retry logic. Useful for games that make many sequential calls.
    """

    def __init__(self, model_mappings_file: Optional[str] = None):
        self.api_key = _get_api_key()
        
        # Load model mappings from YAML file
        if model_mappings_file:
            self.model_mappings = _load_model_mappings(Path(model_mappings_file))
        else:
            self.model_mappings = _load_model_mappings()
        
        logger.info(f"Loaded model mappings with {len(self._flatten_mappings())} models")

    def _flatten_mappings(self) -> Dict[str, str]:
        """Flatten hierarchical mappings to simple name->id dict."""
        flat = {}
        if "thinking" in self.model_mappings:
            flat.update(self.model_mappings["thinking"])
        if "non_thinking" in self.model_mappings:
            flat.update(self.model_mappings["non_thinking"])
        # Also include any flat entries
        for k, v in self.model_mappings.items():
            if k not in ("thinking", "non_thinking") and isinstance(v, str):
                flat[k] = v
        return flat

    def resolve_model(self, model_name: str) -> str:
        """Resolve CLI model name to OpenRouter model ID."""
        flat = self._flatten_mappings()
        if model_name in flat:
            return flat[model_name]
        # If not found, assume it's already a full model ID
        return model_name

    def call_model(self, model_name: str, prompt: str) -> str:
        """Call AI model with retry logic. Returns just the content."""
        result = self.call_model_with_metadata(model_name, prompt)
        return result[0]

    def call_model_with_metadata(self, model_name: str, prompt: str) -> Tuple[str, Dict]:
        """Call AI model with retry logic and return detailed metadata."""
        model_id = self.resolve_model(model_name)
        
        if model_name not in self._flatten_mappings():
            logger.warning(f"Model '{model_name}' not found in mappings, using as-is: {model_id}")
        
        logger.debug(f"Calling model {model_id} (from {model_name}) with prompt length: {len(prompt)}")
        
        # Track timing
        start_time = time.time()
        
        # Use the function-based chat API
        messages = [{"role": "user", "content": prompt}]
        response_data = chat(messages, model_id)
        
        end_time = time.time()
        latency_ms = (end_time - start_time) * 1000
        
        # Extract content
        content = ""
        if response_data.get("choices") and len(response_data["choices"]) > 0:
            content = response_data["choices"][0].get("message", {}).get("content", "")
        
        # Extract metadata
        usage = response_data.get("usage", {})
        metadata = {
            "model_id": model_id,
            "latency_ms": latency_ms,
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
            "openrouter_cost": usage.get("cost", 0.0) or 0.0,
            "upstream_cost": 0.0,
        }
        
        # Extract upstream cost from cost_details
        cost_details = usage.get("cost_details", {})
        if cost_details and "upstream_inference_cost" in cost_details:
            metadata["upstream_cost"] = float(cost_details["upstream_inference_cost"])
        
        logger.info(
            f"Model call completed. Tokens: {metadata['total_tokens']}, "
            f"Latency: {latency_ms:.1f}ms"
        )
        
        return content or "", metadata

    def get_available_models(self) -> List[str]:
        """Get list of available model names."""
        return list(self._flatten_mappings().keys())

