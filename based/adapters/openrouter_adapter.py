"""OpenRouter adapter for AI model calls."""

import logging
import os
import time
from pathlib import Path
from typing import Dict, Optional, Set, Tuple

import yaml
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)


class OpenRouterAdapter:
    """Adapter for calling AI models through OpenRouter."""

    def __init__(self, model_mappings_file: Optional[str] = None):
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable is required")

        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.api_key,
            default_headers={
                "HTTP-Referer": "https://github.com/matsonj/based-eval",
                "X-Title": "BASED Eval"
            }
        )

        # Load model mappings from YAML file
        self.model_mappings = self._load_model_mappings(model_mappings_file)
        
        # Load thinking models set for reasoning model detection
        self.thinking_models = self._load_thinking_models(model_mappings_file)

    def _load_model_mappings(
        self, mappings_file: Optional[str] = None
    ) -> Dict[str, str]:
        """Load model mappings from YAML configuration file."""
        if mappings_file is None:
            # Default to shared/inputs/model_mappings.yml
            file_path = Path(__file__).parent.parent.parent / "shared" / "inputs" / "model_mappings.yml"
        else:
            file_path = Path(mappings_file)

        try:
            with open(file_path, "r") as f:
                data = yaml.safe_load(f)
            raw_mappings = data.get("models", {})
            
            # Flatten hierarchical structure (thinking/non_thinking)
            mappings = {}
            if "thinking" in raw_mappings:
                mappings.update(raw_mappings["thinking"])
            if "non_thinking" in raw_mappings:
                mappings.update(raw_mappings["non_thinking"])
            # Also include any flat entries (legacy format)
            for k, v in raw_mappings.items():
                if k not in ("thinking", "non_thinking") and isinstance(v, str):
                    mappings[k] = v
            
            logger.info(f"Loaded {len(mappings)} model mappings from {file_path}")
            return mappings
        except FileNotFoundError:
            logger.warning(f"Model mappings file not found: {file_path}")
            # Fallback to basic mappings
            return {
                "gpt4": "openai/gpt-4",
                "claude": "anthropic/claude-3.5-sonnet",
                "gemini": "google/gemini-2.5-pro",
            }
        except Exception as e:
            logger.error(f"Error loading model mappings: {e}")
            # Fallback to basic mappings
            return {
                "gpt4": "openai/gpt-4",
                "claude": "anthropic/claude-3.5-sonnet",
                "gemini": "google/gemini-2.5-pro",
            }

    def _load_thinking_models(
        self, mappings_file: Optional[str] = None
    ) -> Set[str]:
        """Load the set of thinking model IDs from model_mappings.yml."""
        if mappings_file is None:
            file_path = Path(__file__).parent.parent.parent / "shared" / "inputs" / "model_mappings.yml"
        else:
            file_path = Path(mappings_file)
        
        try:
            with open(file_path, "r") as f:
                data = yaml.safe_load(f)
            raw_mappings = data.get("models", {})
            thinking_models = raw_mappings.get("thinking", {})
            return set(thinking_models.values())
        except (FileNotFoundError, KeyError, yaml.YAMLError):
            # Fallback to empty set if config loading fails
            return set()

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    def call_model(self, model_name: str, prompt: str) -> str:
        """Call AI model with retry logic."""
        result = self.call_model_with_metadata(model_name, prompt)
        return result[0]  # Return just the content

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    def call_model_with_metadata(self, model_name: str, prompt: str) -> Tuple[str, Dict]:
        """Call AI model with retry logic and return detailed metadata."""
        try:
            # Map model name to OpenRouter model ID
            model_id = self.model_mappings.get(model_name, model_name)

            if model_name not in self.model_mappings:
                logger.warning(
                    f"Model '{model_name}' not found in mappings, using as-is: {model_id}"
                )

            logger.debug(
                f"Calling model {model_id} (from {model_name}) with prompt length: {len(prompt)}"
            )

            # Track timing
            start_time = time.time()

            # Check if this is a reasoning model that doesn't support certain parameters
            is_reasoning_model = self._is_reasoning_model(model_id)

            # Create request parameters based on model type
            is_gemini_reasoning = model_id.startswith('google/gemini-2.5')
            
            # Common parameters for all model types
            common_params = {
                "model": model_id,
                "messages": [{"role": "user", "content": prompt}],
                "extra_body": {
                    "usage": {
                        "include": True  # Request cost and usage information
                    }
                }
            }
            
            if is_reasoning_model:
                response = self.client.chat.completions.create(**common_params)
            elif is_gemini_reasoning:
                # Gemini reasoning models: no max_tokens limit, temperature 0.0
                response = self.client.chat.completions.create(
                    **common_params,
                    temperature=0.0,
                )
            else:
                response = self.client.chat.completions.create(
                    **common_params,
                    max_tokens=5000,
                    temperature=0.0,
                )

            end_time = time.time()
            latency_ms = (end_time - start_time) * 1000

            content = response.choices[0].message.content

            # Extract metadata
            metadata = {
                "model_id": model_id,
                "latency_ms": latency_ms,
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "openrouter_cost": 0.0,
                "upstream_cost": 0.0,

            }

            # Extract token usage if available
            if hasattr(response, "usage") and response.usage:
                metadata["input_tokens"] = getattr(response.usage, "prompt_tokens", 0)
                metadata["output_tokens"] = getattr(response.usage, "completion_tokens", 0)
                metadata["total_tokens"] = getattr(response.usage, "total_tokens", 0)
                
                logger.info(
                    f"Model call completed. Tokens: {metadata['total_tokens']}, "
                    f"Latency: {latency_ms:.1f}ms"
                )

            # Extract cost information from response usage field
            if hasattr(response, "usage") and response.usage:
                usage = response.usage
                
                # Extract cost information using the eval-connections approach
                total_cost = getattr(usage, "cost", None)
                if total_cost is not None:
                    metadata["openrouter_cost"] = float(total_cost)
                    logger.debug(f"Found OpenRouter cost: ${total_cost}")
                
                # Extract upstream cost from cost_details using eval-connections approach
                cost_details = getattr(usage, "cost_details", None)
                if cost_details:
                    # Try both object attribute and dictionary access for compatibility
                    upstream_cost = None
                    if hasattr(cost_details, "upstream_inference_cost"):
                        upstream_cost = getattr(cost_details, "upstream_inference_cost", None)
                    elif hasattr(cost_details, "get"):
                        upstream_cost = cost_details.get("upstream_inference_cost")
                    
                    if upstream_cost is not None:
                        metadata["upstream_cost"] = float(upstream_cost)
                        logger.debug(f"Found upstream cost: ${upstream_cost}")
                
                # Log success if we found any cost info
                if total_cost is not None or metadata.get("upstream_cost") is not None:
                    logger.debug(f"Successfully extracted cost info: OR=${metadata.get('openrouter_cost', 0.0)}, Upstream=${metadata.get('upstream_cost', 0.0)}")
                else:
                    logger.debug("No cost information found in usage field")
            else:
                logger.debug("No usage information in response")

            return content or "", metadata

        except Exception as e:
            logger.error(
                f"Error calling model {model_name} (mapped to {model_id}): {e}"
            )
            raise



    def _is_reasoning_model(self, model_id: str) -> bool:
        """Check if model is a reasoning model that doesn't support standard parameters.
        
        Uses the thinking models from model_mappings.yml as the source of truth.
        """
        # First check if model_id is directly in thinking models set
        if model_id in self.thinking_models:
            return True
        
        # Fallback patterns for models not in mappings (e.g., direct model IDs)
        reasoning_patterns = [
            "o1",
            "o3",
            "o4",
            "gpt-5",
            "grok-4",
            "grok-3-mini",
            "gpt-oss-120b",
            "gpt-oss-20b",
            "deepseek-r1",
        ]
        return any(pattern in model_id for pattern in reasoning_patterns)

    def get_available_models(self) -> list:
        """Get list of available model names."""
        return list(self.model_mappings.keys())
