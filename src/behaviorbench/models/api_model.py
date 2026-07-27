"""
Model wrappers for API-based inference.

Provides:
- OpenAIModel: For remote OpenAI/Azure APIs with token tracking and retry logic
- LocalModel: For local models deployed via ms-swift (OpenAI-compatible API)
- CentaurLocalModel: Completion-mode wrapper for marcelbinz/Llama-3.1-Centaur-*
  (the model is fine-tuned without a chat template; it expects raw completion
  with the answer encapsulated in << >> tokens)
- GPUMonitor: For tracking GPU memory usage with optional wandb logging
"""

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import openai
from tenacity import (
    RetryCallState,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from behaviorbench.models.utils import (
    GPT5_MODELS,
    TokenTracker,
    calculate_cost,
)

logger = logging.getLogger(__name__)


class GPUMonitor:
    """Monitor GPU memory usage and log to wandb."""

    def __init__(self, device_id: int = 0):
        """Initialize GPU monitor.

        Args:
            device_id: GPU device index to monitor
        """
        try:
            import pynvml
        except ImportError as exc:
            raise ImportError(
                "GPU monitoring requires the optional 'monitoring' extra: "
                "uv sync --extra monitoring"
            ) from exc

        self.pynvml = pynvml
        self.device_id = device_id
        self.pynvml.nvmlInit()
        self.handle = self.pynvml.nvmlDeviceGetHandleByIndex(device_id)

    def get_memory_info(self) -> dict:
        """Get current GPU memory usage.

        Returns:
            Dict with memory stats in GB
        """
        mem_info = self.pynvml.nvmlDeviceGetMemoryInfo(self.handle)
        return {
            "gpu_memory_used_gb": mem_info.used / (1024**3),
            "gpu_memory_total_gb": mem_info.total / (1024**3),
            "gpu_memory_free_gb": mem_info.free / (1024**3),
            "gpu_memory_utilization_pct": (mem_info.used / mem_info.total) * 100,
        }

    def get_utilization(self) -> dict:
        """Get GPU utilization percentage.

        Returns:
            Dict with utilization stats
        """
        util = self.pynvml.nvmlDeviceGetUtilizationRates(self.handle)
        return {
            "gpu_utilization_pct": util.gpu,
            "gpu_memory_bandwidth_pct": util.memory,
        }

    def log_to_wandb(self, extra_metrics: dict | None = None):
        """Log GPU metrics to wandb.

        Args:
            extra_metrics: Additional metrics to log alongside GPU stats
        """
        metrics = {}
        metrics.update(self.get_memory_info())
        metrics.update(self.get_utilization())
        if extra_metrics:
            metrics.update(extra_metrics)
        try:
            import wandb
        except ImportError as exc:
            raise ImportError(
                "wandb logging requires the optional 'monitoring' extra: uv sync --extra monitoring"
            ) from exc

        wandb.log(metrics)

    def shutdown(self):
        """Cleanup pynvml."""
        self.pynvml.nvmlShutdown()


def _log_retry(retry_state: RetryCallState):
    """Log retry attempts."""
    exception = retry_state.outcome.exception()
    attempt = retry_state.attempt_number
    wait_time = retry_state.next_action.sleep if retry_state.next_action else 0
    logger.warning(
        f"[Retry] API call failed (attempt {attempt}), "
        f"retrying in {wait_time:.1f}s: {str(exception)[:100]}"
    )


class OpenAIModel:
    """
    Wrapper for OpenAI/Azure API with token tracking and retry logic.

    Usage:
        model = OpenAIModel(model_name="gpt-4o-mini", api_key="...")
        response = model({"system": "...", "user": "..."})
    """

    def __init__(
        self,
        model_name: str,
        api_key: str | None = None,
        api_base: str | None = None,
        max_tokens: int = 16384,
        temperature: float | None = None,
        track_tokens: bool = True,
        use_azure: bool = True,
        azure_api_version: str | None = None,
        concurrency: int = 4,
        reasoning_effort: str | None = None,
    ):
        """
        Initialize OpenAI model wrapper.

        Args:
            model_name: Model name (e.g., "gpt-4o-mini")
            api_key: API key (defaults to OPENAI_API_KEY or AZURE_API_KEY env var)
            api_base: Custom API base URL (for Azure: AZURE_ENDPOINT env var)
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature (defaults to 1.0 for GPT-5, 0.6 for others)
            track_tokens: Whether to track token usage and costs
            use_azure: Whether to use Azure OpenAI API
            azure_api_version: Azure API version (defaults to AZURE_API_VERSION env var)
            concurrency: Number of concurrent API calls (default: 4)
            reasoning_effort: Reasoning effort level (none/low/medium/high/xhigh) for GPT-5 series
        """
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.track_tokens = track_tokens
        self.concurrency = concurrency
        self.reasoning_effort = reasoning_effort

        # All API models default to T=1.0 (provider default for OpenAI, Anthropic, DeepSeek)
        if temperature is None:
            self.temperature = 1.0
        else:
            self.temperature = temperature

        if use_azure:
            azure_key = api_key or os.environ.get("AZURE_API_KEY")
            azure_endpoint = api_base or os.environ.get("AZURE_ENDPOINT")
            api_version = azure_api_version or os.environ.get(
                "AZURE_API_VERSION", "2024-02-15-preview"
            )

            if not azure_key:
                raise ValueError("Azure API key required. Set AZURE_API_KEY in .env file")
            if not azure_endpoint:
                raise ValueError("Azure endpoint required. Set AZURE_ENDPOINT in .env file")

            self.client = openai.AzureOpenAI(
                api_key=azure_key,
                azure_endpoint=azure_endpoint,
                api_version=api_version,
            )
        elif api_base:
            self.client = openai.OpenAI(api_key=api_key or "not-needed", base_url=api_base)
        else:
            self.client = openai.OpenAI(api_key=api_key)

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential_jitter(initial=1, max=60, jitter=2),
        retry=retry_if_exception_type(Exception),
        before_sleep=_log_retry,
        reraise=True,
    )
    def _call_api(self, messages: list[dict[str, str]]) -> tuple[str, dict[str, int]]:
        """Call API with retry logic."""
        kwargs: dict = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature,
        }
        # GPT-5 models use max_completion_tokens instead of max_tokens
        if self.model_name in GPT5_MODELS:
            kwargs["max_completion_tokens"] = self.max_tokens
        else:
            kwargs["max_tokens"] = self.max_tokens
        # Reasoning effort (GPT-5 series, o-series)
        if self.reasoning_effort is not None:
            kwargs["reasoning_effort"] = self.reasoning_effort
        response = self.client.chat.completions.create(**kwargs)
        message = response.choices[0].message
        content = message.content or ""
        # DeepSeek/GLM native APIs use "reasoning_content"; OpenRouter uses "reasoning"
        reasoning = (
            getattr(message, "reasoning_content", None)
            or getattr(message, "reasoning", None)
            or ""
        )
        if reasoning:
            content = f"<think>\n{reasoning}\n</think>\n{content}"
        if not content:
            logger.warning(
                "Model returned empty content (reasoning may have exhausted token limit)"
            )
        usage = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
        }
        # Add reasoning tokens if available (for GPT-5 models)
        if (
            hasattr(response.usage, "completion_tokens_details")
            and response.usage.completion_tokens_details
        ):
            details = response.usage.completion_tokens_details
            if hasattr(details, "reasoning_tokens"):
                usage["reasoning_tokens"] = details.reasoning_tokens
        return content, usage

    def __call__(self, prompt: dict[str, str], **kwargs) -> str:
        """
        Call the model with a prompt.

        Args:
            prompt: Dict with "system" and "user" keys
            **kwargs: Additional arguments (ignored)

        Returns:
            Model response text
        """
        messages = []
        if "system" in prompt and prompt["system"]:
            messages.append({"role": "system", "content": prompt["system"]})
        messages.append({"role": "user", "content": prompt["user"]})

        content, usage = self._call_api(messages)

        if self.track_tokens:
            cost = calculate_cost(
                usage["prompt_tokens"], usage["completion_tokens"], self.model_name
            )
            TokenTracker.track(
                usage["prompt_tokens"],
                usage["completion_tokens"],
                cost,
                self.model_name,
                reasoning_tokens=usage.get("reasoning_tokens", 0),
            )

        return content

    def batch_call(self, prompts: list[dict[str, str]], **kwargs) -> list[str]:
        """
        Call the model with multiple prompts concurrently.

        Args:
            prompts: List of prompt dicts with "system" and "user" keys
            **kwargs: Additional arguments passed to each call

        Returns:
            List of model response texts in the same order as prompts
        """
        results: list[str] = [""] * len(prompts)
        total = len(prompts)
        completed = 0
        failed = 0

        def process_prompt(idx: int, prompt: dict[str, str]) -> tuple[int, str]:
            response = self(prompt, **kwargs)
            return idx, response

        with ThreadPoolExecutor(max_workers=self.concurrency) as executor:
            futures = [
                executor.submit(process_prompt, i, prompt) for i, prompt in enumerate(prompts)
            ]
            for future in as_completed(futures):
                try:
                    idx, response = future.result()
                    results[idx] = response
                    completed += 1
                    print(f"Completed {completed}/{total} API calls", flush=True)
                except Exception as e:
                    failed += 1
                    logger.error(f"API call failed ({failed} total failures): {e}")
                    completed += 1

        if failed > 0:
            raise RuntimeError(
                f"{failed}/{total} API calls failed. Results would contain empty predictions. "
                f"Reduce concurrency or check rate limits."
            )

        return results


class AnthropicModel:
    """
    Wrapper for Anthropic models deployed on Azure AI Foundry.

    Uses the anthropic SDK's AnthropicFoundry client with Azure-specific
    endpoint (different from the OpenAI-compatible Azure endpoint).

    Usage:
        model = AnthropicModel(model_name="claude-opus-4-6")
        response = model({"system": "...", "user": "..."})
    """

    def __init__(
        self,
        model_name: str,
        api_key: str | None = None,
        base_url: str | None = None,
        max_tokens: int = 1024,
        temperature: float | None = None,
        track_tokens: bool = True,
        concurrency: int = 4,
        reasoning_effort: str | None = None,
    ):
        from anthropic import AnthropicFoundry

        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature if temperature is not None else 1.0
        self.track_tokens = track_tokens
        self.concurrency = concurrency
        self.reasoning_effort = reasoning_effort

        api_key = (
            api_key or os.environ.get("AZURE_ANTHROPIC_API_KEY") or os.environ.get("AZURE_API_KEY")
        )
        base_url = base_url or os.environ.get("AZURE_ANTHROPIC_ENDPOINT")

        if not api_key:
            raise ValueError(
                "Anthropic API key required. Set AZURE_ANTHROPIC_API_KEY or AZURE_API_KEY in .env"
            )
        if not base_url:
            raise ValueError(
                "Anthropic Azure endpoint required. Set AZURE_ANTHROPIC_ENDPOINT in .env "
                "(e.g., https://your-hub.services.ai.azure.com/anthropic/)"
            )

        self.client = AnthropicFoundry(api_key=api_key, base_url=base_url)

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential_jitter(initial=1, max=60, jitter=2),
        retry=retry_if_exception_type(Exception),
        before_sleep=_log_retry,
        reraise=True,
    )
    def _call_api(self, system: str, user: str) -> tuple[str, dict[str, int]]:
        """Call Anthropic API with retry logic."""
        kwargs: dict = {
            "model": self.model_name,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": [{"role": "user", "content": user}],
        }
        if system:
            kwargs["system"] = system
        # Effort parameter (Anthropic output_config.effort)
        if self.reasoning_effort is not None:
            kwargs["output_config"] = {"effort": self.reasoning_effort}

        response = self.client.messages.create(**kwargs)
        content = response.content[0].text if response.content else ""
        usage = {
            "prompt_tokens": response.usage.input_tokens,
            "completion_tokens": response.usage.output_tokens,
        }
        return content, usage

    def __call__(self, prompt: dict[str, str], **kwargs) -> str:
        system = prompt.get("system", "")
        user = prompt.get("user", "")

        content, usage = self._call_api(system, user)

        if self.track_tokens:
            cost = calculate_cost(
                usage["prompt_tokens"], usage["completion_tokens"], self.model_name
            )
            TokenTracker.track(
                usage["prompt_tokens"],
                usage["completion_tokens"],
                cost,
                self.model_name,
            )

        return content

    def batch_call(self, prompts: list[dict[str, str]], **kwargs) -> list[str]:
        results: list[str] = [""] * len(prompts)
        total = len(prompts)
        completed = 0
        failed = 0

        def process_prompt(idx: int, prompt: dict[str, str]) -> tuple[int, str]:
            response = self(prompt, **kwargs)
            return idx, response

        with ThreadPoolExecutor(max_workers=self.concurrency) as executor:
            futures = [
                executor.submit(process_prompt, i, prompt) for i, prompt in enumerate(prompts)
            ]
            for future in as_completed(futures):
                try:
                    idx, response = future.result()
                    results[idx] = response
                    completed += 1
                    print(f"Completed {completed}/{total} API calls", flush=True)
                except Exception as e:
                    failed += 1
                    logger.error(f"API call failed ({failed} total failures): {e}")
                    completed += 1

        if failed > 0:
            raise RuntimeError(
                f"{failed}/{total} API calls failed. Results would contain empty predictions. "
                f"Reduce concurrency or check rate limits."
            )

        return results


class VertexAIModel:
    """
    Wrapper for Google Vertex AI (Gemini) using the google-genai SDK with ADC.

    Authenticates via Application Default Credentials — no API key needed.
    Requires `gcloud auth application-default login` to have been run.

    Environment variables (set automatically by __init__ or via env):
        GOOGLE_CLOUD_PROJECT: GCP project ID
        GOOGLE_CLOUD_LOCATION: "global" for preview models, or a region like "us-central1"
        GOOGLE_GENAI_USE_VERTEXAI: must be "True"

    Usage:
        model = VertexAIModel(model_name="gemini-2.5-flash", project="my-project")
        response = model({"system": "...", "user": "..."})
    """

    def __init__(
        self,
        model_name: str,
        project: str,
        region: str = "global",
        max_tokens: int = 16384,
        temperature: float | None = None,
        track_tokens: bool = True,
        concurrency: int = 4,
        reasoning_effort: str | None = None,
    ):
        from google import genai
        from google.genai.types import HttpOptions

        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature if temperature is not None else 1.0
        self.track_tokens = track_tokens
        self.concurrency = concurrency
        self.reasoning_effort = reasoning_effort
        self.project = project
        self.region = region

        # Set env vars for the genai SDK
        os.environ["GOOGLE_CLOUD_PROJECT"] = project
        os.environ["GOOGLE_CLOUD_LOCATION"] = region
        os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "True"

        self.client = genai.Client(http_options=HttpOptions(api_version="v1"))

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential_jitter(initial=1, max=120, jitter=3),
        retry=retry_if_exception_type(Exception),
        before_sleep=_log_retry,
        reraise=True,
    )
    def _call_api(self, system: str | None, user: str) -> tuple[str, dict[str, int]]:
        """Call Vertex AI Gemini API with retry logic."""
        from google.genai.types import GenerateContentConfig, ThinkingConfig

        config_kwargs: dict = {
            "system_instruction": system if system else None,
            "max_output_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        # Thinking level (Gemini 3.x)
        if self.reasoning_effort is not None:
            config_kwargs["thinking_config"] = ThinkingConfig(thinking_level=self.reasoning_effort)
        config = GenerateContentConfig(**config_kwargs)

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=user,
            config=config,
        )
        content = response.text or ""
        usage = {
            "prompt_tokens": response.usage_metadata.prompt_token_count or 0,
            "completion_tokens": response.usage_metadata.candidates_token_count or 0,
            "reasoning_tokens": getattr(response.usage_metadata, "thoughts_token_count", None) or 0,
        }
        return content, usage

    def __call__(self, prompt: dict[str, str], **kwargs) -> str:
        system = prompt.get("system", None)
        user = prompt.get("user", "")

        content, usage = self._call_api(system, user)

        if self.track_tokens:
            # Vertex AI reports candidates (visible) and thoughts (reasoning)
            # separately — combine for cost since both are billed as output.
            total_output = usage["completion_tokens"] + usage.get("reasoning_tokens", 0)
            cost = calculate_cost(usage["prompt_tokens"], total_output, self.model_name)
            TokenTracker.track(
                usage["prompt_tokens"],
                usage["completion_tokens"],
                cost,
                self.model_name,
                reasoning_tokens=usage.get("reasoning_tokens", 0),
            )

        return content

    def batch_call(self, prompts: list[dict[str, str]], **kwargs) -> list[str]:
        results: list[str] = [""] * len(prompts)
        total = len(prompts)
        completed = 0
        failed = 0

        def process_prompt(idx: int, prompt: dict[str, str]) -> tuple[int, str]:
            response = self(prompt, **kwargs)
            return idx, response

        with ThreadPoolExecutor(max_workers=self.concurrency) as executor:
            futures = [
                executor.submit(process_prompt, i, prompt) for i, prompt in enumerate(prompts)
            ]
            for future in as_completed(futures):
                try:
                    idx, response = future.result()
                    results[idx] = response
                    completed += 1
                    print(f"Completed {completed}/{total} API calls", flush=True)
                except Exception as e:
                    failed += 1
                    logger.error(f"API call failed ({failed} total failures): {e}")
                    completed += 1

        if failed > 0:
            raise RuntimeError(
                f"{failed}/{total} API calls failed. Results would contain empty predictions. "
                f"Reduce concurrency or check rate limits."
            )

        return results


class LocalModel:
    """
    Wrapper for local models deployed via ms-swift (OpenAI-compatible API).

    Usage:
        model = LocalModel(
            model_name="qwen3-4b",
            api_base="http://localhost:8000/v1"
        )
        response = model({"system": "...", "user": "..."})
    """

    def __init__(
        self,
        model_name: str,
        api_base: str = "http://localhost:8000/v1",
        max_tokens: int = 64,
        temperature: float = 0.6,
        top_p: float | None = None,
        top_k: int | None = None,
        min_p: float | None = None,
        concurrency: int = 8,
        use_wandb: bool = False,
        gpu_device_id: int = 0,
        timeout: float = 1200.0,
    ):
        """
        Initialize local model wrapper.

        Args:
            model_name: Model name as served by the API
            api_base: API base URL (default: http://localhost:8000/v1)
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature (default: 0.6)
            top_p: Top-p (nucleus) sampling parameter (default: None, use model default)
            top_k: Top-k sampling parameter (default: None, use model default)
            min_p: Min-p sampling parameter (default: None, use model default)
            concurrency: Number of concurrent API calls (default: 8)
            use_wandb: Whether to log GPU metrics to wandb
            gpu_device_id: GPU device index to monitor (default: 0)
            timeout: API request timeout in seconds (default: 1200)
        """
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.min_p = min_p
        self.concurrency = concurrency
        self.client = openai.OpenAI(api_key="not-needed", base_url=api_base, timeout=timeout)
        self.use_wandb = use_wandb
        self.gpu_monitor = GPUMonitor(gpu_device_id) if use_wandb else None

    def __call__(self, prompt: dict[str, str], **kwargs) -> str:
        """
        Call the model with a prompt.

        Args:
            prompt: Dict with "system" and "user" keys
            **kwargs: Additional arguments (ignored)

        Returns:
            Model response text
        """
        messages = []
        if "system" in prompt and prompt["system"]:
            messages.append({"role": "system", "content": prompt["system"]})
        messages.append({"role": "user", "content": prompt["user"]})

        try:
            # Build API call kwargs
            api_kwargs = {
                "model": self.model_name,
                "messages": messages,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
            }

            # Add top_p if specified (standard OpenAI parameter)
            if self.top_p is not None:
                api_kwargs["top_p"] = self.top_p

            # Add top_k and min_p via extra_body (vLLM/swift extension)
            extra_body = {}
            if self.top_k is not None:
                extra_body["top_k"] = self.top_k
            if self.min_p is not None:
                extra_body["min_p"] = self.min_p
            if extra_body:
                api_kwargs["extra_body"] = extra_body

            response = self.client.chat.completions.create(**api_kwargs)
            return response.choices[0].message.content or ""
        except Exception as e:
            logger.error(f"API call failed: {e}")
            raise

    def batch_call(self, prompts: list[dict[str, str]], **kwargs) -> list[str]:
        """
        Call the model with multiple prompts concurrently.

        Args:
            prompts: List of prompt dicts with "system" and "user" keys
            **kwargs: Additional arguments passed to __call__

        Returns:
            List of model response texts in the same order as input prompts
        """
        results: list[str] = [""] * len(prompts)
        total = len(prompts)
        completed = 0
        start_time = time.time()

        call_times: list[float] = []
        call_lengths: list[int] = []

        def process_prompt(idx: int, prompt: dict[str, str]) -> tuple[int, str, float]:
            t0 = time.time()
            response = self(prompt, **kwargs)
            elapsed_call = time.time() - t0
            return idx, response, elapsed_call

        with ThreadPoolExecutor(max_workers=self.concurrency) as executor:
            futures = [
                executor.submit(process_prompt, i, prompt) for i, prompt in enumerate(prompts)
            ]
            for future in as_completed(futures):
                try:
                    idx, response, elapsed_call = future.result()
                    results[idx] = response
                    call_times.append(elapsed_call)
                    call_lengths.append(len(response))
                    completed += 1
                    print(
                        f"Completed {completed}/{total} API calls "
                        f"[{elapsed_call:.1f}s, {len(response)} chars]",
                        flush=True,
                    )

                    # Log GPU metrics to wandb
                    if self.use_wandb and self.gpu_monitor:
                        elapsed = time.time() - start_time
                        throughput = completed / elapsed if elapsed > 0 else 0
                        self.gpu_monitor.log_to_wandb(
                            {
                                "completed_samples": completed,
                                "total_samples": total,
                                "throughput_samples_per_sec": throughput,
                                "elapsed_time_sec": elapsed,
                            }
                        )
                except Exception as e:
                    logger.error(f"API call failed in batch: {e}")
                    raise

        # Log timing summary
        if call_times:
            avg_time = sum(call_times) / len(call_times)
            max_time = max(call_times)
            avg_len = sum(call_lengths) / len(call_lengths)
            max_len = max(call_lengths)
            print(
                f"\n--- Batch timing summary ---\n"
                f"  Calls: {len(call_times)}/{total}\n"
                f"  Per-call time: avg={avg_time:.1f}s, max={max_time:.1f}s\n"
                f"  Response length: avg={avg_len:.0f} chars, max={max_len} chars\n"
                f"  Wall time: {time.time() - start_time:.1f}s",
                flush=True,
            )

        # Log final metrics
        if self.use_wandb and self.gpu_monitor:
            total_time = time.time() - start_time
            self.gpu_monitor.log_to_wandb(
                {
                    "final_throughput_samples_per_sec": total / total_time if total_time > 0 else 0,
                    "total_time_sec": total_time,
                    "batch_completed": True,
                }
            )

        return results


class CentaurLocalModel(LocalModel):
    """
    Centaur (marcelbinz/Llama-3.1-Centaur-{8B,70B}) completion-mode wrapper.

    Centaur was fine-tuned without a chat template; it expects raw text and
    emits a single answer encapsulated in `<<...>>`. We therefore call the
    `/v1/completions` endpoint and append a `<<` cue. The swift server must
    be deployed with `--use_chat_template false` so the prompt reaches the
    model verbatim (no Llama 3 chat tokens).

    The model is trained to continue text and emit a single answer between
    `<<` and `>>` tokens. We therefore:
      - Replace any `[]`-style instruction in the user prompt with `<<>>`,
        so the in-prompt directive matches the cue we append.
      - Append a bare `<<` continuation cue (no placeholder text — including
        `X` as a literal placeholder makes the model echo `X>>`).
    """

    ANSWER_CUE = "\n<<"

    @staticmethod
    def _rewrite_bracket_instructions(text: str) -> str:
        """Swap [] / [$x] / [x] mentions in the user prompt for <<>> / <<$x>> / <<x>>.

        Centaur was trained to emit answers in `<<>>`, so aligning the prompt's
        own format directive avoids a conflict with the appended `<<` cue.
        Conservative rewrite: only touches the small set of literal patterns
        used in benchmark prompts; leaves any other [...] content (e.g., dataset notes)
        alone.
        """
        replacements = [
            ("[$x]", "<<$x>>"),
            ("[$X]", "<<$X>>"),
            ("[x]", "<<x>>"),
            ("[X]", "<<X>>"),
            ("[ ]", "<< >>"),
            (" in [] ", " in << >> "),
            (" in [].", " in << >>."),
            (" in [],", " in << >>,"),
        ]
        for old, new in replacements:
            text = text.replace(old, new)
        return text

    def _build_prompt_text(self, prompt: dict[str, str]) -> str:
        system = prompt.get("system", "") or ""
        user = self._rewrite_bracket_instructions(prompt.get("user", ""))
        if system:
            return f"{system}\n{user}{self.ANSWER_CUE}"
        return f"{user}{self.ANSWER_CUE}"

    @staticmethod
    def _normalize_completion_text(text: str) -> str:
        """Return one complete Centaur answer span, without duplicated closers.

        Some completion servers include the stop text in ``choices[0].text``.
        Since this wrapper already prepends the ``<<`` cue and reattaches a
        closing ``>>`` for downstream parsers, keep only the first answer span
        when the server returns its own close marker.
        """
        text = (text or "").strip()
        if text.startswith("<<"):
            answer = text
        else:
            answer = f"<<{text}"

        close_idx = answer.find(">>")
        if close_idx >= 0:
            return answer[: close_idx + 2]
        return f"{answer}>>"

    def __call__(self, prompt: dict[str, str], **kwargs) -> str:
        prompt_text = self._build_prompt_text(prompt)

        api_kwargs = {
            "model": self.model_name,
            "prompt": prompt_text,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "stop": [">>", "\n\n"],
        }
        if self.top_p is not None:
            api_kwargs["top_p"] = self.top_p

        extra_body = {}
        if self.top_k is not None:
            extra_body["top_k"] = self.top_k
        if self.min_p is not None:
            extra_body["min_p"] = self.min_p
        if extra_body:
            api_kwargs["extra_body"] = extra_body

        try:
            response = self.client.completions.create(**api_kwargs)
            text = response.choices[0].text or ""
            # Reattach the cue we provided so downstream parsers that look
            # for `<<X>>` see a complete pair: model output is e.g. `$50.00`,
            # we return `<<$50.00>>` so existing bracket/numeric parsers work.
            return self._normalize_completion_text(text)
        except Exception as e:
            logger.error(f"Centaur completion call failed: {e}")
            raise
