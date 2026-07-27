"""
Azure Batch API model wrapper.

Uses Azure Batch API (upload JSONL -> submit -> poll -> download)
for 50% cost reduction on supported deployments.
"""

import io
import json
import logging
import os
import sys
import time

import openai

from behaviorbench.models.utils import GPT5_MODELS, TokenTracker, calculate_cost

logger = logging.getLogger(__name__)


def _create_azure_client() -> openai.OpenAI:
    """Create an OpenAI client configured for Azure Batch API.

    Azure Batch API requires the base OpenAI client (not AzureOpenAI)
    with base_url pointing to the Azure resource's /openai/v1/ endpoint.
    """
    api_key = os.environ.get("AZURE_API_KEY")
    endpoint = os.environ.get("AZURE_ENDPOINT")

    if not api_key:
        print("ERROR: AZURE_API_KEY not set", file=sys.stderr)
        sys.exit(1)
    if not endpoint:
        print("ERROR: AZURE_ENDPOINT not set", file=sys.stderr)
        sys.exit(1)

    base_url = endpoint.rstrip("/") + "/openai/v1/"

    return openai.OpenAI(
        api_key=api_key,
        base_url=base_url,
    )


class AzureBatchModel:
    """
    Azure Batch API model with same interface as OpenAIModel.

    Submits all prompts as a single batch job, polls for completion,
    then downloads and parses results. Offers 50% cost reduction
    over real-time API calls on supported deployments.

    Usage:
        model = AzureBatchModel(model_name="gpt-4o-mini-batch")
        responses = model.batch_call(prompts)
    """

    def __init__(
        self,
        model_name: str,
        max_tokens: int = 16384,
        temperature: float | None = None,
        poll_interval: int = 60,
        resume_batch_id: str | None = None,
        reasoning_effort: str | None = None,
    ):
        """
        Initialize Azure Batch model.

        Args:
            model_name: Azure batch deployment name (e.g., "gpt-4o-mini-batch")
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature (defaults to 1.0 for GPT-5, 0.6 for others)
            poll_interval: Seconds between batch status polls
            resume_batch_id: Existing batch ID to resume polling (skip upload+submit)
            reasoning_effort: Reasoning effort level (none/low/medium/high/xhigh) for GPT-5 series
        """
        self.client = _create_azure_client()
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.poll_interval = poll_interval
        self.resume_batch_id = resume_batch_id
        self.reasoning_effort = reasoning_effort

        if temperature is None:
            self.temperature = 1.0 if model_name in GPT5_MODELS else 0.6
        else:
            self.temperature = temperature

    def __call__(self, prompt: dict[str, str], **kwargs) -> str:
        """Single-prompt fallback: call batch_call with 1 prompt."""
        return self.batch_call([prompt])[0]

    def _build_messages(self, prompt: dict[str, str]) -> list[dict[str, str]]:
        """Convert prompt dict to messages list."""
        messages = []
        if "system" in prompt and prompt["system"]:
            messages.append({"role": "system", "content": prompt["system"]})
        messages.append({"role": "user", "content": prompt["user"]})
        return messages

    def _build_batch_jsonl(self, prompts: list[dict[str, str]]) -> str:
        """Build JSONL string for Azure Batch API."""
        lines = []
        for idx, prompt in enumerate(prompts):
            body = {
                "model": self.model_name,
                "messages": self._build_messages(prompt),
                "temperature": self.temperature,
            }
            # GPT-5 models use max_completion_tokens instead of max_tokens
            if self.model_name in GPT5_MODELS:
                body["max_completion_tokens"] = self.max_tokens
            else:
                body["max_tokens"] = self.max_tokens
            # Reasoning effort (GPT-5 series)
            if self.reasoning_effort is not None:
                body["reasoning_effort"] = self.reasoning_effort

            line = {
                "custom_id": f"sample_{idx}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": body,
            }
            lines.append(json.dumps(line))
        return "\n".join(lines)

    def _submit_batch(self, jsonl_content: str) -> str:
        """Upload JSONL and submit batch job. Returns batch_id."""
        file_bytes = jsonl_content.encode("utf-8")
        file_obj = io.BytesIO(file_bytes)
        file_obj.name = "behaviorbench_batch.jsonl"

        uploaded = self.client.files.create(file=file_obj, purpose="batch")
        print(f"  Uploaded batch file: {uploaded.id} ({len(file_bytes)} bytes)")

        batch = self.client.batches.create(
            input_file_id=uploaded.id,
            endpoint="/chat/completions",
            completion_window="24h",
        )
        print(f"  Submitted batch: {batch.id}")
        return batch.id

    def _poll_batch(self, batch_id: str):
        """Poll batch until completed. Returns batch object."""
        while True:
            batch = self.client.batches.retrieve(batch_id)
            status = batch.status
            completed = batch.request_counts.completed if batch.request_counts else 0
            total = batch.request_counts.total if batch.request_counts else 0
            failed = batch.request_counts.failed if batch.request_counts else 0

            print(
                f"  Batch {batch_id}: status={status}, "
                f"completed={completed}/{total}, failed={failed}",
                flush=True,
            )

            if status == "completed":
                return batch
            elif status in ("failed", "expired", "cancelled", "canceled"):
                print(
                    f"ERROR: Batch {batch_id} ended with status: {status}",
                    file=sys.stderr,
                )
                if hasattr(batch, "errors") and batch.errors:
                    for err in batch.errors.data[:5]:
                        print(f"  Error: {err.code} - {err.message}", file=sys.stderr)
                sys.exit(1)

            time.sleep(self.poll_interval)

    def _download_results(self, batch) -> dict[str, dict]:
        """Download batch output and parse into {custom_id: response_body}."""
        output_file_id = batch.output_file_id
        if not output_file_id:
            print("ERROR: No output file from batch", file=sys.stderr)
            sys.exit(1)

        content = self.client.files.content(output_file_id)
        results = {}
        for line in content.text.strip().split("\n"):
            if not line.strip():
                continue
            obj = json.loads(line)
            cid = obj["custom_id"]
            resp_body = obj.get("response", {}).get("body", {})
            results[cid] = resp_body
        return results

    def batch_call(self, prompts: list[dict[str, str]], **kwargs) -> list[str]:
        """
        Call the model with multiple prompts via Azure Batch API.

        1. Build batch JSONL
        2. Upload file and submit batch (or resume existing batch)
        3. Poll until completed
        4. Download results and extract responses
        5. Track token usage

        Args:
            prompts: List of prompt dicts with "system" and "user" keys

        Returns:
            List of model response texts in the same order as input prompts
        """
        total = len(prompts)
        print(f"\nAzure Batch API: {total} prompts, model={self.model_name}")

        # Submit or resume
        if self.resume_batch_id:
            batch_id = self.resume_batch_id
            print(f"  Resuming batch: {batch_id}")
        else:
            jsonl_content = self._build_batch_jsonl(prompts)
            batch_id = self._submit_batch(jsonl_content)

        # Poll
        batch = self._poll_batch(batch_id)

        # Download and parse
        raw_results = self._download_results(batch)
        print(f"  Downloaded {len(raw_results)} results")

        # Reorder by custom_id and extract content
        results: list[str] = [""] * total
        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_reasoning_tokens = 0

        for idx in range(total):
            cid = f"sample_{idx}"
            resp_body = raw_results.get(cid, {})
            choices = resp_body.get("choices", [])
            if choices:
                message = choices[0].get("message", {})
                content = message.get("content", "")
                reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
                if reasoning:
                    content = f"<think>\n{reasoning}\n</think>\n{content}"
                results[idx] = content or ""
            else:
                logger.warning(f"No choices for {cid}")
                results[idx] = ""

            # Accumulate token usage
            usage = resp_body.get("usage", {})
            total_prompt_tokens += usage.get("prompt_tokens", 0)
            total_completion_tokens += usage.get("completion_tokens", 0)
            # Reasoning tokens from completion_tokens_details
            details = usage.get("completion_tokens_details", {})
            if isinstance(details, dict):
                total_reasoning_tokens += details.get("reasoning_tokens", 0)

        # Track aggregated token usage
        cost = calculate_cost(total_prompt_tokens, total_completion_tokens, self.model_name)
        TokenTracker.track(
            total_prompt_tokens,
            total_completion_tokens,
            cost,
            self.model_name,
            reasoning_tokens=total_reasoning_tokens,
        )

        print(
            f"  Batch complete: {total_prompt_tokens} prompt tokens, "
            f"{total_completion_tokens} completion tokens, ${cost:.4f}"
        )

        return results
