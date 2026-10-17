"""Run a bounded live context-cache audit using environment credentials.

Example: uv run --env-file .env python -m tests.cache_audit.run_branching
    --output records/review/context_cache_branching/RUN_NAME
"""

import argparse
import asyncio
import json
import os
import uuid
from pathlib import Path

import httpx

from ecs_agent.logging import configure_logging
from ecs_agent.providers.config import ApiFormat, ProviderConfig
from ecs_agent.providers.openai_model import OpenAIModel
from tests.cache_audit.branching import BranchingAudit, CaptureTransport, analyze_saved


async def main() -> None:
    """Run once; never retry until a desired cache hit rate appears."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-requests", type=int, default=32)
    parser.add_argument("--initial-records", type=int, default=220)
    parser.add_argument("--turn-records", type=int, default=18)
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args()
    if args.analyze_only:
        report = analyze_saved(args.output)
        print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2))
        return
    key = os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise SystemExit("LLM_API_KEY or OPENAI_API_KEY is required")
    base = os.environ.get("LLM_RESPONSES_BASE_URL") or os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1")
    model_id = os.environ.get("LLM_MODEL", "gpt-5.6-sol")
    args.output.mkdir(parents=True, exist_ok=False)
    configure_logging(level="WARNING")
    capture = CaptureTransport(httpx.AsyncHTTPTransport(retries=0), limit=args.max_requests, directory=args.output)
    model = OpenAIModel(config=ProviderConfig(
        provider_id="live-branch-cache", base_url=base, api_key=key,
        api_format=ApiFormat.OPENAI_RESPONSES, enable_store=False, enable_response_chaining=False,
        extra_headers={"Accept-Encoding": "identity"},
    ), model=model_id, read_timeout=120, stream_read_timeout=120)
    await model._client.aclose()
    model._client = httpx.AsyncClient(transport=capture, timeout=model._timeout, trust_env=False)
    audit = BranchingAudit(model, capture, salt=uuid.uuid4().hex[:12],
                           initial_records=args.initial_records, turn_records=args.turn_records)
    (args.output / "config.json").write_text(json.dumps({
        "endpoint": base, "model": model_id, "salt": audit.salt, "max_requests": args.max_requests,
        "initial_records": args.initial_records, "turn_records": args.turn_records,
        "store": False, "automatic_chaining": False, "synthetic_data_only": True,
        "prediction_preregistered": {
            "method": "Before HTTP: cold controls=0; verified ancestor=prior input_tokens; exact repeat=previous identical input_tokens.",
            "assumptions": "Matching cache is already written, reachable, not evicted, and server rendering is stable.",
            "scope": "Conditional reusable-prefix baseline, not a backend routing probability or guaranteed hit rate.",
            "cache_metrics": "Standard cached_tokens; gateway observed/billed extensions kept separate when present.",
            "schedule": "Same 23-turn scenario; no added warm-up, sleep, or retries to chase cache hits.",
        },
    }, indent=2))
    try:
        async with asyncio.timeout(1800):
            report = await audit.run()
        print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2), flush=True)
        if report["client_contract_failures"] or report["recall_passed"] != report["turns"]:
            raise SystemExit(1)
    except BaseException as exc:
        # Partial samples and unknown usage remain reviewable after a failure.
        report = audit.analyze()
        report["aborted"] = type(exc).__name__
        audit.save(report)
        raise
    finally:
        await model.aclose()


if __name__ == "__main__":
    asyncio.run(main())
