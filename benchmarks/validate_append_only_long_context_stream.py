# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Streamed append-only long-context validation against an OpenAI API server."""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

DEFAULT_STAGES = (
    "1:32:OMEGA-1001",
    "2:128:OMEGA-2002",
    "3:512:OMEGA-3003",
    "4:1024:OMEGA-4004",
)


@dataclass(frozen=True)
class StageSpec:
    stage: int
    repeat: int
    omega: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a streamed append-only long-context validation."
    )
    parser.add_argument(
        "--url",
        default="http://127.0.0.1:38081/v1/chat/completions",
        help="OpenAI-compatible chat completions endpoint.",
    )
    parser.add_argument("--model", default="gemma4-31b")
    parser.add_argument("--alpha", default="ALPHA-7319")
    parser.add_argument(
        "--stage",
        action="append",
        dest="stages",
        default=None,
        help="Stage spec in STAGE:REPEAT:OMEGA form. Can be passed multiple times.",
    )
    parser.add_argument("--connectivity-omega", default="OMEGA-CONNECT")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=48)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument(
        "--execute-mode",
        choices=["full", "final_only"],
        default="full",
        help=(
            "'full' runs connectivity plus every append-only stage. "
            "'final_only' still builds the full cumulative document but only "
            "sends the final append-only request."
        ),
    )
    parser.add_argument(
        "--request-nonce-mode",
        choices=["none", "stage"],
        default="none",
        help=(
            "Inject a per-request diagnostic nonce into the system prompt to "
            "prevent prefix reuse across stages."
        ),
    )
    parser.add_argument(
        "--cache-salt-mode",
        choices=["none", "stage"],
        default="none",
        help=(
            "Attach a cache_salt to each request. 'stage' keeps prompt content "
            "unchanged while forcing prefix-cache isolation across stages."
        ),
    )
    return parser.parse_args()


def parse_stage_specs(raw_specs: list[str] | None) -> list[StageSpec]:
    specs = raw_specs if raw_specs is not None else list(DEFAULT_STAGES)
    parsed: list[StageSpec] = []
    for raw in specs:
        try:
            stage_s, repeat_s, omega = raw.split(":", 2)
            parsed.append(
                StageSpec(stage=int(stage_s), repeat=int(repeat_s), omega=omega)
            )
        except ValueError as exc:
            raise ValueError(
                f"Invalid --stage value {raw!r}; expected STAGE:REPEAT:OMEGA."
            ) from exc
    return parsed


def make_system_prompt() -> str:
    return (
        "Task: Read the provided user content and output exactly one line.\n"
        "Required output format: ALPHA=<actual opening ALPHA value>; "
        "OMEGA=<actual last OMEGA value in the full document>.\n"
        "Replace the values with the actual records from the user content.\n"
        "Do not explain. Do not quote the instructions. Do not output anything "
        "else.\n\n"
    )


def build_system_prompt(*, request_nonce_mode: str, stage: int) -> str:
    prompt = make_system_prompt()
    if request_nonce_mode == "stage":
        prompt += (
            f"Diagnostic request nonce: stage-{stage}. "
            "Ignore this field when answering.\n\n"
        )
    return prompt


def make_user_prefix(alpha: str) -> str:
    return f"Opening fixed record: ALPHA={alpha}\n\n"


def make_connectivity_prompt(alpha: str, omega: str) -> str:
    return (
        f"Opening fixed record: ALPHA={alpha}\n"
        "Connectivity probe document.\n"
        f"Final record: OMEGA={omega}\n"
    )


def make_chunk(stage: int, repeat: int, omega: str) -> str:
    lines = [f"--- CHUNK {stage} START ---"]
    for line_idx in range(repeat):
        lines.append(
            f"CHUNK {stage} LINE {line_idx:05d}: packed int attention, "
            "kv cache update, sliding window decode, preserve the last omega only."
        )
    lines.append(f"CHUNK {stage} FINAL RECORD: OMEGA={omega}")
    lines.append(f"--- CHUNK {stage} END ---")
    return "\n".join(lines) + "\n"


def stream_stage(
    *,
    url: str,
    model: str,
    messages: list[dict[str, str]],
    label: str,
    stage: int,
    repeat: int,
    expected_omega: str,
    alpha: str,
    temperature: float,
    max_tokens: int,
    timeout: int,
    cache_salt_mode: str,
) -> dict[str, object]:
    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        "stream_options": {"include_usage": True},
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if cache_salt_mode == "stage":
        payload["cache_salt"] = f"stage-{stage}"
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    text_parts: list[str] = []
    usage: dict[str, object] | None = None
    started = time.perf_counter()
    delta_count = 0

    print(
        json.dumps(
            {
                "event": "stage_start",
                "label": label,
                "stage": stage,
                "repeat": repeat,
                "expected_alpha": alpha,
                "expected_omega": expected_omega,
                "message_chars": sum(len(msg["content"]) for msg in messages),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            for raw_line in resp:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                event = json.loads(data)
                if event.get("usage") is not None:
                    usage = event["usage"]
                choices = event.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                content = delta.get("content")
                if content:
                    delta_count += 1
                    text_parts.append(content)
                    print(
                        json.dumps(
                            {
                                "event": "delta",
                                "label": label,
                                "stage": stage,
                                "delta_index": delta_count,
                                "text": content,
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        print(f"\nHTTPError: {exc.code} {body}", file=sys.stderr, flush=True)
        raise

    elapsed = time.perf_counter() - started
    text = "".join(text_parts).strip()
    expected_response = f"ALPHA={alpha}; OMEGA={expected_omega}"
    exact_match = text == expected_response
    return {
        "event": "stage_end",
        "label": label,
        "stage": stage,
        "repeat": repeat,
        "expected_alpha": alpha,
        "expected_omega": expected_omega,
        "expected_response": expected_response,
        "response": text,
        "delta_count": delta_count,
        "contains_alpha": alpha in text,
        "contains_omega": expected_omega in text,
        "exact_match": exact_match,
        "ok": exact_match,
        "usage": usage,
        "elapsed_s": round(elapsed, 3),
    }


def main() -> int:
    args = parse_args()
    stages = parse_stage_specs(args.stages)
    user_prompt = make_user_prefix(args.alpha)

    connectivity_summary = stream_stage(
        url=args.url,
        model=args.model,
        messages=[
            {
                "role": "system",
                "content": build_system_prompt(
                    request_nonce_mode=args.request_nonce_mode,
                    stage=0,
                ),
            },
            {
                "role": "user",
                "content": make_connectivity_prompt(
                    args.alpha, args.connectivity_omega
                ),
            },
        ],
        label="connectivity",
        stage=0,
        repeat=0,
        expected_omega=args.connectivity_omega,
        alpha=args.alpha,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
        cache_salt_mode=args.cache_salt_mode,
    )
    print(json.dumps(connectivity_summary, ensure_ascii=False), flush=True)

    for stage_idx, stage in enumerate(stages):
        user_prompt += make_chunk(stage.stage, stage.repeat, stage.omega)
        if args.execute_mode == "final_only" and stage_idx != len(stages) - 1:
            continue
        summary = stream_stage(
            url=args.url,
            model=args.model,
            messages=[
                {
                    "role": "system",
                    "content": build_system_prompt(
                        request_nonce_mode=args.request_nonce_mode,
                        stage=stage.stage,
                    ),
                },
                {"role": "user", "content": user_prompt},
            ],
            label="append_only",
            stage=stage.stage,
            repeat=stage.repeat,
            expected_omega=stage.omega,
            alpha=args.alpha,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
            cache_salt_mode=args.cache_salt_mode,
        )
        print(json.dumps(summary, ensure_ascii=False), flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
