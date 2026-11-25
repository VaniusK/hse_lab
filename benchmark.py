import csv
import json
import os
import signal
import subprocess
import sys
import time
from contextlib import ExitStack, contextmanager
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import httpx
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

PROJECT = "hse-colibri"
COLIBRI_URL = "http://127.0.0.1:8000"
METRICS = (
    "input_tokens",
    "output_tokens",
    "ttft",
    "output_speed",
    "run_seconds",
    "startup_seconds",
    "total_with_startup_seconds",
)
CSV_FIELDS = (
    "suite_id", "trace_id", "run_id", "mode", "model", "finish_reason",
    *METRICS, "prompt", "answer",
)


def wait_for_http(client, url, timeout, process=None):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process is not None and process.poll() is not None:
            raise RuntimeError(f"Colibri завершился с кодом {process.returncode}.")
        try:
            response = client.get(url, timeout=1)
            if response.status_code == 200:
                return response
        except httpx.TransportError:
            pass
        time.sleep(0.1)
    raise TimeoutError(f"Сервис не ответил за {timeout} с: {url}")


@contextmanager
def colibri(context_length):
    command = [
        sys.executable, "/opt/colibri/coli", "serve", "--model", "/model",
        "--host", "127.0.0.1", "--port", "8000", "--gpu", "none",
        "--ctx", str(context_length),
    ]
    process = subprocess.Popen(command, stdout=sys.stderr, start_new_session=True)
    try:
        yield process
    finally:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def measure(client, model, prompt, max_tokens, span):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "enable_thinking": False,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    parts = []
    usage = {}
    first_text = last_text = None
    finish_reason = None
    started = time.perf_counter()

    with client.stream("POST", f"{COLIBRI_URL}/v1/chat/completions", json=payload) as response:
        if response.is_error:
            raise RuntimeError(f"Colibri HTTP {response.status_code}: {response.read().decode()}")
        for line in response.iter_lines():
            if not line.startswith("data:"):
                continue
            data = line.removeprefix("data:").strip()
            if data == "[DONE]":
                break
            event = json.loads(data)
            if event.get("error"):
                raise RuntimeError(f"Ошибка потока Colibri: {event['error']}")
            if event.get("usage"):
                usage = event["usage"]
            for choice in event.get("choices", []):
                if choice.get("finish_reason"):
                    finish_reason = choice["finish_reason"]
                text = choice.get("delta", {}).get("content")
                if text:
                    last_text = time.perf_counter()
                    if first_text is None:
                        first_text = last_text
                        span.add_event("first_response_text")
                    parts.append(text)
        else:
            raise RuntimeError("Поток Colibri оборвался до [DONE].")
        elapsed = time.perf_counter() - started

    output_tokens = usage.get("completion_tokens")
    output_speed = None
    if output_tokens is not None and output_tokens > 1 and first_text is not None and last_text > first_text:
        output_speed = (output_tokens - 1) / (last_text - first_text)

    return {
        "input_tokens": usage.get("prompt_tokens"),
        "output_tokens": output_tokens,
        "ttft": None if first_text is None else first_text - started,
        "output_speed": output_speed,
        "run_seconds": elapsed,
        "finish_reason": finish_reason,
        "answer": "".join(parts),
    }


def run_suite(client, tracer, prompts, output, context_length, max_tokens):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    suite_id = f"{stamp}-{uuid4().hex[:8]}"
    with tracer.start_as_current_span("task-1 cold and warm") as suite:
        suite.set_attributes({
            "openinference.span.kind": "CHAIN",
            "benchmark.suite_id": suite_id,
            "benchmark.context_length": context_length,
            "benchmark.max_output_tokens": max_tokens,
            "benchmark.backend": "cpu",
            "colibri.version": os.environ.get("BENCHMARK_COLIBRI_VERSION", "unknown"),
        })
        trace_id = f"{suite.get_span_context().trace_id:032x}"
        path = output / f"{suite_id}.csv"
        with path.open("x", encoding="utf-8", newline="") as report:
            writer = csv.DictWriter(report, fieldnames=CSV_FIELDS)
            writer.writeheader()
            print(f"Phoenix: {PROJECT}, trace {trace_id}. CSV: {path}", flush=True)
            with ExitStack() as engine:
                with tracer.start_as_current_span("colibri.startup") as startup_span:
                    started = time.perf_counter()
                    process = engine.enter_context(colibri(context_length))
                    wait_for_http(
                        client, f"{COLIBRI_URL}/health",
                        float(os.environ.get("STARTUP_TIMEOUT", "300")), process,
                    )
                    startup_seconds = time.perf_counter() - started
                    startup_span.set_attribute("benchmark.startup_seconds", startup_seconds)
                models = client.get(f"{COLIBRI_URL}/v1/models").raise_for_status().json()
                model = models["data"][0]["id"]
                suite.set_attribute("llm.model_name", model)

                for run_id, (mode, prompt) in enumerate(zip(("cold", "warm-1", "warm-2"), prompts), 1):
                    with tracer.start_as_current_span(f"colibri.chat {mode}") as span:
                        span.set_attributes({
                            "openinference.span.kind": "LLM",
                            "benchmark.run_id": run_id,
                            "benchmark.mode": mode,
                            "llm.model_name": model,
                            "input.value": prompt,
                        })
                        row = measure(client, model, prompt, max_tokens, span)
                        row.update(
                            suite_id=suite_id, trace_id=trace_id, run_id=run_id,
                            mode=mode, model=model, prompt=prompt,
                            startup_seconds=startup_seconds if run_id == 1 else None,
                            total_with_startup_seconds=(
                                startup_seconds + row["run_seconds"] if run_id == 1 else None
                            ),
                        )
                        span.set_attribute("output.value", row["answer"])
                        for field in METRICS:
                            if row[field] is not None:
                                span.set_attribute(f"benchmark.{field}", row[field])
                        for field, key in (
                            ("input_tokens", "llm.token_count.prompt"),
                            ("output_tokens", "llm.token_count.completion"),
                            ("finish_reason", "benchmark.finish_reason"),
                        ):
                            if row[field] is not None:
                                span.set_attribute(key, row[field])
                        writer.writerow(row)
                        report.flush()
                        missing = [field for field in METRICS[:5] if row[field] is None]
                        print(f"{mode}: {row['run_seconds']:.3f} с")
                        if missing:
                            print(f"  Нет замера: {', '.join(missing)}", file=sys.stderr)


def main():
    prompts = [
        Path(f"/prompts/{number:02}.txt").read_text(encoding="utf-8-sig").strip()
        for number in range(1, 4)
    ]
    if not all(prompts):
        raise ValueError("В каждом из трёх файлов prompts должен быть текст.")
    context_length = int(os.environ.get("BENCHMARK_CONTEXT_LENGTH", "4096"))
    max_tokens = int(os.environ.get("MAX_OUTPUT_TOKENS", "48"))
    if not 0 < max_tokens < context_length:
        raise ValueError("Лимит ответа должен быть больше нуля и меньше контекстного окна.")
    endpoint = os.environ["OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"]
    provider = TracerProvider(resource=Resource.create({
        "service.name": "colibri-benchmark",
        "openinference.project.name": PROJECT,
    }))
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))
    try:
        with httpx.Client(timeout=float(os.environ.get("REQUEST_TIMEOUT", "300"))) as client:
            wait_for_http(client, endpoint.removesuffix("/v1/traces") + "/healthz", 60)
            run_suite(client, provider.get_tracer(__name__), prompts, Path("/output"),
                      context_length, max_tokens)
    finally:
        provider.shutdown()


if __name__ == "__main__":
    main()
