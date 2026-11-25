import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

import benchmark


def stream(*events):
    return "\n\n".join(f"data: {json.dumps(event)}" for event in events) + "\n\n"


class BenchmarkTest(unittest.TestCase):
    def test_suite_uses_one_engine_and_records_three_independent_requests(self):
        prompts = ["Первый", "Второй", "Третий"]
        requests = []

        def handle(request):
            if request.url.path == "/health":
                return httpx.Response(200)
            if request.url.path == "/v1/models":
                return httpx.Response(200, json={"data": [{"id": "test-model"}]})
            requests.append(json.loads(request.content))
            return httpx.Response(200, text=stream(
                {"choices": [{"delta": {"role": "assistant"}}]},
                {"choices": [{"delta": {"content": "Ответ"}}]},
                {"choices": [{"delta": {"content": "."}, "finish_reason": "stop"}]},
                {"choices": [], "usage": {"prompt_tokens": 12, "completion_tokens": 2}},
            ) + "data: [DONE]\n\n")

        exporter = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        self.addCleanup(provider.shutdown)
        with tempfile.TemporaryDirectory() as directory, (
            httpx.Client(transport=httpx.MockTransport(handle))
        ) as client, patch.object(benchmark, "colibri") as process:
            process.return_value.__enter__.return_value.poll.return_value = None
            output = Path(directory)
            benchmark.run_suite(client, provider.get_tracer("test"), prompts, output, 4096, 48)
            with next(output.glob("*.csv")).open(encoding="utf-8") as csv_file:
                rows = list(csv.DictReader(csv_file))

        process.assert_called_once_with(4096)
        self.assertEqual([request["messages"] for request in requests], [
            [{"role": "user", "content": prompt}] for prompt in prompts
        ])
        self.assertTrue(all(request["stream_options"] == {"include_usage": True}
                            for request in requests))
        self.assertEqual([row["mode"] for row in rows], ["cold", "warm-1", "warm-2"])
        self.assertTrue(all(row["answer"] == "Ответ." and row["input_tokens"] == "12"
                            and row["output_tokens"] == "2" for row in rows))
        self.assertTrue(rows[0]["startup_seconds"])
        self.assertEqual(rows[1]["startup_seconds"], "")
        self.assertAlmostEqual(float(rows[0]["total_with_startup_seconds"]),
                               float(rows[0]["startup_seconds"]) + float(rows[0]["run_seconds"]))

        startup, *requests_spans, suite = exporter.get_finished_spans()
        self.assertEqual(startup.name, "colibri.startup")
        self.assertEqual(len(requests_spans), 3)
        for row, span in zip(rows, requests_spans):
            self.assertEqual(span.parent.span_id, suite.context.span_id)
            self.assertEqual(row["trace_id"], f"{span.context.trace_id:032x}")
            self.assertEqual(span.attributes["benchmark.output_tokens"], 2)
            self.assertEqual(span.events[0].name, "first_response_text")

    def test_incomplete_stream_fails(self):
        response = stream({"choices": [{"delta": {"content": "Ответ"}}]})
        provider = TracerProvider()
        self.addCleanup(provider.shutdown)
        with httpx.Client(transport=httpx.MockTransport(
            lambda _: httpx.Response(200, text=response)
        )) as client, provider.get_tracer("test").start_as_current_span("request") as span:
            with self.assertRaisesRegex(RuntimeError, r"\[DONE\]"):
                benchmark.measure(client, "test-model", "Промпт", 48, span)


if __name__ == "__main__":
    unittest.main()
