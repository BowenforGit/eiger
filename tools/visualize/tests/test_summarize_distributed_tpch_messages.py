# Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# See LICENSE and THIRD_PARTY_NOTICES.md for license information.

from __future__ import annotations

import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "summarize_distributed_tpch_messages.py"
SPEC = importlib.util.spec_from_file_location("summarize_messages", SCRIPT)
assert SPEC and SPEC.loader
summarize_messages = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(summarize_messages)


class MessageSummaryTests(unittest.TestCase):
    def test_discovers_latest_pass_and_excludes_receives(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            campaign = Path(temporary) / "campaign"
            attempt = campaign / "arms/legacy/messages/q01/attempt-0001"
            attempt.mkdir(parents=True)
            (attempt / "result.json").write_text(json.dumps({"status": "pass"}))
            with (attempt / "messages.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(
                    stream,
                    fieldnames=[
                        "query", "ranks", "rank", "phase_name", "api",
                        "direction", "component", "payload_bytes",
                    ],
                )
                writer.writeheader()
                writer.writerows(
                    [
                        {
                            "query": "Q1", "ranks": "8", "rank": "0",
                            "phase_name": "shuffle.exchange", "api": "ncclSend",
                            "direction": "send", "component": "fixed_value",
                            "payload_bytes": "100",
                        },
                        {
                            "query": "Q1", "ranks": "8", "rank": "1",
                            "phase_name": "shuffle.exchange", "api": "ncclSend",
                            "direction": "send", "component": "fixed_value",
                            "payload_bytes": "300",
                        },
                        {
                            "query": "Q1", "ranks": "8", "rank": "1",
                            "phase_name": "shuffle.exchange", "api": "ncclRecv",
                            "direction": "receive", "component": "fixed_value",
                            "payload_bytes": "100",
                        },
                    ]
                )

            inputs = summarize_messages.discover_inputs([campaign])
            rows = summarize_messages.summarize(
                inputs, include_receives=False, by_rank=False
            )
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["message_count"], 2)
            self.assertEqual(rows[0]["total_payload_bytes"], 400)
            self.assertEqual(rows[0]["p50_payload_bytes"], 100)
            self.assertEqual(rows[0]["p95_payload_bytes"], 300)

    def test_explicit_csv_inputs_from_attempt_directories_remain_distinct(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            csv_paths = []
            for campaign_name, payload in (("campaign-a", 100), ("campaign-b", 300)):
                messages = (
                    root
                    / campaign_name
                    / "arms/legacy/messages/q01/attempt-0001/messages.csv"
                )
                messages.parent.mkdir(parents=True)
                with messages.open("w", newline="") as stream:
                    writer = csv.DictWriter(
                        stream,
                        fieldnames=[
                            "query", "ranks", "rank", "phase_name", "api",
                            "direction", "component", "payload_bytes",
                        ],
                    )
                    writer.writeheader()
                    writer.writerow(
                        {
                            "query": "Q1", "ranks": "8", "rank": "0",
                            "phase_name": "shuffle.exchange", "api": "ncclSend",
                            "direction": "send", "component": "fixed_value",
                            "payload_bytes": str(payload),
                        }
                    )
                csv_paths.append(messages)

            inputs = summarize_messages.discover_inputs(csv_paths)
            self.assertEqual(
                [label for label, _ in inputs],
                [str(path.resolve()) for path in csv_paths],
            )
            rows = summarize_messages.summarize(
                inputs, include_receives=False, by_rank=False
            )
            self.assertEqual(len(rows), 2)
            self.assertEqual(
                {row["total_payload_bytes"] for row in rows}, {100, 300}
            )


if __name__ == "__main__":
    unittest.main()
