# Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# See LICENSE and THIRD_PARTY_NOTICES.md for license information.

from __future__ import annotations

import importlib.util
import math
import sys
import unittest
from pathlib import Path


VISUALIZE_DIR = Path(__file__).resolve().parents[1]
SCRIPT = VISUALIZE_DIR / "model_distributed_tpch_strong_scaling.py"
sys.path.insert(0, str(VISUALIZE_DIR))
SPEC = importlib.util.spec_from_file_location("strong_scaling_model", SCRIPT)
assert SPEC and SPEC.loader
MODEL = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODEL
SPEC.loader.exec_module(MODEL)


def message_row(**overrides: str) -> dict[str, str]:
    row = {
        "query": "Q1",
        "rank": "0",
        "step_index": "4",
        "step_kind": "DistributedEqJoin",
        "step_label": "join",
        "phase_index": "1",
        "phase_name": "shuffle.left.exchange",
        "communication_pattern": "shuffle",
        "collective_primitive": "nccl_send_recv",
        "api": "ncclSend",
        "direction": "send",
        "peer": "1",
        "root": "-1",
        "column_ordinal": "0",
        "column_type": "type_id::INT64",
        "component": "fixed_value",
        "segment": "whole",
        "payload_bytes": "100",
        "is_self": "0",
    }
    row.update(overrides)
    return row


class StrongScalingModelTests(unittest.TestCase):
    def test_power_fit_recovers_exact_parameters(self) -> None:
        alpha = 7_500.0
        beta = -0.93
        gpus = [8, 16, 24, 32]
        values = [alpha * n**beta for n in gpus]
        fitted_alpha, fitted_beta = MODEL.fit_power_xy(gpus, values)
        self.assertAlmostEqual(fitted_alpha, alpha, places=8)
        self.assertAlmostEqual(fitted_beta, beta, places=10)

    def test_bandwidth_fit_recovers_exact_curve(self) -> None:
        a = 4.0e-6
        b = 1.5e-12
        messages = [2.0**exponent for exponent in range(16, 29)]
        bandwidths = [m / (a + b * m) / MODEL.BYTES_PER_GB for m in messages]
        fit = MODEL.fit_bandwidth_xy(messages, bandwidths, "synthetic")
        self.assertTrue(math.isclose(fit.a_seconds, a, rel_tol=1e-8))
        self.assertTrue(math.isclose(fit.b_seconds_per_byte, b, rel_tol=1e-8))
        self.assertLess(fit.rmse_log, 1e-10)

    def test_shuffle_uses_send_side_and_excludes_self(self) -> None:
        rows = []
        for rank in range(3):
            for peer in range(3):
                rows.append(
                    message_row(
                        rank=str(rank),
                        peer=str(peer),
                        payload_bytes=str(100 + 10 * rank + peer),
                        is_self="1" if rank == peer else "0",
                    )
                )
                rows.append(
                    message_row(
                        rank=str(rank),
                        peer=str(peer),
                        api="ncclRecv",
                        direction="receive",
                        payload_bytes="999999",
                        is_self="1" if rank == peer else "0",
                    )
                )
        fit = MODEL.BandwidthFit(
            scope="test",
            gpu_counts="3",
            a_seconds=4e-6,
            b_seconds_per_byte=1.5e-12,
            knee_message_bytes=4e-6 / 1.5e-12,
            saturation_gbps=1.0 / 1.5e-12 / MODEL.BYTES_PER_GB,
            r2_log=1.0,
            rmse_log=0.0,
            mape_percent=0.0,
            max_ape_percent=0.0,
            sample_count=1,
            objective="test",
        )
        output = MODEL.extract_shuffle_columns(
            rows,
            scale=1000,
            gpus=3,
            bandwidth_fit=fit,
            calibrated_min=1.0,
            calibrated_max=1e9,
            source=Path("messages.csv"),
        )
        self.assertEqual(len(output), 1)
        expected = sum(100 + 10 * rank + peer for rank in range(3) for peer in range(3) if rank != peer)
        self.assertEqual(output[0]["total_cross_bytes_all_ranks"], expected)
        self.assertEqual(output[0]["observed_messages"], 6)
        self.assertEqual(output[0]["expected_messages"], 6)
        self.assertAlmostEqual(output[0]["average_per_peer_message_bytes"], expected / 6)

    def test_allgather_deduplicates_observer_ranks(self) -> None:
        rows = []
        root_payloads = (100, 200, 300)
        for observer in range(3):
            for root, payload in enumerate(root_payloads):
                rows.append(
                    message_row(
                        rank=str(observer),
                        peer="-1",
                        root=str(root),
                        communication_pattern="broadcast",
                        collective_primitive="nccl_broadcast",
                        api="ncclBroadcast",
                        direction="collective",
                        phase_name="broadcast.right",
                        payload_bytes=str(payload),
                    )
                )
        fit = MODEL.BandwidthFit(
            scope="test",
            gpu_counts="3",
            a_seconds=4e-6,
            b_seconds_per_byte=1.5e-12,
            knee_message_bytes=4e-6 / 1.5e-12,
            saturation_gbps=1.0 / 1.5e-12 / MODEL.BYTES_PER_GB,
            r2_log=1.0,
            rmse_log=0.0,
            mape_percent=0.0,
            max_ape_percent=0.0,
            sample_count=1,
            objective="test",
        )
        output = MODEL.extract_allgather_columns(
            rows,
            scale=1000,
            gpus=3,
            bandwidth_fit=fit,
            calibrated_min=1.0,
            calibrated_max=1e9,
            source=Path("messages.csv"),
        )
        self.assertEqual(len(output), 1)
        self.assertEqual(output[0]["observed_messages"], 3)
        self.assertEqual(output[0]["expected_messages"], 3)
        self.assertEqual(output[0]["average_per_peer_message_bytes"], 200.0)
        self.assertEqual(output[0]["total_cross_bytes_all_ranks"], 1200.0)
        self.assertEqual(output[0]["average_cross_bytes_per_gpu"], 400.0)

    def test_message_projection_uses_shuffle_square_and_allgather_linear_scaling(self) -> None:
        fit = MODEL.BandwidthFit(
            scope="test",
            gpu_counts="8",
            a_seconds=4e-6,
            b_seconds_per_byte=1.5e-12,
            knee_message_bytes=4e-6 / 1.5e-12,
            saturation_gbps=1.0 / 1.5e-12 / MODEL.BYTES_PER_GB,
            r2_log=1.0,
            rmse_log=0.0,
            mape_percent=0.0,
            max_ape_percent=0.0,
            sample_count=1,
            objective="test",
        )
        common = {
            "scale_factor": 1000,
            "gpus": 8,
            "query": 1,
            "step_index": 4,
            "step_kind": "DistributedEqJoin",
            "step_label": "join",
            "phase_index": 1,
            "column_ordinal": 0,
            "column_type": "type_id::INT64",
            "component": "fixed_value",
            "segment": "whole",
            "average_per_peer_message_bytes": 800.0,
            "minimum_message_bytes": 400.0,
            "maximum_message_bytes": 1200.0,
            "observed_messages": 56,
            "expected_messages": 56,
            "total_cross_bytes_all_ranks": 44_800.0,
            "average_cross_bytes_per_gpu": 5_600.0,
            "achievable_per_gpu_cross_gbps": 1.0,
            "predicted_time_ms": 1.0,
            "outside_calibrated_message_range": 0,
            "source_messages_csv": "messages.csv",
        }
        measured = [
            {
                **common,
                "communication_kind": "shuffle",
                "phase_name": "shuffle.left.exchange",
                "communication_pattern": "shuffle",
                "collective_primitive": "nccl_send_recv",
            },
            {
                **common,
                "communication_kind": "allgather",
                "phase_name": "broadcast.right",
                "communication_pattern": "broadcast",
                "collective_primitive": "nccl_broadcast",
                "observed_messages": 8,
                "expected_messages": 8,
            },
        ]
        projected = MODEL.project_message_columns(
            measured,
            anchor_gpus=8,
            prediction_gpu_counts=(8, 16),
            bandwidth_fit=fit,
            calibrated_min=1.0,
            calibrated_max=1e9,
        )
        shuffle_16 = next(
            row
            for row in projected
            if row["gpus"] == 16 and row["communication_kind"] == "shuffle"
        )
        allgather_16 = next(
            row
            for row in projected
            if row["gpus"] == 16 and row["communication_kind"] == "allgather"
        )
        self.assertEqual(shuffle_16["average_per_peer_message_bytes"], 200.0)
        self.assertEqual(allgather_16["average_per_peer_message_bytes"], 400.0)
        self.assertEqual(shuffle_16["average_cross_bytes_per_gpu"], 3_000.0)
        self.assertEqual(allgather_16["average_cross_bytes_per_gpu"], 6_000.0)


if __name__ == "__main__":
    unittest.main()
