# Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# See LICENSE and THIRD_PARTY_NOTICES.md for license information.

import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "run_alps_shuffle_study.py"
SPEC = importlib.util.spec_from_file_location("run_alps_shuffle_study", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class ShuffleStudyTest(unittest.TestCase):
    def test_five_arms_include_aligned_and_odd_ce_push(self):
        self.assertEqual(len(MODULE.ARMS), 5)
        aligned = MODULE.ARM_BY_NAME["p2p_ce_push_aligned"]
        odd = MODULE.ARM_BY_NAME["p2p_ce_push_odd"]
        self.assertIn("push", aligned.benchmark_flags)
        self.assertFalse(aligned.odd_count)
        self.assertIn("--odd-count", odd.benchmark_flags)
        self.assertTrue(odd.odd_count)

    def test_cta_environment_is_owned_by_arm(self):
        inherited = {key: "17" for key in MODULE.CONTROLLED_KEYS}
        auto, auto_effective = MODULE.arm_environment(
            inherited, MODULE.ARM_BY_NAME["nccl_auto_aligned"]
        )
        for key in MODULE.CONTROLLED_KEYS:
            self.assertNotIn(key, auto)
            self.assertEqual(auto_effective[key], "<unset>")

        fixed, fixed_effective = MODULE.arm_environment(
            inherited, MODULE.ARM_BY_NAME["nccl_cta64_aligned"]
        )
        self.assertEqual(fixed["NCCL_MIN_CTAS"], "64")
        self.assertEqual(fixed["NCCL_MAX_CTAS"], "64")
        self.assertEqual(fixed_effective["NCCL_MIN_CTAS"], "64")
        for key in MODULE.CONTROLLED_KEYS[2:]:
            self.assertNotIn(key, fixed)

    def test_topology_uses_two_and_six_hosts(self):
        hosts = MODULE.parse_hosts(
            "h1:4,h2:4,h3:4,h4:4,h5:4,h6:4", 4
        )
        self.assertEqual(MODULE.hosts_for_gpu_count(hosts, 4, 8), ("h1", "h2"))
        self.assertEqual(
            MODULE.hosts_for_gpu_count(hosts, 4, 24),
            ("h1", "h2", "h3", "h4", "h5", "h6"),
        )

    def test_odd_nominal_size_adds_one_int32(self):
        base = {"per_peer_bytes": "65532", "g": "8"}
        odd = MODULE.enrich_row(
            base, 8, 1, MODULE.ARM_BY_NAME["p2p_ce_push_odd"]
        )
        aligned = MODULE.enrich_row(
            {"per_peer_bytes": "65536", "g": "8"},
            8,
            1,
            MODULE.ARM_BY_NAME["p2p_ce_push_aligned"],
        )
        self.assertEqual(odd["nominal_per_peer_bytes"], 65536)
        self.assertEqual(aligned["nominal_per_peer_bytes"], 65536)

    def test_command_sets_requested_ctas_inside_rank_environment(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = types.SimpleNamespace(
                mpirun=Path("/mpi"),
                ssh_port=2222,
                gpus_per_node=4,
                benchmark=Path("/bench"),
                msg_exp_min=16,
                msg_exp_max=30,
                iters=20,
                warmup=5,
            )
            command = MODULE.build_command(
                args,
                ("h1", "h2", "h3", "h4", "h5", "h6"),
                24,
                1,
                MODULE.ARM_BY_NAME["nccl_cta64_aligned"],
                root / "x.csv",
            )
            self.assertIn("h1:4,h2:4,h3:4,h4:4,h5:4,h6:4", command)
            self.assertIn("NCCL_MIN_CTAS=64", command)
            self.assertIn("NCCL_MAX_CTAS=64", command)


if __name__ == "__main__":
    unittest.main()
