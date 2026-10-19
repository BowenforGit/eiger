# Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# See LICENSE and THIRD_PARTY_NOTICES.md for license information.

from __future__ import annotations

import importlib.util
import struct
import tempfile
import unittest
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[2]
MODULE_PATH = REPOSITORY / "tools" / "visualize" / "plot_alps_shuffle_study.py"
SPEC = importlib.util.spec_from_file_location("plot_alps_shuffle_study", MODULE_PATH)
assert SPEC and SPEC.loader
plot = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(plot)


class PlotShuffleStudyTest(unittest.TestCase):
    def test_readable_binary_sizes(self) -> None:
        self.assertEqual(plot.readable_binary_size(1 << 16), "64 KiB")
        self.assertEqual(plot.readable_binary_size(1 << 28), "256 MiB")

    def test_png_and_pdf_are_bandwidth_only_and_high_resolution(self) -> None:
        rows = []
        for gpu_count in (8, 24):
            for arm in plot.ARM_STYLE:
                for exponent in (16, 28):
                    rows.append(
                        {
                            "g": str(gpu_count),
                            "arm": arm,
                            "nominal_per_peer_bytes": str(1 << exponent),
                            "per_gpu_cross_gbps_median_median_across_reps": "500",
                            "per_gpu_cross_gbps_median_std_across_reps": "5",
                        }
                    )
        with tempfile.TemporaryDirectory() as directory:
            output_png = Path(directory) / "bandwidth.png"
            output_pdf = Path(directory) / "bandwidth.pdf"
            plot.render_figure(rows, output_png, output_pdf)
            png = output_png.read_bytes()
            pdf = output_pdf.read_bytes()

        self.assertEqual(png[:8], b"\x89PNG\r\n\x1a\n")
        width, height = struct.unpack(">II", png[16:24])
        self.assertGreaterEqual(width, 2800)
        self.assertGreaterEqual(height, 1500)
        pixels_per_meter = round(plot.PNG_DPI / 0.0254)
        self.assertIn(b"pHYs" + struct.pack(">IIB", pixels_per_meter, pixels_per_meter, 1), png)
        self.assertTrue(pdf.startswith(b"%PDF-"))
        self.assertGreater(len(pdf), 10_000)

    def test_labels_use_misaligned_terminology(self) -> None:
        labels = " ".join(style[0] for style in plot.ARM_STYLE.values())
        self.assertIn("misaligned", labels)
        self.assertNotIn(" odd", labels)


if __name__ == "__main__":
    unittest.main()
