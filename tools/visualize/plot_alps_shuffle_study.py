#!/usr/bin/env python3
# Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# See LICENSE and THIRD_PARTY_NOTICES.md for license information.

"""Render paper-ready PNG/PDF plots and a Markdown digest for a shuffle study."""

from __future__ import annotations

import argparse
import csv
import ctypes
import ctypes.util
import math
import struct
import zlib
from pathlib import Path


ARM_STYLE = {
    # Okabe-Ito-derived, colorblind-safe palette. Markers and dashes keep the
    # figure legible in grayscale and when printed at a reduced column width.
    "nccl_auto_aligned": ("NCCL default", "#0072B2", "-", "o"),
    "p2p_ce_push_aligned": ("CE push aligned", "#D55E00", "-", "s"),
    "nccl_cta64_aligned": ("NCCL CTA64 aligned", "#009E73", "-", "^"),
    "nccl_cta64_odd": ("NCCL CTA64 misaligned", "#56B4E9", "--", "D"),
    "p2p_ce_push_odd": ("CE push misaligned", "#CC79A7", "--", "X"),
}
PANELS = (
    ("per_gpu_cross_gbps_median_median_across_reps", "Per-GPU cross bandwidth (GB/s)"),
    ("cross_gpu_gbps_median_median_across_reps", "Aggregate cross bandwidth (GB/s)"),
    ("comm_ms_median_median_across_reps", "Communication latency (ms)"),
)

FIGURE_WIDTH_PT = 7.25 * 72
FIGURE_HEIGHT_PT = 4.0 * 72
PNG_DPI = 400


class CairoTextExtents(ctypes.Structure):
    _fields_ = [
        ("x_bearing", ctypes.c_double),
        ("y_bearing", ctypes.c_double),
        ("width", ctypes.c_double),
        ("height", ctypes.c_double),
        ("x_advance", ctypes.c_double),
        ("y_advance", ctypes.c_double),
    ]


def load_cairo() -> ctypes.CDLL:
    library = ctypes.util.find_library("cairo")
    if not library:
        # The benchmark container keeps Cairo in the shared conda package
        # cache, not in the active Python environment or the loader path.
        cairo_candidates = sorted(Path("/storage/conda/pkgs").glob("cairo-*/lib/libcairo.so.2"))
        pixman_candidates = sorted(Path("/storage/conda/pkgs").glob("pixman-*/lib/libpixman-1.so.0"))
        if cairo_candidates and pixman_candidates:
            ctypes.CDLL(str(pixman_candidates[-1]), mode=ctypes.RTLD_GLOBAL)
            library = str(cairo_candidates[-1])
        else:
            raise RuntimeError("libcairo is required to render the PNG and PDF plots")
    cairo = ctypes.CDLL(library)
    pointer = ctypes.c_void_p
    cairo.cairo_image_surface_create.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]
    cairo.cairo_image_surface_create.restype = pointer
    cairo.cairo_pdf_surface_create.argtypes = [ctypes.c_char_p, ctypes.c_double, ctypes.c_double]
    cairo.cairo_pdf_surface_create.restype = pointer
    cairo.cairo_create.argtypes = [pointer]
    cairo.cairo_create.restype = pointer
    cairo.cairo_destroy.argtypes = [pointer]
    cairo.cairo_surface_finish.argtypes = [pointer]
    cairo.cairo_surface_destroy.argtypes = [pointer]
    cairo.cairo_surface_write_to_png.argtypes = [pointer, ctypes.c_char_p]
    cairo.cairo_surface_write_to_png.restype = ctypes.c_int
    for name, arguments in {
        "cairo_scale": [pointer, ctypes.c_double, ctypes.c_double],
        "cairo_set_source_rgb": [pointer, ctypes.c_double, ctypes.c_double, ctypes.c_double],
        "cairo_set_source_rgba": [pointer, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double],
        "cairo_rectangle": [pointer, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double],
        "cairo_move_to": [pointer, ctypes.c_double, ctypes.c_double],
        "cairo_line_to": [pointer, ctypes.c_double, ctypes.c_double],
        "cairo_arc": [pointer, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double],
        "cairo_set_line_width": [pointer, ctypes.c_double],
        "cairo_set_dash": [pointer, ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.c_double],
        "cairo_translate": [pointer, ctypes.c_double, ctypes.c_double],
        "cairo_rotate": [pointer, ctypes.c_double],
        "cairo_set_font_size": [pointer, ctypes.c_double],
        "cairo_show_text": [pointer, ctypes.c_char_p],
        "cairo_text_extents": [pointer, ctypes.c_char_p, ctypes.POINTER(CairoTextExtents)],
    }.items():
        getattr(cairo, name).argtypes = arguments
    cairo.cairo_select_font_face.argtypes = [pointer, ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
    for name in (
        "cairo_fill",
        "cairo_stroke",
        "cairo_close_path",
        "cairo_save",
        "cairo_restore",
        "cairo_new_path",
    ):
        getattr(cairo, name).argtypes = [pointer]
    return cairo


def rgb(hex_color: str) -> tuple[float, float, float]:
    return tuple(int(hex_color[index:index + 2], 16) / 255 for index in (1, 3, 5))


def set_source(cairo: ctypes.CDLL, context: int, color: str, alpha: float = 1.0) -> None:
    red, green, blue = rgb(color)
    cairo.cairo_set_source_rgba(context, red, green, blue, alpha)


def draw_text(
    cairo: ctypes.CDLL,
    context: int,
    x: float,
    y: float,
    value: str,
    size: float,
    *,
    color: str = "#202428",
    align: str = "left",
    bold: bool = False,
    rotation: float = 0.0,
) -> None:
    encoded = value.encode("utf-8")
    cairo.cairo_save(context)
    cairo.cairo_translate(context, x, y)
    cairo.cairo_rotate(context, rotation)
    cairo.cairo_select_font_face(context, b"sans-serif", 0, 1 if bold else 0)
    cairo.cairo_set_font_size(context, size)
    extents = CairoTextExtents()
    cairo.cairo_text_extents(context, encoded, ctypes.byref(extents))
    offset = 0.0
    if align == "center":
        offset = -(extents.x_bearing + extents.width / 2)
    elif align == "right":
        offset = -(extents.x_bearing + extents.width)
    set_source(cairo, context, color)
    cairo.cairo_move_to(context, offset, 0)
    cairo.cairo_show_text(context, encoded)
    cairo.cairo_restore(context)


def set_dash(cairo: ctypes.CDLL, context: int, dashed: bool) -> None:
    if dashed:
        values = (ctypes.c_double * 2)(6.0, 4.0)
        cairo.cairo_set_dash(context, values, 2, 0.0)
    else:
        cairo.cairo_set_dash(context, None, 0, 0.0)


def draw_marker(
    cairo: ctypes.CDLL, context: int, x: float, y: float, marker: str, color: str, size: float = 3.2
) -> None:
    set_dash(cairo, context, False)
    set_source(cairo, context, color)
    cairo.cairo_new_path(context)
    if marker == "o":
        cairo.cairo_arc(context, x, y, size, 0, 2 * math.pi)
    elif marker == "s":
        cairo.cairo_rectangle(context, x - size, y - size, 2 * size, 2 * size)
    elif marker == "^":
        cairo.cairo_move_to(context, x, y - size - 0.6)
        cairo.cairo_line_to(context, x - size - 0.4, y + size)
        cairo.cairo_line_to(context, x + size + 0.4, y + size)
        cairo.cairo_close_path(context)
    elif marker == "D":
        cairo.cairo_move_to(context, x, y - size - 0.4)
        cairo.cairo_line_to(context, x + size + 0.4, y)
        cairo.cairo_line_to(context, x, y + size + 0.4)
        cairo.cairo_line_to(context, x - size - 0.4, y)
        cairo.cairo_close_path(context)
    else:
        cairo.cairo_set_line_width(context, 2.2)
        cairo.cairo_move_to(context, x - size, y - size)
        cairo.cairo_line_to(context, x + size, y + size)
        cairo.cairo_move_to(context, x - size, y + size)
        cairo.cairo_line_to(context, x + size, y - size)
        cairo.cairo_stroke(context)
        return
    cairo.cairo_fill(context)


def load_summary(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def readable_binary_size(byte_count: int) -> str:
    if byte_count >= 1024 * 1024:
        return f"{byte_count // (1024 * 1024)} MiB"
    return f"{byte_count // 1024} KiB"


def render_figure(rows: list[dict[str, str]], output_png: Path, output_pdf: Path) -> None:
    gpu_counts = sorted({int(row["g"]) for row in rows})
    metric = PANELS[0][0]
    std_metric = "per_gpu_cross_gbps_median_std_across_reps"
    nominal = [int(row["nominal_per_peer_bytes"]) for row in rows]
    observed_max = max(
        float(row[metric]) + float(row.get(std_metric, 0.0)) for row in rows
    )
    y_max = max(100.0, math.ceil(observed_max / 100.0) * 100.0)

    cairo = load_cairo()

    def draw(context: int) -> None:
        set_source(cairo, context, "#FFFFFF")
        cairo.cairo_rectangle(context, 0, 0, FIGURE_WIDTH_PT, FIGURE_HEIGHT_PT)
        cairo.cairo_fill(context)

        left, right, top, bottom, gap = 58.0, 8.0, 79.0, 58.0, 28.0
        panel_width = (FIGURE_WIDTH_PT - left - right - gap * (len(gpu_counts) - 1)) / len(gpu_counts)
        panel_height = FIGURE_HEIGHT_PT - top - bottom
        # Keep the endpoint labels and markers inside each compact panel. The
        # padding is in log2 units, so it remains symmetric on the log axis.
        x_min = math.log2(min(nominal)) - 0.8
        x_max = math.log2(max(nominal)) + 0.8

        def x_coord(x0: float, byte_count: int) -> float:
            return x0 + panel_width * (math.log2(byte_count) - x_min) / (x_max - x_min)

        def y_coord(value: float) -> float:
            bounded = max(0.0, min(value, y_max))
            return top + panel_height * (1.0 - bounded / y_max)

        # Two-row shared legend, large enough to read at final paper size.
        legend_positions = ((15, 18), (176, 18), (342, 18), (91, 40), (300, 40))
        for (label, color, line_style, marker), (x, y) in zip(ARM_STYLE.values(), legend_positions):
            set_source(cairo, context, color)
            cairo.cairo_set_line_width(context, 1.8)
            set_dash(cairo, context, line_style == "--")
            cairo.cairo_move_to(context, x, y - 3)
            cairo.cairo_line_to(context, x + 24, y - 3)
            cairo.cairo_stroke(context)
            draw_marker(cairo, context, x + 12, y - 3, marker, color, 2.8)
            draw_text(cairo, context, x + 30, y, label, 10.8)

        tick_sizes = sorted({min(nominal), 1 << 20, 1 << 24, max(nominal)})
        tick_sizes = [size for size in tick_sizes if min(nominal) <= size <= max(nominal)]
        for panel_index, gpu_count in enumerate(gpu_counts):
            x0 = left + panel_index * (panel_width + gap)
            panel_rows = [row for row in rows if int(row["g"]) == gpu_count]
            draw_text(
                cairo,
                context,
                x0 + panel_width / 2,
                top - 10,
                f"({chr(97 + panel_index)}) {gpu_count} GPUs",
                13.0,
                align="center",
                bold=True,
            )

            for tick in range(5):
                value = y_max * tick / 4
                y = y_coord(value)
                set_source(cairo, context, "#D9DEE3")
                cairo.cairo_set_line_width(context, 0.7)
                set_dash(cairo, context, False)
                cairo.cairo_move_to(context, x0, y)
                cairo.cairo_line_to(context, x0 + panel_width, y)
                cairo.cairo_stroke(context)
                if panel_index == 0:
                    draw_text(cairo, context, x0 - 7, y + 3.7, f"{value:.0f}", 10.5, color="#343A40", align="right")

            for size in tick_sizes:
                x = x_coord(x0, size)
                set_source(cairo, context, "#ECEFF1")
                cairo.cairo_set_line_width(context, 0.6)
                cairo.cairo_move_to(context, x, top)
                cairo.cairo_line_to(context, x, top + panel_height)
                cairo.cairo_stroke(context)
                draw_text(cairo, context, x, top + panel_height + 17, readable_binary_size(size), 10.5, color="#343A40", align="center")

            # Uncertainty bands are drawn before the curves.
            for arm, (_, color, _, _) in ARM_STYLE.items():
                series = sorted(
                    (row for row in panel_rows if row["arm"] == arm),
                    key=lambda row: int(row["nominal_per_peer_bytes"]),
                )
                if not series:
                    continue
                upper = [
                    (x_coord(x0, int(row["nominal_per_peer_bytes"])), y_coord(float(row[metric]) + float(row.get(std_metric, 0.0))))
                    for row in series
                ]
                lower = [
                    (x_coord(x0, int(row["nominal_per_peer_bytes"])), y_coord(max(0.0, float(row[metric]) - float(row.get(std_metric, 0.0)))))
                    for row in reversed(series)
                ]
                cairo.cairo_new_path(context)
                cairo.cairo_move_to(context, *upper[0])
                for point in upper[1:] + lower:
                    cairo.cairo_line_to(context, *point)
                cairo.cairo_close_path(context)
                set_source(cairo, context, color, 0.10)
                cairo.cairo_fill(context)

            for arm, (_, color, line_style, marker) in ARM_STYLE.items():
                series = sorted(
                    (row for row in panel_rows if row["arm"] == arm),
                    key=lambda row: int(row["nominal_per_peer_bytes"]),
                )
                points = [
                    (x_coord(x0, int(row["nominal_per_peer_bytes"])), y_coord(float(row[metric])))
                    for row in series
                ]
                if not points:
                    continue
                cairo.cairo_new_path(context)
                cairo.cairo_move_to(context, *points[0])
                for point in points[1:]:
                    cairo.cairo_line_to(context, *point)
                set_source(cairo, context, color)
                cairo.cairo_set_line_width(context, 1.75)
                set_dash(cairo, context, line_style == "--")
                cairo.cairo_stroke(context)
                for x, y in points:
                    draw_marker(cairo, context, x, y, marker, color)

            set_source(cairo, context, "#343A40")
            cairo.cairo_set_line_width(context, 1.0)
            set_dash(cairo, context, False)
            cairo.cairo_move_to(context, x0, top)
            cairo.cairo_line_to(context, x0, top + panel_height)
            cairo.cairo_line_to(context, x0 + panel_width, top + panel_height)
            cairo.cairo_stroke(context)

        draw_text(cairo, context, FIGURE_WIDTH_PT / 2 + 12, FIGURE_HEIGHT_PT - 20, "Per-peer message size", 12.0, align="center")
        draw_text(
            cairo,
            context,
            15,
            top + panel_height / 2,
            "Per-GPU cross bandwidth (GB/s)",
            12.0,
            align="center",
            rotation=-math.pi / 2,
        )
    output_png.parent.mkdir(parents=True, exist_ok=True)
    scale = PNG_DPI / 72
    png_surface = cairo.cairo_image_surface_create(
        0, round(FIGURE_WIDTH_PT * scale), round(FIGURE_HEIGHT_PT * scale)
    )
    png_context = cairo.cairo_create(png_surface)
    cairo.cairo_scale(png_context, scale, scale)
    draw(png_context)
    cairo.cairo_destroy(png_context)
    status = cairo.cairo_surface_write_to_png(png_surface, str(output_png).encode())
    cairo.cairo_surface_destroy(png_surface)
    if status != 0:
        raise RuntimeError(f"Cairo failed to write {output_png} (status {status})")
    add_png_resolution(output_png, PNG_DPI)

    pdf_surface = cairo.cairo_pdf_surface_create(
        str(output_pdf).encode(), FIGURE_WIDTH_PT, FIGURE_HEIGHT_PT
    )
    pdf_context = cairo.cairo_create(pdf_surface)
    draw(pdf_context)
    cairo.cairo_destroy(pdf_context)
    cairo.cairo_surface_finish(pdf_surface)
    cairo.cairo_surface_destroy(pdf_surface)


def add_png_resolution(path: Path, dpi: int) -> None:
    """Add a PNG pHYs chunk so publishing tools recognize the physical DPI."""
    content = path.read_bytes()
    if content[:8] != b"\x89PNG\r\n\x1a\n":
        raise RuntimeError(f"not a PNG: {path}")
    pixels_per_meter = round(dpi / 0.0254)
    data = struct.pack(">IIB", pixels_per_meter, pixels_per_meter, 1)
    chunk_type = b"pHYs"
    chunk = struct.pack(">I", len(data)) + chunk_type + data
    chunk += struct.pack(">I", zlib.crc32(chunk_type + data) & 0xFFFFFFFF)
    ihdr_end = 8 + 4 + 4 + 13 + 4
    path.write_bytes(content[:ihdr_end] + chunk + content[ihdr_end:])


def write_report(rows: list[dict[str, str]], output: Path) -> None:
    repetitions = sorted({int(row["repetitions"]) for row in rows})
    gpu_counts = sorted({int(row["g"]) for row in rows})
    nominal_sizes = sorted({int(row["nominal_per_peer_bytes"]) for row in rows})
    lines = [
        "# ALPS shuffle bandwidth study",
        "",
        "Bandwidth values are useful cross-GPU payload divided by the maximum rank time. "
        "Misaligned arms use actual bytes in the numerator and nominal power-of-two bytes on the x-axis.",
        "",
        "## Methodology",
        "",
        f"The campaign covers {', '.join(str(value) for value in gpu_counts)} GPUs, "
        f"{len(nominal_sizes)} nominal per-peer sizes from {min(nominal_sizes):,} through "
        f"{max(nominal_sizes):,} bytes, and {repetitions[0] if len(repetitions) == 1 else repetitions} "
        "independent repetitions per point. Each reported value is the median across repetition-level "
        "medians; each repetition uses the maximum rank time so a slow rank cannot be hidden.",
        "",
        "The five arms are NCCL with its automatic CTA policy, CE push, aligned NCCL forced to 64 CTAs, "
        "misaligned NCCL forced to 64 CTAs, and misaligned CE push. The misaligned arms transfer one fewer `int32` "
        "element per peer (nominal bytes minus four).",
        "",
        "![Per-GPU shuffle cross bandwidth](bandwidth.png)",
        "",
        "[Vector PDF version](bandwidth.pdf)",
        "",
    ]
    for gpu_count in gpu_counts:
        lines.extend((f"## {gpu_count} GPUs", "", "| Arm | Peak per-GPU GB/s | Largest-size per-GPU GB/s | Largest-size latency ms |", "|---|---:|---:|---:|"))
        largest_by_arm: dict[str, dict[str, str]] = {}
        for arm, (label, _, _, _) in ARM_STYLE.items():
            series = [row for row in rows if int(row["g"]) == gpu_count and row["arm"] == arm]
            if not series:
                continue
            largest = max(series, key=lambda row: int(row["nominal_per_peer_bytes"]))
            largest_by_arm[arm] = largest
            peak = max(float(row[PANELS[0][0]]) for row in series)
            lines.append(
                f"| {label} | {peak:.3f} | {float(largest[PANELS[0][0]]):.3f} | "
                f"{float(largest[PANELS[2][0]]):.3f} |"
            )
        lines.append("")
        if len(largest_by_arm) == len(ARM_STYLE):
            bandwidth = {
                arm: float(row[PANELS[0][0]]) for arm, row in largest_by_arm.items()
            }
            lines.extend(
                (
                    "Largest-size ratios:",
                    "",
                    f"- NCCL CTA64 / NCCL auto, aligned: {bandwidth['nccl_cta64_aligned'] / bandwidth['nccl_auto_aligned']:.3f}×",
                    f"- NCCL CTA64 misaligned / aligned: {bandwidth['nccl_cta64_odd'] / bandwidth['nccl_cta64_aligned']:.3f}×",
                    f"- CE push misaligned / aligned: {bandwidth['p2p_ce_push_odd'] / bandwidth['p2p_ce_push_aligned']:.3f}×",
                    f"- CE push aligned / NCCL auto aligned: {bandwidth['p2p_ce_push_aligned'] / bandwidth['nccl_auto_aligned']:.3f}×",
                    "",
                )
            )
    geometry = output.parent / "profiles" / "CTA_GEOMETRY.md"
    if geometry.is_file():
        lines.extend(
            (
                "## CTA launch validation",
                "",
                "Nsight Systems confirms that the long-message NCCL grid is 32 blocks under the "
                "automatic policy and 64 blocks with `NCCL_MIN_CTAS=NCCL_MAX_CTAS=64`, at both GPU counts. "
                "See [the kernel-geometry evidence](profiles/CTA_GEOMETRY.md).",
                "",
            )
        )
    notes = output.parent / "CAMPAIGN_NOTES.md"
    if notes.is_file():
        lines.extend(
            (
                "## Reproduction and provenance",
                "",
                "See [the campaign notes](CAMPAIGN_NOTES.md) for the exact launch, environment ownership, "
                "host mapping, common-grid decision, and bandwidth accounting caveat.",
                "",
            )
        )
    comparison = output.parent / "ORDERED_CE_COMPARISON.md"
    if comparison.is_file():
        lines.extend(
            (
                "## CE policy comparison",
                "",
                "See [the ordered-versus-multi-stream comparison](ORDERED_CE_COMPARISON.md) for the "
                "causal rerun and the corrected NCCL comparison.",
                "",
            )
        )
    output.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path)
    args = parser.parse_args()
    summary = args.campaign / "summary.csv"
    if not summary.is_file():
        parser.error(f"missing summary: {summary}")
    rows = load_summary(summary)
    output_png = args.campaign / "bandwidth.png"
    output_pdf = args.campaign / "bandwidth.pdf"
    render_figure(rows, output_png, output_pdf)
    # Do not leave a stale legacy plot beside the authoritative PNG/PDF pair.
    (args.campaign / "bandwidth.svg").unlink(missing_ok=True)
    write_report(rows, args.campaign / "REPORT.md")
    print(output_png)
    print(output_pdf)
    print(args.campaign / "REPORT.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
