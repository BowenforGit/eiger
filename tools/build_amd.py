#!/usr/bin/env python3

# Adapted from PyTorch project
# https://github.com/pytorch/pytorch/blob/main/tools/amd_build/build_amd.py
# PyTorch copyright notices and BSD terms: LICENSES/PyTorch-BSD.txt.
# Exact original revision is not recorded; see THIRD_PARTY_NOTICES.md.

import argparse
import os
import sys


sys.path.append(
    os.path.realpath(
        os.path.join(
            __file__, os.path.pardir, "hipify_torch"
        )
    )
)

from hipify_torch import hipify_python  # type: ignore[import]


parser = argparse.ArgumentParser(
    description="Top-level script for HIPifying, filling in most common parameters"
)
parser.add_argument(
    "--out-of-place-only",
    action="store_true",
    help="Whether to only run hipify out-of-place on source files",
)

parser.add_argument(
    "--project-directory",
    type=str,
    default="",
    help="The root of the project.",
    required=False,
)

parser.add_argument(
    "--output-directory",
    type=str,
    default="",
    help="The directory to store the hipified project",
    required=False,
)

parser.add_argument(
    "--extra-include-dir",
    type=str,
    default=[],
    nargs="+",
    help="The list of extra directories in caffe2 to hipify",
    required=False,
)
parser.add_argument(
    "--clean",
    action="store_true",
    default=False,
    help="Clean up the files generated during hipifification",
)

args = parser.parse_args()


amd_build_dir = os.path.dirname(os.path.realpath(__file__))
proj_dir = os.path.join(os.path.dirname(amd_build_dir))

print(f"Project directory: {proj_dir}")

if args.project_directory:
    proj_dir = args.project_directory

out_dir = proj_dir
if args.output_directory:
    out_dir = args.output_directory

print(f"Output directory: {out_dir}")

if args.clean:
    os.system(f"find {os.path.join(proj_dir, 'src/')} -name 'hip' -type d -exec rm -rf {{}} +")
    os.system(f"find {os.path.join(proj_dir, 'build/')} -name 'hip' -type d -exec rm -rf {{}} +")
    os.system(f"find {proj_dir} -name '*_hip.*' -type f -delete")
    os.system(f"find {proj_dir} -name '*.hip' -type f -delete")
    sys.exit(0)

includes = [
    "src/cuda/*",
    "src/cuda/CMakeLists.txt",
    "src/cuda/tests/CMakeLists.txt",
    "src/cuda/benchmarks/CMakeLists.txt",
    # "build/_deps/cuco-src/*",
]

headers_include = [
    # "build/_deps/cuco-src/include/",
]

includes = [os.path.join(proj_dir, include) for include in includes]

for new_dir in args.extra_include_dir:
    abs_new_dir = os.path.join(proj_dir, new_dir)
    if os.path.exists(abs_new_dir):
        abs_new_dir = os.path.join(abs_new_dir, "**/*")
        includes.append(abs_new_dir)

ignores = [
    # "src/cuda/benchmarks/*",
]

ignores = [os.path.join(proj_dir, ignore) for ignore in ignores]

# Check if the compiler is hip-clang.
def is_hip_clang() -> bool:
    try:
        hip_path = os.getenv("HIP_PATH", "/opt/rocm/hip")
        with open(hip_path + "/lib/.hipInfo") as f:
            return "HIP_COMPILER=clang" in f.read()
    except OSError:
        return False

hipify_python.hipify(
    project_directory=proj_dir,
    output_directory=out_dir,
    includes=includes,
    ignores=ignores,
    header_include_dirs=headers_include,
    out_of_place_only=args.out_of_place_only,
    is_pytorch_extension = True,
    custom_map_list = os.path.join(__file__, os.path.pardir, "custom_hipify_mapping.json"),
    hip_clang_launch=is_hip_clang(),
)