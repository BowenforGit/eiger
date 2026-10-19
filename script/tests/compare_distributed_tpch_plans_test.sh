#!/usr/bin/env bash
# Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# See LICENSE and THIRD_PARTY_NOTICES.md for license information.


set -euo pipefail

repository=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
test_root=$(mktemp -d /tmp/compare-distributed-tpch-plans-test.XXXXXX)
trap 'rm -rf -- "$test_root"' EXIT

test_script_dir="$test_root/script"
mkdir -p "$test_script_dir" "$test_root/plans"
cp "$repository/script/compare_distributed_tpch_plans.sh" \
    "$test_script_dir/compare_distributed_tpch_plans.sh"

cat > "$test_script_dir/run_distributed_tpch_bench.sh" <<'EOF'
#!/usr/bin/env bash
printf 'bench' >> "$CALL_LOG"
printf ' %q' "$@" >> "$CALL_LOG"
printf '\n' >> "$CALL_LOG"
EOF

cat > "$test_script_dir/collect_distributed_tpch_nsys.sh" <<'EOF'
#!/usr/bin/env bash
printf 'profile' >> "$CALL_LOG"
printf ' %q' "$@" >> "$CALL_LOG"
printf '\n' >> "$CALL_LOG"
EOF

chmod +x "$test_script_dir/"*.sh
export CALL_LOG="$test_root/calls.log"

"$test_script_dir/compare_distributed_tpch_plans.sh" \
    --scale 1 \
    --queries 1 \
    --ranks 2 \
    --physical-plan-dir "$test_root/plans" \
    --output-dir "$test_root/output" \
    --skip-profiles

[[ $(wc -l < "$CALL_LOG") -eq 6 ]]
[[ $(grep -c -- '--timing-only' "$CALL_LOG") -eq 2 ]]
[[ $(grep -c -- '--memory-profile' "$CALL_LOG") -eq 2 ]]
[[ $(grep -c -- '--steps-only' "$CALL_LOG") -eq 2 ]]
[[ $(grep -c -- '--step-iterations 5' "$CALL_LOG") -eq 2 ]]
[[ $(grep -c -- '/stock/breakdown' "$CALL_LOG") -eq 1 ]]
[[ $(grep -c -- '/synthesized/breakdown' "$CALL_LOG") -eq 1 ]]
[[ $(grep -c -- '--physical-plan-dir' "$CALL_LOG") -eq 3 ]]
