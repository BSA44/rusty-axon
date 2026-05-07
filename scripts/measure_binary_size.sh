#!/usr/bin/env bash
# Phase 10: binary-size measurement automation.
#
# Drives the combo matrix described in docs/PAPER_REWORK_PLAN.md (§ Phase 10):
#
#   ID  Profile        Features                          Target   Strip
#   --  -------------  --------------------------------  -------  -----
#   A   release        train,inference,matrixmultiply    host     no
#   B   release        inference                         host     no
#   C   release-edge   inference                         host     yes
#   D   release-edge   inference,quant-i8                host     yes
#   E   release-edge   inference                         aarch64  yes
#   F   release-edge   inference,quant-i8                aarch64  yes
#
# `release-edge` already strips at link time (`strip = "symbols"` in
# Cargo.toml), so the "Strip" column reflects the *effective* state of the
# measured artifact rather than an extra post-step.  The artifact in every
# row is `examples/min_inference` — the smallest realistic inference binary.
#
# aarch64 rows are only built when one of `cross` or `cargo-zigbuild` is
# available on PATH; otherwise those rows are recorded with an empty size
# and skipped.  The Windows mirror is `scripts/measure_binary_size.ps1`.
#
# Usage:
#   bash scripts/measure_binary_size.sh           # full matrix
#   bash scripts/measure_binary_size.sh host      # skip aarch64 rows
#
# Outputs:
#   binary_sizes.csv         (raw numbers, regenerated each run)
#   docs/BINARY_SIZE.md      (rendered table; via scripts/sizes_to_md.py)

set -euo pipefail

SCOPE="${1:-full}"
case "${SCOPE}" in
    full|host) ;;
    *) echo "unknown scope: ${SCOPE} (expected 'full' or 'host')" >&2; exit 2 ;;
esac

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${REPO_ROOT}"

CSV="binary_sizes.csv"
AARCH64_TARGET="aarch64-unknown-linux-gnu"

# Pipe-separated combos: id|profile|features|target|strip
COMBOS=(
    "A|release|train,inference,matrixmultiply|host|no"
    "B|release|inference|host|no"
    "C|release-edge|inference|host|yes"
    "D|release-edge|inference,quant-i8|host|yes"
    "E|release-edge|inference|aarch64|yes"
    "F|release-edge|inference,quant-i8|aarch64|yes"
)

# Host artifact extension: empty on Linux/mac, ".exe" under MSYS/Cygwin.
HOST_EXT=""
case "$(uname -s)" in
    MINGW*|MSYS*|CYGWIN*) HOST_EXT=".exe" ;;
esac

# Pick an aarch64 backend if the user didn't restrict to host scope.
AARCH64_BUILDER=""
if [ "${SCOPE}" = "full" ]; then
    if command -v cross >/dev/null 2>&1; then
        AARCH64_BUILDER="cross"
    elif command -v cargo-zigbuild >/dev/null 2>&1; then
        AARCH64_BUILDER="zigbuild"
    fi
fi

echo "id,profile,features,target,strip,artifact,bytes" > "${CSV}"

run_combo() {
    local id="$1" profile="$2" features="$3" target="$4" strip="$5"
    echo "[measure] combo=${id} profile=${profile} features=${features} target=${target}"

    local profile_arg=()
    local profile_dir
    case "${profile}" in
        release)      profile_arg=(--release);              profile_dir="release" ;;
        release-edge) profile_arg=(--profile release-edge); profile_dir="release-edge" ;;
        *) echo "  !! unknown profile: ${profile}" >&2; return 1 ;;
    esac

    local builder=() target_arg=() artifact_dir artifact_name
    if [ "${target}" = "host" ]; then
        builder=(cargo)
        artifact_dir="target/${profile_dir}/examples"
        artifact_name="min_inference${HOST_EXT}"
    else
        if [ -z "${AARCH64_BUILDER}" ]; then
            echo "  -> skipping (no cross/cargo-zigbuild on PATH)"
            echo "${id},${profile},\"${features}\",${target},${strip},,," >> "${CSV}"
            return 0
        fi
        case "${AARCH64_BUILDER}" in
            cross)    builder=(cross) ;;
            zigbuild) builder=(cargo zigbuild) ;;
        esac
        target_arg=(--target "${AARCH64_TARGET}")
        artifact_dir="target/${AARCH64_TARGET}/${profile_dir}/examples"
        artifact_name="min_inference"
    fi

    "${builder[@]}" build \
        "${profile_arg[@]}" \
        --no-default-features --features "${features}" \
        "${target_arg[@]}" \
        --example min_inference

    local artifact_path="${artifact_dir}/${artifact_name}"
    if [ ! -f "${artifact_path}" ]; then
        echo "  !! artifact not found: ${artifact_path}" >&2
        echo "${id},${profile},\"${features}\",${target},${strip},${artifact_path}," >> "${CSV}"
        return 0
    fi
    local bytes
    bytes=$(wc -c < "${artifact_path}" | tr -d ' \t\n\r')
    echo "  -> ${bytes} bytes (${artifact_path})"
    echo "${id},${profile},\"${features}\",${target},${strip},${artifact_path},${bytes}" >> "${CSV}"
}

for combo in "${COMBOS[@]}"; do
    IFS='|' read -r id profile features target strip <<< "${combo}"
    run_combo "${id}" "${profile}" "${features}" "${target}" "${strip}"
done

# Render the markdown table.  Tolerate either `python3` or `python` on PATH.
PY=""
if command -v python3 >/dev/null 2>&1; then
    PY="python3"
elif command -v python >/dev/null 2>&1; then
    PY="python"
fi
if [ -n "${PY}" ]; then
    "${PY}" scripts/sizes_to_md.py "${CSV}" docs/BINARY_SIZE.md
else
    echo "[measure] python not found; skipping markdown render" >&2
fi

echo "[measure] done.  CSV: ${CSV}"
