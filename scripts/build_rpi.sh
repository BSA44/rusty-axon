#!/usr/bin/env bash
# Phase 9: cross-compile the rusty-axon Pi Zero 2 W demo binaries.
#
# Two backends, both produce the same artifacts under
#   target/aarch64-unknown-linux-gnu/release-edge/examples/
#
#   - cross (default): Docker-backed, matches the CI cross-compile job.
#   - cargo-zigbuild:  Docker-free; lets the user pin glibc explicitly via
#                      `--target aarch64-unknown-linux-gnu.2.31`.
#
# Usage:
#   bash scripts/build_rpi.sh                # cross + release-edge (default)
#   bash scripts/build_rpi.sh zigbuild       # cargo-zigbuild instead
#   bash scripts/build_rpi.sh cross debug    # debug profile, helpful for `qemu-aarch64`
#
# After build, run `aarch64-linux-gnu-strip` on the artifacts so the
# Phase 10 binary-size table reflects stripped sizes.  `cross` images
# already provide that strip binary; `zigbuild` users on bare hosts may
# need the `gcc-aarch64-linux-gnu` package (Debian/Ubuntu) for it.

set -euo pipefail

BACKEND="${1:-cross}"
PROFILE_FLAG="${2:-release-edge}"
TARGET="aarch64-unknown-linux-gnu"

case "${PROFILE_FLAG}" in
    release-edge) PROFILE_ARG=(--profile release-edge); OUTDIR="release-edge" ;;
    release)      PROFILE_ARG=(--release);              OUTDIR="release" ;;
    debug)        PROFILE_ARG=();                       OUTDIR="debug" ;;
    *) echo "unknown profile: ${PROFILE_FLAG}" >&2; exit 2 ;;
esac

case "${BACKEND}" in
    cross)
        if ! command -v cross >/dev/null 2>&1; then
            echo "error: 'cross' not found.  Install with:" >&2
            echo "  cargo install cross --git https://github.com/cross-rs/cross" >&2
            exit 1
        fi
        BUILDER=(cross)
        ;;
    zigbuild)
        if ! command -v cargo-zigbuild >/dev/null 2>&1; then
            echo "error: 'cargo-zigbuild' not found.  Install with:" >&2
            echo "  cargo install cargo-zigbuild" >&2
            exit 1
        fi
        BUILDER=(cargo zigbuild)
        ;;
    *) echo "unknown backend: ${BACKEND}; expected 'cross' or 'zigbuild'" >&2; exit 2 ;;
esac

echo "[build_rpi] backend=${BACKEND} profile=${PROFILE_FLAG} target=${TARGET}"

# --- Inference-only example (Phase 10's binary-size matrix lead) ---
"${BUILDER[@]}" build "${PROFILE_ARG[@]}" \
    --no-default-features --features inference \
    --target "${TARGET}" --example min_inference

"${BUILDER[@]}" build "${PROFILE_ARG[@]}" \
    --no-default-features --features inference \
    --target "${TARGET}" --example rpi_inference

# --- Train-capable example (Phase 11 fine-tune demo target) ---
# Phase 11 will add `rpi_finetune_mnist` and `rpi_sensor_drift`; build them
# here too once they exist (skipped silently if absent).
"${BUILDER[@]}" build "${PROFILE_ARG[@]}" \
    --no-default-features --features train,matrixmultiply \
    --target "${TARGET}" --examples

# --- Strip stripped artifacts for binary-size table reproducibility. ---
ARTIFACT_DIR="target/${TARGET}/${OUTDIR}/examples"
if command -v aarch64-linux-gnu-strip >/dev/null 2>&1 && [ -d "${ARTIFACT_DIR}" ]; then
    echo "[build_rpi] stripping artifacts in ${ARTIFACT_DIR}"
    find "${ARTIFACT_DIR}" -maxdepth 1 -type f -executable \
        -not -name '*.d' -not -name '*-*' \
        -exec aarch64-linux-gnu-strip --strip-unneeded {} +
fi

echo "[build_rpi] done.  artifacts in ${ARTIFACT_DIR}/"
