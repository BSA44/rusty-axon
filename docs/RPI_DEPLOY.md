# Raspberry Pi Zero 2 W deployment

Phase 9 of the [paper rework plan](PAPER_REWORK_PLAN.md) wires `rusty-axon`
up for cross-compilation to the Raspberry Pi Zero 2 W.  This document covers
the full toolchain, the build commands, flashing the Pi, and the smoke
tests that validate a deployed binary.

> **Important — 64-bit Pi OS only.**  The Pi Zero 2 W has a 64-bit
> Cortex-A53 CPU.  rusty-axon targets `aarch64-unknown-linux-gnu`; the
> 32-bit (`armv6` / `armv7`) Raspberry Pi OS images **will not run** these
> binaries.  Flash **Raspberry Pi OS Lite (64-bit)** (Bookworm or newer).

## 1. Target selection rationale

| Property             | Value                                                  |
|----------------------|--------------------------------------------------------|
| Triple               | `aarch64-unknown-linux-gnu`                            |
| Tuned CPU            | `cortex-a53` (Broadcom BCM2710A1, 4 cores @ 1 GHz)     |
| SIMD                 | NEON (ARMv8 advanced SIMD; auto-used by matrixmultiply)|
| RAM                  | 512 MB                                                 |
| Toolchain            | pinned in [`rust-toolchain.toml`](../rust-toolchain.toml) |
| Cargo profile        | `release-edge` (LTO, panic=abort, opt-level=z, strip)  |

The compile flags live in [`.cargo/config.toml`](../.cargo/config.toml):

```toml
[target.aarch64-unknown-linux-gnu]
linker = "aarch64-linux-gnu-gcc"
rustflags = ["-C", "target-cpu=cortex-a53", "-C", "target-feature=+neon"]
```

Binaries built with these flags run on Pi 3 / 4 / 5 (newer ARMv8 cores)
unchanged but are scheduled for the A53 in-order pipeline.

## 2. Two build backends

Both produce identical artifacts under
`target/aarch64-unknown-linux-gnu/release-edge/examples/`.

### 2a. `cross` (recommended; matches CI)

`cross` runs the build inside a Docker container that already has
`aarch64-linux-gnu-gcc` and `binutils-aarch64-linux-gnu` installed, so the
host needs only Docker (or Podman).

```bash
cargo install cross --locked --git https://github.com/cross-rs/cross

# Inference-only artifacts (Phase 10's binary-size matrix lead).
cross build --profile release-edge \
    --no-default-features --features inference \
    --target aarch64-unknown-linux-gnu --example rpi_inference
cross build --profile release-edge \
    --no-default-features --features inference \
    --target aarch64-unknown-linux-gnu --example min_inference

# Train-capable artifacts (Phase 11 fine-tune demos land here).
cross build --profile release-edge \
    --no-default-features --features train,matrixmultiply \
    --target aarch64-unknown-linux-gnu --examples
```

The driver `scripts/build_rpi.sh` wraps these commands for both `cross` and
`cargo-zigbuild`:

```bash
bash scripts/build_rpi.sh                # cross + release-edge (default)
bash scripts/build_rpi.sh zigbuild       # cargo-zigbuild backend
bash scripts/build_rpi.sh cross debug    # debug profile (qemu-aarch64 friendly)
```

PowerShell mirror: `pwsh scripts/build_rpi.ps1`.

### 2b. `cargo-zigbuild` (Docker-free; explicit glibc pin)

`cargo-zigbuild` uses Zig as the cross-linker.  No Docker required, and it
lets you pin the **maximum glibc version** the binary depends on — useful
when targeting older Pi OS images.

```bash
cargo install cargo-zigbuild
rustup target add aarch64-unknown-linux-gnu

# Pi OS Bookworm has glibc 2.36; Bullseye has 2.31.  Pin to 2.31 if you
# want a single artifact that works on either.
cargo zigbuild --profile release-edge \
    --no-default-features --features inference \
    --target aarch64-unknown-linux-gnu.2.31 \
    --example rpi_inference
```

**glibc version note.**  The default `cross` image is Ubuntu 20.04
(glibc 2.31); its symbol set links forward-compatibly against newer
glibcs (2.36 / 2.38 on Pi OS Bookworm / Trixie).  If you cross-compile
without a Docker image and end up linking against a glibc newer than the
target Pi's, you'll see `GLIBC_2.x not found` errors at runtime — pin
explicitly via `cargo-zigbuild`.

## 3. Stripping

`release-edge` already sets `strip = "symbols"`, so artifacts are stripped
post-link.  `scripts/build_rpi.sh` additionally runs `aarch64-linux-gnu-strip
--strip-unneeded` as belt-and-braces; the second strip is a no-op when the
profile setting already did the work but is harmless and stays compatible
with hosts whose Cargo predates the `strip` profile key.

## 4. Verifying NEON / matrixmultiply made it into the binary

`matrixmultiply` 0.3 auto-selects an aarch64 NEON kernel at compile time
when the target supports it.  Confirm with `objdump`:

```bash
aarch64-linux-gnu-objdump -d \
    target/aarch64-unknown-linux-gnu/release-edge/examples/rpi_inference \
    | grep -E '\b(fmla|fmul|fadd)\b\s+v[0-9]+' | head
```

`fmla v…` (vector fused multiply-add) is the headline NEON FP SIMD
instruction inside the matmul inner loop.  An empty result means the NEON
path didn't compile in (rare; usually a misconfigured `target-feature`).

The CI job `rpi-cross` performs this audit automatically — see
[`.github/workflows/ci.yml`](../.github/workflows/ci.yml).

## 5. Flashing 64-bit Pi OS Lite

1. Download **Raspberry Pi OS Lite (64-bit)** from
   <https://www.raspberrypi.com/software/operating-systems/>.
2. Flash to a microSD card with **Raspberry Pi Imager** (`>= 1.7.5`).
   Use the imager's *settings cog* to set hostname, enable SSH, configure
   the WPA Personal network, and create a user account.  This avoids
   needing a keyboard / monitor for first boot.
3. Insert the card, power on, wait ~30 s for first boot, then SSH in.

## 6. Copying and running

```bash
ART=target/aarch64-unknown-linux-gnu/release-edge/examples
scp $ART/rpi_inference  pi@rpi-zero:/home/pi/
scp $ART/min_inference  pi@rpi-zero:/home/pi/
scp examples/your_model.axn pi@rpi-zero:/home/pi/

ssh pi@rpi-zero ./rpi_inference your_model.axn 4 1000
```

`rpi_inference` prints architecture, arena size, median + p95 latency,
RSS before/after, and the first four output coordinates.  Phase 11 layers
the fine-tune (`rpi_finetune_mnist`, `rpi_sensor_drift`) binaries on top
of this baseline.

## 7. Smoke-testing without a Pi (`qemu-aarch64`)

For a quick functional check on a development host:

```bash
sudo apt-get install qemu-user-static
qemu-aarch64-static \
    target/aarch64-unknown-linux-gnu/release-edge/examples/rpi_inference \
    your_model.axn 4 100
```

`qemu-user-static` does not model the A53 micro-architecture, so reported
latencies are meaningless — but it validates that the binary loads,
parses `.axn`, and runs to completion without dynamic-linker errors.

## 8. Troubleshooting

| Symptom                                     | Cause / fix                                                                                  |
|---------------------------------------------|-----------------------------------------------------------------------------------------------|
| `cannot execute binary file: Exec format error` | 32-bit Pi OS.  Reflash with the 64-bit image.                                                 |
| `version 'GLIBC_2.36' not found`            | Host cross-compiler links against newer glibc than the target Pi.  Use `cargo-zigbuild` with `--target aarch64-unknown-linux-gnu.2.31`. |
| `aarch64-linux-gnu-gcc: not found`          | Install `gcc-aarch64-linux-gnu` (Debian/Ubuntu) or use the `cross` backend.                   |
| `failed to load 'model.axn': Crc32Mismatch` | File was truncated in transit.  Re-`scp`; run `sha256sum` on both sides to confirm.           |
| `cross: error: container engine`            | Docker daemon not running.  `systemctl start docker` (Linux) or launch Docker Desktop (Win/Mac). |

## 9. What's wired up where

- [`.cargo/config.toml`](../.cargo/config.toml) — linker + `target-cpu` + `+neon`.
- [`Cross.toml`](../Cross.toml) — `cross` configuration (env passthrough).
- [`scripts/build_rpi.sh`](../scripts/build_rpi.sh), [`scripts/build_rpi.ps1`](../scripts/build_rpi.ps1) — the canonical build commands.
- [`examples/rpi_inference.rs`](../examples/rpi_inference.rs) — minimal arena-backed inference demo.
- [`.github/workflows/ci.yml`](../.github/workflows/ci.yml) — `rpi-cross` job runs on every push.
