# Phase 9: cross-compile the rusty-axon Pi Zero 2 W demo binaries (Windows).
#
# Mirror of `scripts/build_rpi.sh`.  See that file for the full rationale.
# Note: `cross` requires a working Docker / Podman daemon on Windows; the
# WSL path also works via `bash scripts/build_rpi.sh` if Docker Desktop
# is not installed.
#
# Usage:
#   pwsh scripts/build_rpi.ps1                       # cross + release-edge
#   pwsh scripts/build_rpi.ps1 -Backend zigbuild     # cargo-zigbuild instead
#   pwsh scripts/build_rpi.ps1 -Profile debug        # qemu-friendly debug

[CmdletBinding()]
param(
    [ValidateSet("cross","zigbuild")]
    [string]$Backend = "cross",
    [ValidateSet("release-edge","release","debug")]
    [string]$Profile = "release-edge"
)

$ErrorActionPreference = "Stop"
$Target = "aarch64-unknown-linux-gnu"

switch ($Profile) {
    "release-edge" { $ProfileArgs = @("--profile","release-edge"); $OutDir = "release-edge" }
    "release"      { $ProfileArgs = @("--release");                $OutDir = "release" }
    "debug"        { $ProfileArgs = @();                           $OutDir = "debug" }
}

switch ($Backend) {
    "cross" {
        if (-not (Get-Command cross -ErrorAction SilentlyContinue)) {
            Write-Error "'cross' not found. Install with: cargo install cross --git https://github.com/cross-rs/cross"
        }
        $Builder = @("cross")
    }
    "zigbuild" {
        if (-not (Get-Command cargo-zigbuild -ErrorAction SilentlyContinue)) {
            Write-Error "'cargo-zigbuild' not found. Install with: cargo install cargo-zigbuild"
        }
        $Builder = @("cargo","zigbuild")
    }
}

Write-Host "[build_rpi] backend=$Backend profile=$Profile target=$Target"

$BuildCmd = $Builder[0]
$BuildPre = @()
if ($Builder.Count -gt 1) { $BuildPre = $Builder[1..($Builder.Count - 1)] }

# Inference-only artifacts.
& $BuildCmd @BuildPre build @ProfileArgs `
    --no-default-features --features inference `
    --target $Target --example min_inference
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $BuildCmd @BuildPre build @ProfileArgs `
    --no-default-features --features inference `
    --target $Target --example rpi_inference
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

# Train-capable artifacts (Phase 11 demos land here).
& $BuildCmd @BuildPre build @ProfileArgs `
    --no-default-features --features train,matrixmultiply `
    --target $Target --examples
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$ArtifactDir = Join-Path (Join-Path (Join-Path "target" $Target) $OutDir) "examples"
if ((Get-Command aarch64-linux-gnu-strip -ErrorAction SilentlyContinue) -and (Test-Path $ArtifactDir)) {
    Write-Host "[build_rpi] stripping artifacts in $ArtifactDir"
    Get-ChildItem -Path $ArtifactDir -File |
        Where-Object { $_.Name -notlike "*.d" -and $_.Name -notlike "*-*" } |
        ForEach-Object { & aarch64-linux-gnu-strip --strip-unneeded $_.FullName }
}

Write-Host "[build_rpi] done. artifacts in $ArtifactDir/"
