# Phase 10: binary-size measurement automation (Windows mirror).
#
# Mirror of `scripts/measure_binary_size.sh`.  See that file for the full
# rationale and combo matrix definition.
#
# Usage:
#   pwsh scripts/measure_binary_size.ps1                    # full matrix
#   pwsh scripts/measure_binary_size.ps1 -Scope host        # skip aarch64 rows
#
# aarch64 rows require either `cross` (Docker-backed) or `cargo-zigbuild`
# on PATH; otherwise they are recorded with an empty size and skipped.

[CmdletBinding()]
param(
    [ValidateSet("full","host")]
    [string]$Scope = "full"
)

$ErrorActionPreference = "Stop"

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $RepoRoot

$Csv = "binary_sizes.csv"
$Aarch64Target = "aarch64-unknown-linux-gnu"

# id, profile, features, target, strip
$Combos = @(
    [pscustomobject]@{Id="A"; Profile="release";      Features="train,inference,matrixmultiply"; Target="host";    Strip="no"},
    [pscustomobject]@{Id="B"; Profile="release";      Features="inference";                       Target="host";    Strip="no"},
    [pscustomobject]@{Id="C"; Profile="release-edge"; Features="inference";                       Target="host";    Strip="yes"},
    [pscustomobject]@{Id="D"; Profile="release-edge"; Features="inference,quant-i8";              Target="host";    Strip="yes"},
    [pscustomobject]@{Id="E"; Profile="release-edge"; Features="inference";                       Target="aarch64"; Strip="yes"},
    [pscustomobject]@{Id="F"; Profile="release-edge"; Features="inference,quant-i8";              Target="aarch64"; Strip="yes"}
)

# On Windows the host artifact has the .exe extension.
$HostExt = ".exe"

$Aarch64Builder = ""
if ($Scope -eq "full") {
    if (Get-Command cross -ErrorAction SilentlyContinue) {
        $Aarch64Builder = "cross"
    } elseif (Get-Command cargo-zigbuild -ErrorAction SilentlyContinue) {
        $Aarch64Builder = "zigbuild"
    }
}

# CSV header.
"id,profile,features,target,strip,artifact,bytes" | Out-File -FilePath $Csv -Encoding ascii

function Add-CsvRow {
    param([string]$Id, [string]$Profile, [string]$Features, [string]$Target, [string]$Strip, [string]$Artifact, [string]$Bytes)
    $line = '{0},{1},"{2}",{3},{4},{5},{6}' -f $Id, $Profile, $Features, $Target, $Strip, $Artifact, $Bytes
    Add-Content -Path $Csv -Value $line -Encoding ascii
}

foreach ($c in $Combos) {
    Write-Host "[measure] combo=$($c.Id) profile=$($c.Profile) features=$($c.Features) target=$($c.Target)"

    switch ($c.Profile) {
        "release"      { $ProfileArgs = @("--release");                $ProfileDir = "release" }
        "release-edge" { $ProfileArgs = @("--profile","release-edge"); $ProfileDir = "release-edge" }
    }

    if ($c.Target -eq "host") {
        $BuilderCmd = "cargo"
        $BuilderPre = @()
        $TargetArgs = @()
        $ArtifactDir = Join-Path "target" (Join-Path $ProfileDir "examples")
        $ArtifactName = "min_inference$HostExt"
    } else {
        if ([string]::IsNullOrEmpty($Aarch64Builder)) {
            Write-Host "  -> skipping (no cross/cargo-zigbuild on PATH)"
            Add-CsvRow $c.Id $c.Profile $c.Features $c.Target $c.Strip "" ""
            continue
        }
        if ($Aarch64Builder -eq "cross") {
            $BuilderCmd = "cross"
            $BuilderPre = @()
        } else {
            $BuilderCmd = "cargo"
            $BuilderPre = @("zigbuild")
        }
        $TargetArgs = @("--target", $Aarch64Target)
        $ArtifactDir = Join-Path "target" (Join-Path $Aarch64Target (Join-Path $ProfileDir "examples"))
        $ArtifactName = "min_inference"
    }

    & $BuilderCmd @BuilderPre build @ProfileArgs `
        --no-default-features --features $c.Features `
        @TargetArgs `
        --example min_inference
    if ($LASTEXITCODE -ne 0) {
        Write-Error "build failed for combo $($c.Id) (exit $LASTEXITCODE)"
    }

    $ArtifactPath = Join-Path $ArtifactDir $ArtifactName
    if (-not (Test-Path $ArtifactPath)) {
        Write-Warning "artifact not found: $ArtifactPath"
        Add-CsvRow $c.Id $c.Profile $c.Features $c.Target $c.Strip $ArtifactPath ""
        continue
    }
    $Bytes = (Get-Item $ArtifactPath).Length
    Write-Host "  -> $Bytes bytes ($ArtifactPath)"
    # Use forward slashes in the CSV so the markdown table is portable across OSes.
    $ArtifactPathPosix = $ArtifactPath -replace '\\','/'
    Add-CsvRow $c.Id $c.Profile $c.Features $c.Target $c.Strip $ArtifactPathPosix "$Bytes"
}

# Render markdown.  On Windows the canonical interpreter is `python`; the
# `python3` shim resolves to the Microsoft Store stub on stock installs and
# would silently fail.  Try the Python launcher (`py -3`), then `python`,
# then `python3` as a last resort.
$PyCmd = $null
$PyArgs = @()
if (Get-Command py -ErrorAction SilentlyContinue) {
    $PyCmd = "py"
    $PyArgs = @("-3")
} elseif (Get-Command python -ErrorAction SilentlyContinue) {
    $PyCmd = "python"
} elseif (Get-Command python3 -ErrorAction SilentlyContinue) {
    $PyCmd = "python3"
}
if ($null -ne $PyCmd) {
    & $PyCmd @PyArgs scripts/sizes_to_md.py $Csv docs/BINARY_SIZE.md
} else {
    Write-Warning "[measure] python not found; skipping markdown render"
}

Write-Host "[measure] done. CSV: $Csv"
