//! Phase 7 — INT8 weights-only post-training quantization.
//!
//! Per-tensor symmetric quantization: each weight tensor `W: [out, in]` is
//! mapped to `(qW: Vec<i8>, scale: f32)` via
//!
//! ```text
//! scale = max(|W|) / 127
//! qW[i] = clamp(round(W[i] / scale), -127, 127)
//! ```
//!
//! Biases stay in f32 (so `Mlp::save_quantized` writes a mix of F32 and I8
//! tensors into the same `.axn` file).  Inference dequantizes on the fly:
//! `y = bias + scale * (qW @ x)`.
//!
//! **INT8 = inference only.**  The training surface (`Mlp::forward`,
//! `Linear::forward`) panics if any layer is quantized — fine-tuning requires
//! reloading f32 weights and re-quantizing afterwards.  No QAT, no STE.
//!
//! Two dispatch strategies on the dequant-fused matmul, chosen at runtime by
//! the `m * k` size of the weight matrix:
//! - **Small** (`m * k <= QUANT_DISPATCH_THRESHOLD`): scalar loop, one cast
//!   per (i, j) — no scratch allocation.
//! - **Large** (`m * k >  QUANT_DISPATCH_THRESHOLD`): dequantize the full
//!   weight matrix into a scratch `Vec<f32>` then route through `sgemm_rm`.
//!   Cheaper per-FLOP at scale; the per-call allocation goes away once the
//!   Phase 8 arena is wired in.

use crate::nn::matmul::sgemm_rm;

/// Element-count threshold below which the dequant-fused matmul stays in a
/// scalar loop instead of allocating a full f32 scratch matrix.  Tuned to
/// match the plan (4096 elements ≈ a 64×64 weight matrix).
pub const QUANT_DISPATCH_THRESHOLD: usize = 4096;

/// Per-tensor symmetric quantization.
///
/// Returns `(qweights, scale)` where `qweights[i] = round(w[i] / scale)`
/// clamped to `[-127, 127]` (we deliberately keep the negative end at -127
/// rather than -128 so the quantization grid stays symmetric around zero —
/// this matches `dequantize`'s assumption and the standard PTQ recipe).
///
/// An all-zero tensor returns `scale = 1.0` so the round-trip
/// `dequantize(quantize(w))` still yields zeros without dividing by zero.
pub fn quantize_per_tensor_symmetric(w: &[f32]) -> (Vec<i8>, f32) {
    let max_abs = w.iter().fold(0.0_f32, |acc, &v| acc.max(v.abs()));
    let scale = if max_abs > 0.0 { max_abs / 127.0 } else { 1.0 };
    let inv_scale = 1.0 / scale;
    let qweights: Vec<i8> = w
        .iter()
        .map(|&v| {
            let q = (v * inv_scale).round();
            // Clip to [-127, 127] to stay symmetric.
            q.clamp(-127.0, 127.0) as i8
        })
        .collect();
    (qweights, scale)
}

/// Inverse of [`quantize_per_tensor_symmetric`].  Writes `out[i] = scale *
/// qweights[i] as f32`.
///
/// # Panics
///
/// Panics if `out.len() != qweights.len()`.
pub fn dequantize(qweights: &[i8], scale: f32, out: &mut [f32]) {
    assert_eq!(
        out.len(),
        qweights.len(),
        "dequantize: out length must match qweights length"
    );
    for (o, &q) in out.iter_mut().zip(qweights.iter()) {
        *o = scale * (q as f32);
    }
}

/// Fused dequantize + matvec + bias add: `output = bias + scale * (qW @ x)`.
///
/// `qweights` is row-major `[out_dim, in_dim]`; `bias` is `[out_dim]`;
/// `input` is `[in_dim]`; `output` is `[out_dim]` (overwritten).  Picks
/// between the scalar loop and the sgemm-with-scratch strategies based on
/// `out_dim * in_dim` vs [`QUANT_DISPATCH_THRESHOLD`].
///
/// # Panics
///
/// Panics if any slice length disagrees with `out_dim` / `in_dim`.
pub fn quant_matvec(
    out_dim: usize,
    in_dim: usize,
    qweights: &[i8],
    scale: f32,
    bias: &[f32],
    input: &[f32],
    output: &mut [f32],
) {
    assert_eq!(qweights.len(), out_dim * in_dim, "qweights length mismatch");
    assert_eq!(bias.len(), out_dim, "bias length mismatch");
    assert_eq!(input.len(), in_dim, "input length mismatch");
    assert_eq!(output.len(), out_dim, "output length mismatch");

    if out_dim * in_dim <= QUANT_DISPATCH_THRESHOLD {
        // Scalar path: one cast per (i, j), no scratch allocation.  Pulls
        // `bias[i]` in via the accumulator init so we never zero `output`
        // first (caller may reuse the same buffer across layers).
        for i in 0..out_dim {
            let row_off = i * in_dim;
            let mut acc = 0.0_f32;
            for j in 0..in_dim {
                acc += (qweights[row_off + j] as f32) * input[j];
            }
            output[i] = bias[i] + scale * acc;
        }
    } else {
        // Large path: dequant the full matrix into scratch, then route
        // through the same sgemm_rm kernel as f32 inference.  Phase 8's
        // arena will replace this Vec with a cached buffer.
        let mut scratch = vec![0.0_f32; out_dim * in_dim];
        dequantize(qweights, scale, &mut scratch);
        output.copy_from_slice(bias);
        sgemm_rm(out_dim, in_dim, 1, 1.0, &scratch, in_dim, input, 1, 1.0, output, 1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .fold(0.0_f32, |acc, (x, y)| acc.max((x - y).abs()))
    }

    #[test]
    fn quantize_round_trip_within_scale() {
        let w: Vec<f32> = (0..32).map(|i| (i as f32 - 16.0) * 0.1).collect();
        let (qw, scale) = quantize_per_tensor_symmetric(&w);
        let mut back = vec![0.0_f32; w.len()];
        dequantize(&qw, scale, &mut back);
        // Max round-trip error must not exceed `scale` (= half-resolution
        // bound of the quantization grid is `scale / 2`, but rounding can
        // walk one full step in pathological cases).
        let err = max_abs_diff(&w, &back);
        assert!(err <= scale, "round-trip error {} exceeds scale {}", err, scale);
    }

    #[test]
    fn quantize_zeros_does_not_panic() {
        let w = vec![0.0_f32; 8];
        let (qw, scale) = quantize_per_tensor_symmetric(&w);
        assert!(qw.iter().all(|&q| q == 0));
        // scale must be finite and non-zero so dequantize is well-defined.
        assert!(scale > 0.0 && scale.is_finite());
    }

    #[test]
    fn quantize_clips_to_symmetric_range() {
        // Pathological tensor where one entry sits exactly at +max_abs.
        // The clip should keep us in [-127, 127], never hit -128.
        let w = vec![1.0_f32, -1.0, 0.5, -0.5];
        let (qw, _) = quantize_per_tensor_symmetric(&w);
        assert!(qw.iter().all(|&q| q >= -127));
    }

    #[test]
    fn quant_matvec_scalar_path_matches_reference() {
        // Small layer: hits the scalar-loop branch.
        let out_dim = 4;
        let in_dim = 3;
        let w = vec![
            0.1_f32, -0.2, 0.3, 0.4, -0.5, 0.6, -0.7, 0.8, -0.9, 1.0, -1.1, 1.2,
        ];
        let bias = vec![0.01_f32, 0.02, -0.03, 0.04];
        let input = vec![1.5_f32, -2.5, 3.5];

        let (qw, scale) = quantize_per_tensor_symmetric(&w);

        // Reference: dequantize then full f32 matvec.
        let mut wf = vec![0.0_f32; out_dim * in_dim];
        dequantize(&qw, scale, &mut wf);
        let mut reference = bias.clone();
        for i in 0..out_dim {
            for j in 0..in_dim {
                reference[i] += wf[i * in_dim + j] * input[j];
            }
        }

        let mut got = vec![0.0_f32; out_dim];
        quant_matvec(out_dim, in_dim, &qw, scale, &bias, &input, &mut got);
        assert!(
            max_abs_diff(&got, &reference) < 1e-5,
            "scalar quant matvec disagrees with reference: got {:?} vs {:?}",
            got,
            reference,
        );
    }

    #[test]
    fn quant_matvec_large_path_matches_reference() {
        // Force the sgemm-with-scratch path by crossing the threshold.
        let out_dim = 64;
        let in_dim = 80; // 64 * 80 = 5120 > 4096
        let w: Vec<f32> = (0..out_dim * in_dim)
            .map(|i| ((i as f32) * 0.013).sin())
            .collect();
        let bias: Vec<f32> = (0..out_dim).map(|i| (i as f32) * 0.001).collect();
        let input: Vec<f32> = (0..in_dim).map(|i| ((i as f32) * 0.07).cos()).collect();

        let (qw, scale) = quantize_per_tensor_symmetric(&w);

        let mut wf = vec![0.0_f32; out_dim * in_dim];
        dequantize(&qw, scale, &mut wf);
        let mut reference = bias.clone();
        for i in 0..out_dim {
            for j in 0..in_dim {
                reference[i] += wf[i * in_dim + j] * input[j];
            }
        }

        let mut got = vec![0.0_f32; out_dim];
        quant_matvec(out_dim, in_dim, &qw, scale, &bias, &input, &mut got);
        assert!(
            max_abs_diff(&got, &reference) < 1e-3,
            "sgemm quant matvec disagrees with reference (max diff {})",
            max_abs_diff(&got, &reference),
        );
    }
}
