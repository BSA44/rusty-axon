# Captured measurement logs

Raw stdout from each run that produced a cell value in
[`../COMPARISON.md`](../COMPARISON.md). Kept verbatim so the paper
numbers are auditable -- if you reproduce a run, the new log lands
here next to the old one and the diff is the experimental delta.

| Log                              | Cells it sources                         |
|----------------------------------|------------------------------------------|
| `rpi_finetune_mnist.log`         | Tables 6, 7 (fine-tune row), 8 row 1, plus the personalization-demo summary in the paper text |
| `rpi_sensor_drift.log`           | Tables 8 row 2, 10                       |
| `rpi_inference_axon_rss.log`     | Table 5 (rusty-axon row), single-shot inference latency cross-check |
| `rpi_inference_burn_rss.log`     | Table 5 (Burn row)                       |

Bench-harness logs (criterion stdout for `forward_train`,
`forward_infer_f32`, `forward_infer_i8`, `training_step`,
`finetune_step`, `matmul_kernel`, and the Burn equivalents) live
under each harness's own `target/criterion/` JSON tree -- they're
not re-checked-in here because criterion already preserves them in a
machine-readable form.
