# `.axn` Wire Format — v1

`rusty-axon` serializes models to a single self-describing binary file with the
`.axn` extension. The format is designed for:

- **Edge-friendly loads.** Fixed-size prelude, all offsets random-access; the
  reader needs only `Read + Seek`, no allocator besides the parsed tensors.
- **Reproducibility.** Fully little-endian; no machine-dependent layout.
- **Defensive integrity.** Per-tensor CRC32 plus a header-region CRC32 — the
  paper-grade reproducibility flow detects truncation, bit-flips, and
  unrelated file types before any tensor is read.
- **PTQ-ready.** Phase 7 reuses the same container for INT8 weights via the
  `Dtype::I8` discriminant and the per-tensor `scale` field; no v2 bump.

The reference implementation lives in
[`src/format/axn.rs`](../src/format/axn.rs).  Round-trip + corruption tests
are in [`src/format/axn_tests.rs`](../src/format/axn_tests.rs).

## Layout

All multi-byte fields are **little-endian**. Reading on a big-endian target is
rejected at compile time (`compile_error!` in [`src/format/mod.rs`](../src/format/mod.rs)).

```
offset  size  field
------  ----  ------------------------------------------------
  0      4    magic         = b"AXN\0"
  4      2    version       = 0x0001  (u16)
  6      1    flags         (bit 0: has_int8_quant, bit 1: per_channel_scales [reserved])
  7      1    reserved      = 0
  8      4    num_tensors   (u32)
 12      4    header_len    (u32; covers prelude + tensor headers, padded to 4 B)
 16    var    tensor headers (num_tensors entries; see below)
 ...    pad   zero pad up to header_len (4-byte alignment)
 ...    var   raw tensor data (each tensor 4-byte aligned)
EOF-4    4    crc32 of bytes [0 .. header_len)   (u32, IEEE polynomial)
```

### Tensor header (variable-length)

```
size  field
----  ------------------------------------------------
  2   name_len      (u16)
  N   name          (utf-8, length = name_len)
  1   dtype         (u8; 0 = F32, 1 = I8)
  1   rank          (u8)
4*r   dims          ([u32; rank])
  4   scale         (f32; 0.0 when dtype == F32)
  8   data_offset   (u64; absolute file offset)
  8   data_len      (u64; bytes — equals prod(dims) * elem_size(dtype))
  4   crc32         (u32; IEEE polynomial, of raw tensor bytes)
```

### CRC32

IEEE polynomial `0xEDB88320`, initial value `0xFFFFFFFF`, final XOR
`0xFFFFFFFF`.  Reference vector: `crc32(b"123456789") == 0xCBF43926`.

The implementation is ~25 LoC and inlined in `src/format/axn.rs`; we
deliberately do not depend on `crc32fast` to keep the dependency footprint
small.

## Naming convention used by `Mlp::save` / `Mlp::load`

For an MLP whose `i`th `Linear` layer has shape `in_dim → out_dim`:

| Tensor name        | dtype     | dims                     |
|--------------------|-----------|--------------------------|
| `layer{i}.weight`  | F32 or I8 | `[out_dim, in_dim]` (row-major) |
| `layer{i}.bias`    | F32       | `[out_dim]`              |

Activations are **not** serialized in v1.  `Mlp::load(path, activations)`
takes the activation list explicitly, mirroring how the model is built with
`Mlp::new`.  Storing activations is a candidate for v2 if user feedback
demands it.

## Quantized (INT8) tensors — Phase 7

When `dtype == I8`:

- `scale` is the per-tensor symmetric scale (`scale = max(|w|) / 127`); the
  dequantized value is `f32(qw) * scale`.
- `flags & FLAG_HAS_INT8_QUANT` is set in the prelude.
- Per-channel scales are reserved as `flags` bit 1 for v0.4 if needed.

Biases stay F32 — the storage savings on small bias vectors are negligible
and keeping accumulation in f32 simplifies the dequant-fused matmul kernel.

## Size accounting (paper-baseline reference)

For the canonical MNIST classifier (784 → 64 → 32 → 10):

| Section          | Bytes  |
|------------------|--------|
| Prelude          | 16     |
| 6 tensor headers | ~210   |
| Header pad to 4 B alignment | 0–3 |
| Weight + bias data (F32) | 222,392 |
| Trailing CRC     | 4      |
| **Total**        | **≈ 217 KB** |

INT8 quantization (Phase 7) drops the weight section by 4× (~55 KB) while
the F32 biases (~424 B) are unchanged, landing the file under 60 KB —
matches the binary-size table target in `docs/PAPER_REWORK_PLAN.md` Phase 7.

## Forward / backward compatibility

- `version` is a hard equality check; bumping it is breaking.  v1 readers
  reject v2+ files explicitly.
- New optional metadata can be added by appending tensors with a reserved
  name prefix (e.g. `__meta__.<key>`); v1 readers will surface them in
  `tensors()` and a v2-aware loader can pick them up.  The reserved-name
  convention keeps the format additive without a version bump.
