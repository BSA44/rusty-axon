//! Wire format for `.axn` model files.
//!
//! Layout (little-endian throughout):
//!
//! ```text
//! 0   4   magic = b"AXN\0"
//! 4   2   version = 0x0001
//! 6   1   flags  (bit 0: has_int8_quant, bit 1: per_channel_scales [reserved])
//! 7   1   reserved = 0
//! 8   4   num_tensors:  u32
//! 12  4   header_len:   u32   (covers prelude + tensor headers, padded to 4 B)
//! 16  ... tensor headers, each:
//!         2   name_len: u16
//!         N   name:     utf8
//!         1   dtype:    u8     (0 = F32, 1 = I8)
//!         1   rank:     u8
//!         4r  dims:     [u32; rank]
//!         4   scale:    f32    (0.0 if not quantized)
//!         8   data_offset: u64 (absolute file offset)
//!         8   data_len:    u64 (bytes)
//!         4   crc32:       u32 (IEEE polynomial, of raw tensor bytes)
//!     ... raw tensor bytes (each tensor 4-byte aligned) ...
//! final 4 bytes: crc32 of the header region [0 .. header_len)
//! ```
//!
//! Tensor naming convention used by `Mlp::save` / `Mlp::load`:
//! - `layer{N}.weight` — row-major `[out_dim, in_dim]`, dtype F32 or I8.
//! - `layer{N}.bias`   — `[out_dim]`, always F32 (Phase 7 keeps biases f32).

use std::io::{self, Read, Seek, SeekFrom, Write};

pub const MAGIC: [u8; 4] = *b"AXN\0";
pub const VERSION: u16 = 0x0001;
pub const FLAG_HAS_INT8_QUANT: u8 = 0x01;

/// Tensor element type recorded in each header.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Dtype {
    F32 = 0,
    I8 = 1,
}

impl Dtype {
    fn from_byte(b: u8) -> io::Result<Self> {
        match b {
            0 => Ok(Dtype::F32),
            1 => Ok(Dtype::I8),
            other => Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("unknown dtype byte: {}", other),
            )),
        }
    }

    /// Bytes per element.
    pub fn elem_size(self) -> usize {
        match self {
            Dtype::F32 => 4,
            Dtype::I8 => 1,
        }
    }
}

/// Parsed tensor metadata; produced by [`AxnReader::open`] and consumed by
/// [`AxnReader::read_tensor_f32`] / [`AxnReader::read_tensor_i8`].
#[derive(Debug, Clone)]
pub struct TensorMeta {
    pub name: String,
    pub dtype: Dtype,
    pub dims: Vec<u32>,
    /// Quantization scale; `0.0` when `dtype == F32`.
    pub scale: f32,
    pub data_offset: u64,
    pub data_len: u64,
    pub data_crc32: u32,
}

impl TensorMeta {
    pub fn num_elements(&self) -> usize {
        self.dims.iter().map(|&d| d as usize).product()
    }
}

// ---------------------------------------------------------------------------
// Writer
// ---------------------------------------------------------------------------

struct PendingTensor {
    name: String,
    dtype: Dtype,
    dims: Vec<u32>,
    scale: f32,
    data: Vec<u8>,
    data_crc32: u32,
}

/// Streaming-style writer.  Tensors are buffered in memory until [`finish`]
/// is called, at which point the whole file is laid out and written in a
/// single pass.  Buffering keeps the header layout simple and avoids
/// rewinding `Write` implementations that aren't truly seekable in practice
/// (e.g. `BufWriter` over stdout would not be).
///
/// [`finish`]: AxnWriter::finish
pub struct AxnWriter<W: Write + Seek> {
    inner: W,
    tensors: Vec<PendingTensor>,
    has_int8: bool,
}

impl<W: Write + Seek> AxnWriter<W> {
    pub fn new(inner: W) -> Self {
        Self {
            inner,
            tensors: Vec::new(),
            has_int8: false,
        }
    }

    /// Append an F32 tensor.  `dims` is the logical shape (e.g. `[out, in]`
    /// for a Linear weight matrix); `data.len()` must equal `prod(dims)`.
    pub fn add_tensor_f32(&mut self, name: &str, dims: &[u32], data: &[f32]) {
        assert_eq!(
            data.len(),
            prod_dims(dims),
            "tensor `{}`: data length {} does not match dims {:?}",
            name,
            data.len(),
            dims
        );
        let bytes = f32_slice_to_le_bytes(data);
        self.push_tensor(name, Dtype::F32, dims, 0.0, bytes);
    }

    /// Append an I8 quantized tensor with its per-tensor `scale`.
    pub fn add_tensor_i8(&mut self, name: &str, dims: &[u32], scale: f32, data: &[i8]) {
        assert_eq!(
            data.len(),
            prod_dims(dims),
            "tensor `{}`: data length {} does not match dims {:?}",
            name,
            data.len(),
            dims
        );
        // SAFETY: i8 and u8 share the same layout and validity invariants.
        let bytes: Vec<u8> = data.iter().map(|&x| x as u8).collect();
        self.has_int8 = true;
        self.push_tensor(name, Dtype::I8, dims, scale, bytes);
    }

    fn push_tensor(&mut self, name: &str, dtype: Dtype, dims: &[u32], scale: f32, data: Vec<u8>) {
        assert!(
            name.len() <= u16::MAX as usize,
            "tensor name longer than u16::MAX bytes"
        );
        assert!(dims.len() <= u8::MAX as usize, "tensor rank exceeds u8::MAX");
        let crc = crc32(&data);
        self.tensors.push(PendingTensor {
            name: name.to_string(),
            dtype,
            dims: dims.to_vec(),
            scale,
            data,
            data_crc32: crc,
        });
    }

    /// Lay out and emit the file.  Returns the inner writer.
    pub fn finish(mut self) -> io::Result<W> {
        // Compute header sizes.
        let mut header_len: usize = 16;
        for t in &self.tensors {
            header_len += tensor_header_size(&t.name, t.dims.len());
        }
        // Pad the header region to a 4-byte boundary so tensor data is aligned.
        let header_len_aligned = align_up(header_len, 4);

        // Pre-compute data offsets (each tensor 4-byte aligned).
        let mut data_offsets: Vec<u64> = Vec::with_capacity(self.tensors.len());
        let mut cursor = header_len_aligned as u64;
        for t in &self.tensors {
            data_offsets.push(cursor);
            cursor += t.data.len() as u64;
            cursor = align_up(cursor as usize, 4) as u64;
        }

        // Build the header region in memory so we can CRC it before writing.
        let mut header = Vec::with_capacity(header_len_aligned);
        header.extend_from_slice(&MAGIC);
        header.extend_from_slice(&VERSION.to_le_bytes());
        let flags: u8 = if self.has_int8 {
            FLAG_HAS_INT8_QUANT
        } else {
            0
        };
        header.push(flags);
        header.push(0); // reserved
        header.extend_from_slice(&(self.tensors.len() as u32).to_le_bytes());
        header.extend_from_slice(&(header_len_aligned as u32).to_le_bytes());

        for (t, &offset) in self.tensors.iter().zip(data_offsets.iter()) {
            header.extend_from_slice(&(t.name.len() as u16).to_le_bytes());
            header.extend_from_slice(t.name.as_bytes());
            header.push(t.dtype as u8);
            header.push(t.dims.len() as u8);
            for &d in &t.dims {
                header.extend_from_slice(&d.to_le_bytes());
            }
            header.extend_from_slice(&t.scale.to_le_bytes());
            header.extend_from_slice(&offset.to_le_bytes());
            header.extend_from_slice(&(t.data.len() as u64).to_le_bytes());
            header.extend_from_slice(&t.data_crc32.to_le_bytes());
        }
        // Pad header to alignment with zeros.
        while header.len() < header_len_aligned {
            header.push(0);
        }
        debug_assert_eq!(header.len(), header_len_aligned);

        let header_crc = crc32(&header);

        // Emit: header, then tensor data with inter-tensor padding, then trailing CRC.
        self.inner.seek(SeekFrom::Start(0))?;
        self.inner.write_all(&header)?;
        let mut written = header.len() as u64;
        for (t, &offset) in self.tensors.iter().zip(data_offsets.iter()) {
            // Pad to this tensor's offset.
            while written < offset {
                self.inner.write_all(&[0u8])?;
                written += 1;
            }
            self.inner.write_all(&t.data)?;
            written += t.data.len() as u64;
        }
        self.inner.write_all(&header_crc.to_le_bytes())?;
        self.inner.flush()?;
        Ok(self.inner)
    }
}

fn tensor_header_size(name: &str, rank: usize) -> usize {
    // name_len(2) + name(N) + dtype(1) + rank(1) + dims(4r) + scale(4)
    // + data_offset(8) + data_len(8) + crc32(4)
    2 + name.len() + 1 + 1 + 4 * rank + 4 + 8 + 8 + 4
}

fn prod_dims(dims: &[u32]) -> usize {
    dims.iter().map(|&d| d as usize).product()
}

fn align_up(value: usize, align: usize) -> usize {
    (value + align - 1) & !(align - 1)
}

fn f32_slice_to_le_bytes(data: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len() * 4);
    for &x in data {
        out.extend_from_slice(&x.to_le_bytes());
    }
    out
}

// ---------------------------------------------------------------------------
// Reader
// ---------------------------------------------------------------------------

/// Random-access reader.  Parses and validates the header region at
/// [`AxnReader::open`]; tensor data is read on demand and CRC-checked at
/// each read.
pub struct AxnReader<R: Read + Seek> {
    inner: R,
    tensors: Vec<TensorMeta>,
    #[allow(dead_code)]
    flags: u8,
}

impl<R: Read + Seek> AxnReader<R> {
    /// Parse the prelude + tensor headers, validate the trailing header CRC,
    /// and surface tensor metadata.
    pub fn open(mut inner: R) -> io::Result<Self> {
        let total_len = inner.seek(SeekFrom::End(0))?;
        if total_len < 16 + 4 {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "file too small to be a valid .axn",
            ));
        }
        inner.seek(SeekFrom::Start(0))?;

        let mut prelude = [0u8; 16];
        inner.read_exact(&mut prelude)?;
        if prelude[0..4] != MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "bad magic: not an .axn file",
            ));
        }
        let version = u16::from_le_bytes([prelude[4], prelude[5]]);
        if version != VERSION {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("unsupported .axn version: {:#06x}", version),
            ));
        }
        let flags = prelude[6];
        // prelude[7] reserved
        let num_tensors = u32::from_le_bytes([prelude[8], prelude[9], prelude[10], prelude[11]]);
        let header_len =
            u32::from_le_bytes([prelude[12], prelude[13], prelude[14], prelude[15]]) as u64;

        if header_len < 16 || header_len + 4 > total_len {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "header_len out of bounds",
            ));
        }

        // Read the entire header region for CRC validation.
        let mut header = vec![0u8; header_len as usize];
        inner.seek(SeekFrom::Start(0))?;
        inner.read_exact(&mut header)?;

        // Trailing header-region CRC sits at the end of the file.
        inner.seek(SeekFrom::Start(total_len - 4))?;
        let mut trail = [0u8; 4];
        inner.read_exact(&mut trail)?;
        let expected_header_crc = u32::from_le_bytes(trail);
        let actual_header_crc = crc32(&header);
        if actual_header_crc != expected_header_crc {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "header CRC32 mismatch (file truncated or corrupted)",
            ));
        }

        // Parse tensor headers from the in-memory header buffer.
        let mut cur = HeaderCursor::new(&header[16..]);
        let mut tensors = Vec::with_capacity(num_tensors as usize);
        for _ in 0..num_tensors {
            let name_len = cur.read_u16()? as usize;
            let name_bytes = cur.read_bytes(name_len)?;
            let name = std::str::from_utf8(name_bytes)
                .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "tensor name not utf8"))?
                .to_string();
            let dtype = Dtype::from_byte(cur.read_u8()?)?;
            let rank = cur.read_u8()? as usize;
            let mut dims = Vec::with_capacity(rank);
            for _ in 0..rank {
                dims.push(cur.read_u32()?);
            }
            let scale = f32::from_le_bytes(cur.read_array4()?);
            let data_offset = cur.read_u64()?;
            let data_len = cur.read_u64()?;
            let data_crc32 = cur.read_u32()?;

            // Sanity-check geometry vs file size.
            if data_offset
                .checked_add(data_len)
                .map(|end| end + 4 > total_len)
                .unwrap_or(true)
            {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("tensor `{}`: data range extends past file end", name),
                ));
            }
            let expected_bytes = (prod_dims(&dims) as u64) * dtype.elem_size() as u64;
            if expected_bytes != data_len {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "tensor `{}`: data_len {} does not match dims/dtype ({} expected)",
                        name, data_len, expected_bytes
                    ),
                ));
            }

            tensors.push(TensorMeta {
                name,
                dtype,
                dims,
                scale,
                data_offset,
                data_len,
                data_crc32,
            });
        }

        Ok(Self {
            inner,
            tensors,
            flags,
        })
    }

    pub fn tensors(&self) -> &[TensorMeta] {
        &self.tensors
    }

    /// Read an F32 tensor by index, validating its data CRC.
    pub fn read_tensor_f32(&mut self, idx: usize) -> io::Result<Vec<f32>> {
        let (dtype, name, num_elements) = {
            let meta = self
                .tensors
                .get(idx)
                .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "tensor index OOB"))?;
            (meta.dtype, meta.name.clone(), meta.num_elements())
        };
        if dtype != Dtype::F32 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("tensor `{}` is not F32", name),
            ));
        }
        let bytes = self.read_data_validated(idx)?;
        let mut out = Vec::with_capacity(num_elements);
        for chunk in bytes.chunks_exact(4) {
            out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
        }
        Ok(out)
    }

    /// Read an I8 tensor by index, returning `(data, scale)`.  Validates the
    /// data CRC.
    pub fn read_tensor_i8(&mut self, idx: usize) -> io::Result<(Vec<i8>, f32)> {
        let (dtype, name, scale) = {
            let meta = self
                .tensors
                .get(idx)
                .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "tensor index OOB"))?;
            (meta.dtype, meta.name.clone(), meta.scale)
        };
        if dtype != Dtype::I8 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("tensor `{}` is not I8", name),
            ));
        }
        let bytes = self.read_data_validated(idx)?;
        let data: Vec<i8> = bytes.into_iter().map(|b| b as i8).collect();
        Ok((data, scale))
    }

    fn read_data_validated(&mut self, idx: usize) -> io::Result<Vec<u8>> {
        let meta = self.tensors[idx].clone();
        self.inner.seek(SeekFrom::Start(meta.data_offset))?;
        let mut buf = vec![0u8; meta.data_len as usize];
        self.inner.read_exact(&mut buf)?;
        let actual = crc32(&buf);
        if actual != meta.data_crc32 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("tensor `{}`: data CRC32 mismatch", meta.name),
            ));
        }
        Ok(buf)
    }
}

// Lightweight slice-cursor for parsing the header region.
struct HeaderCursor<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> HeaderCursor<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }
    fn read_bytes(&mut self, n: usize) -> io::Result<&'a [u8]> {
        if self.pos + n > self.buf.len() {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "header truncated",
            ));
        }
        let s = &self.buf[self.pos..self.pos + n];
        self.pos += n;
        Ok(s)
    }
    fn read_u8(&mut self) -> io::Result<u8> {
        Ok(self.read_bytes(1)?[0])
    }
    fn read_u16(&mut self) -> io::Result<u16> {
        let b = self.read_bytes(2)?;
        Ok(u16::from_le_bytes([b[0], b[1]]))
    }
    fn read_u32(&mut self) -> io::Result<u32> {
        let b = self.read_bytes(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }
    fn read_u64(&mut self) -> io::Result<u64> {
        let b = self.read_bytes(8)?;
        Ok(u64::from_le_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
    }
    fn read_array4(&mut self) -> io::Result<[u8; 4]> {
        let b = self.read_bytes(4)?;
        Ok([b[0], b[1], b[2], b[3]])
    }
}

// ---------------------------------------------------------------------------
// CRC32 (IEEE polynomial 0xEDB88320), inline so we avoid pulling in crc32fast.
// ---------------------------------------------------------------------------

pub(crate) fn crc32(data: &[u8]) -> u32 {
    const POLY: u32 = 0xEDB8_8320;
    let mut crc: u32 = 0xFFFF_FFFF;
    for &byte in data {
        crc ^= byte as u32;
        for _ in 0..8 {
            let mask = (crc & 1).wrapping_neg();
            crc = (crc >> 1) ^ (POLY & mask);
        }
    }
    !crc
}
