//! Round-trip and corruption tests for the `.axn` wire format.

use std::io::Cursor;

use super::axn::*;

fn round_trip_bytes(build: impl FnOnce(&mut AxnWriter<Cursor<Vec<u8>>>)) -> Vec<u8> {
    let mut writer = AxnWriter::new(Cursor::new(Vec::new()));
    build(&mut writer);
    writer.finish().unwrap().into_inner()
}

#[test]
fn round_trip_single_f32_tensor() {
    let bytes = round_trip_bytes(|w| {
        w.add_tensor_f32("a", &[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    });
    let mut reader = AxnReader::open(Cursor::new(bytes)).unwrap();
    assert_eq!(reader.tensors().len(), 1);
    let meta = reader.tensors()[0].clone();
    assert_eq!(meta.name, "a");
    assert_eq!(meta.dtype, Dtype::F32);
    assert_eq!(meta.dims, vec![2, 3]);
    let data = reader.read_tensor_f32(0).unwrap();
    assert_eq!(data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

#[test]
fn round_trip_multiple_mixed_tensors() {
    let weights: Vec<f32> = (0..32).map(|i| i as f32 * 0.1).collect();
    let bias: Vec<f32> = vec![-1.0, 0.0, 1.0, 2.0];
    let qweights: Vec<i8> = (-32..32).collect();

    let bytes = round_trip_bytes(|w| {
        w.add_tensor_f32("layer0.weight", &[4, 8], &weights);
        w.add_tensor_f32("layer0.bias", &[4], &bias);
        w.add_tensor_i8("layer1.qweight", &[8, 8], 0.0123, &qweights);
    });

    let mut reader = AxnReader::open(Cursor::new(bytes)).unwrap();
    assert_eq!(reader.tensors().len(), 3);

    let w_back = reader.read_tensor_f32(0).unwrap();
    assert_eq!(w_back, weights);

    let b_back = reader.read_tensor_f32(1).unwrap();
    assert_eq!(b_back, bias);

    let (q_back, scale) = reader.read_tensor_i8(2).unwrap();
    assert_eq!(q_back, qweights);
    assert!((scale - 0.0123).abs() < 1e-9);
}

#[test]
fn corrupted_header_byte_detected() {
    let mut bytes = round_trip_bytes(|w| {
        w.add_tensor_f32("t", &[4], &[0.1, 0.2, 0.3, 0.4]);
    });
    // Flip a byte inside the tensor-header region (just after the 16-byte
    // prelude, somewhere in the tensor name length / name area).
    bytes[20] ^= 0xFF;
    let err = match AxnReader::open(Cursor::new(bytes)) {
        Ok(_) => panic!("expected open() to fail"),
        Err(e) => e,
    };
    let msg = err.to_string();
    assert!(
        msg.contains("CRC32") || msg.contains("not utf8") || msg.contains("truncated"),
        "expected header validation failure, got: {}",
        msg
    );
}

#[test]
fn corrupted_data_byte_detected() {
    let mut bytes = round_trip_bytes(|w| {
        w.add_tensor_f32("t", &[4], &[0.1, 0.2, 0.3, 0.4]);
    });
    // Header opens cleanly; flip a byte inside the data region.
    let header_end = {
        let reader = AxnReader::open(Cursor::new(bytes.clone())).unwrap();
        reader.tensors()[0].data_offset as usize
    };
    bytes[header_end] ^= 0xFF;
    let mut reader = AxnReader::open(Cursor::new(bytes)).unwrap();
    let err = reader.read_tensor_f32(0).unwrap_err();
    assert!(
        err.to_string().contains("CRC32"),
        "expected data CRC mismatch, got: {}",
        err
    );
}

#[test]
fn bad_magic_rejected() {
    let mut bytes = round_trip_bytes(|w| {
        w.add_tensor_f32("t", &[1], &[0.0]);
    });
    bytes[0] = b'X';
    let err = match AxnReader::open(Cursor::new(bytes)) {
        Ok(_) => panic!("expected open() to fail"),
        Err(e) => e,
    };
    assert!(err.to_string().contains("magic"));
}

#[test]
fn bad_version_rejected() {
    let mut bytes = round_trip_bytes(|w| {
        w.add_tensor_f32("t", &[1], &[0.0]);
    });
    bytes[4] = 0xFE;
    bytes[5] = 0xFF;
    let err = match AxnReader::open(Cursor::new(bytes)) {
        Ok(_) => panic!("expected open() to fail"),
        Err(e) => e,
    };
    assert!(err.to_string().contains("version"));
}

#[test]
fn dtype_mismatch_when_reading() {
    let bytes = round_trip_bytes(|w| {
        w.add_tensor_f32("t", &[1], &[1.0]);
    });
    let mut reader = AxnReader::open(Cursor::new(bytes)).unwrap();
    let err = reader.read_tensor_i8(0).unwrap_err();
    assert!(err.to_string().contains("not I8"));
}

#[test]
fn crc32_known_value() {
    // CRC32 of "123456789" is 0xCBF43926 (standard reference vector).
    assert_eq!(super::axn::crc32(b"123456789"), 0xCBF4_3926);
}
