//! The file formats and readers, at their edges: bundles, sealed bundles, published checkpoints, parts lists,
//! CSV tables and the preprocessing. Damaged or hostile input must give an error, never a panic or a huge
//! allocation; good input must round-trip exactly. Property tests (proptest) try many random cases of each.

mod common;
use common::*;
use proptest::prelude::*;
use std::io;
use std::sync::Arc;
use tabicl_anywhere::crypt::{read_key, seal_file, Sealed};
use tabicl_anywhere::csv::{describe, parse, Schema};
use tabicl_anywhere::model::{Load, Model};
use tabicl_anywhere::prep::*;
use tabicl_anywhere::store::*;

fn mem_bundle(tensors: &[(&str, &str, Vec<usize>)], fill: impl Fn(&BundleWriter)) -> Vec<u8> {
    let out = MemStore::new();
    let w = BundleWriter::create_on(Box::new(out.clone()), tensors).unwrap();
    fill(&w);
    drop(w);
    out.bytes()
}

// ---- bundles -------------------------------------------------------------------------------------------------

#[test]
fn bundle_round_trip_every_type() {
    let bytes = mem_bundle(
        &[("a", "float32", vec![2, 3]), ("b", "int32", vec![4]), ("c", "float16", vec![3]), ("d", "uint8", vec![5]), ("e", "float64", vec![1]), ("f", "int8", vec![0]), ("g", "int64", vec![2])],
        |w| {
            w.write_f32_at("a", 0, &[1.0, 2.0, 3.0]).unwrap();
            w.write_f32_at("a", 3, &[4.0, 5.0, 6.0]).unwrap();
            w.write_bytes("b", bytemuck::cast_slice(&[1i32, -2, 3, i32::MAX])).unwrap();
            w.write_bytes("c", &[0x00, 0x3c, 0x00, 0xc0, 0x00, 0x7c]).unwrap(); // 1, -2, +infinity
            w.write_bytes_at("d", 2, b"xyz").unwrap();
            w.write_bytes("e", &2.5f64.to_le_bytes()).unwrap();
            assert_eq!(w.entry("a").len(), 6);
        },
    );
    assert_eq!(bytes.len() % 64, 0, "tensors start and the file ends on 64-byte boundaries");
    let b = bytes_bundle(bytes);
    assert!(b.entries.iter().all(|e| e.offset % 64 == 0));
    assert_eq!(b.f32s("a").unwrap(), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let mut part = [0.0f32; 2];
    b.read_f32_at("a", 4, &mut part).unwrap();
    assert_eq!(part, [5.0, 6.0]);
    assert_eq!(b.i32s("b").unwrap(), [1, -2, 3, i32::MAX]);
    assert_eq!(b.f32s("c").unwrap(), [1.0, -2.0, f32::INFINITY], "16-bit floats widen on reading");
    let mut c1 = [0.0f32; 1];
    b.read_f32_at("c", 1, &mut c1).unwrap();
    assert_eq!(c1, [-2.0]);
    assert_eq!(b.bytes("d").unwrap(), b"\0\0xyz");
    let mut d2 = [0u8; 2];
    b.read_bytes_at("d", 3, &mut d2).unwrap();
    assert_eq!(&d2, b"yz");
    assert_eq!(b.bytes("f").unwrap(), b"");
    assert!(b.has("e") && !b.has("zz"));
    // wrong use is an error
    assert!(b.entry("zz").is_err());
    assert!(b.f32s("b").is_err(), "int32 read as floats");
    assert!(b.i32s("a").is_err(), "floats read as int32");
    assert!(b.read_f32_at("a", 5, &mut [0.0; 2]).is_err(), "past the end");
    assert!(b.read_f32_at("c", 2, &mut [0.0; 2]).is_err(), "past the end of a float16 tensor");
    assert!(b.read_bytes_at("d", 4, &mut [0u8; 2]).is_err());
}

#[test]
fn bundle_on_disk() {
    let path = scratch("bundle");
    let w = BundleWriter::create(&path, &[("x", "float32", vec![3])]).unwrap();
    w.write_f32_at("x", 0, &[7.0, 8.0, 9.0]).unwrap();
    w.file.sync().unwrap();
    drop(w);
    assert_eq!(Bundle::open(&path).unwrap().f32s("x").unwrap(), [7.0, 8.0, 9.0]);
    let missing = Bundle::open(path.with_extension("missing")).err().unwrap();
    assert_eq!(missing.kind(), io::ErrorKind::NotFound);
    assert!(missing.to_string().contains("missing"), "the error names the file: {missing}");
    let _ = std::fs::remove_file(path);
}

#[test]
fn writer_refuses_bad_tensors() {
    let err = |t: &[(&str, &str, Vec<usize>)]| BundleWriter::create_on(Box::new(MemStore::new()), t).err().unwrap().to_string();
    assert!(err(&[("a", "complex64", vec![1])]).contains("unknown tensor type"));
    assert!(err(&[("a b", "uint8", vec![1])]).contains("space"));
    assert!(err(&[("", "uint8", vec![1])]).contains("empty"));
}

#[test]
fn damaged_bundles_are_errors() {
    let good = mem_bundle(&[("x", "float32", vec![4])], |w| w.write_f32_at("x", 0, &[1.0; 4]).unwrap());
    let open = |b: Vec<u8>| Bundle::from_source(Box::new(Arc::new(b)));
    assert!(open(good.clone()).is_ok());
    assert!(open(b"short".to_vec()).is_err(), "shorter than the header");
    let mut magic = good.clone();
    magic[0] = b'X';
    assert!(open(magic).unwrap_err().to_string().contains("not a tabicl-anywhere bundle"));
    let mut huge = good.clone();
    huge[8..16].copy_from_slice(&u64::MAX.to_le_bytes());
    assert!(open(huge).unwrap_err().to_string().contains("damaged"), "a huge index length is refused before allocating");
    let mut long = good.clone();
    long[8..16].copy_from_slice(&(good.len() as u64).to_le_bytes());
    assert!(open(long).is_err(), "index running past the end");
    let with_index = |text: &str| {
        let mut b = b"TIRSBNDL".to_vec();
        b.extend((text.len() as u64).to_le_bytes());
        b.extend(text.as_bytes());
        b.resize(4096, 0);
        open(b)
    };
    assert!(with_index("x float32 64 16 4\n").is_ok());
    assert!(with_index("x float32 64\n").unwrap_err().to_string().contains("short index line"));
    assert!(with_index("x float32 sixty 16 4\n").is_err());
    assert!(with_index("x float32 64 16 5\n").unwrap_err().to_string().contains("disagree"), "shape says 20 bytes, length 16");
    assert!(with_index("x float32 64 16 -4\n").is_err());
    assert!(with_index("x weird 64 16 4\n").unwrap_err().to_string().contains("unknown tensor type"));
    assert!(with_index("x uint8 18446744073709551615 1 1\n").is_err(), "offset + length overflows");
    assert!(with_index("x uint8 64 18446744073709551615 18446744073709551615\n").is_err());
    let mut not_text = b"TIRSBNDL".to_vec();
    not_text.extend(2u64.to_le_bytes());
    not_text.extend([0xff, 0xfe]);
    assert!(open(not_text).is_err());
    // an entry pointing past the end of the data reads as an error, not garbage
    let past = with_index("x float32 100000 16 4\n").unwrap();
    assert!(past.f32s("x").is_err());
}

#[test]
fn byte_sources_check_their_bounds() {
    let src = Arc::new(vec![1u8, 2, 3]);
    let mut b = [0u8; 2];
    src.read_exact_at(&mut b, 1).unwrap();
    assert_eq!(b, [2, 3]);
    assert_eq!(src.read_exact_at(&mut b, 2).unwrap_err().kind(), io::ErrorKind::UnexpectedEof);
    assert!(src.read_exact_at(&mut b, u64::MAX).is_err(), "an absurd offset is an error, not an overflow");
    let m = MemStore::new();
    m.write_all_at(b"abc", 2).unwrap();
    m.read_exact_at(&mut b, 3).unwrap();
    assert_eq!(&b, b"bc");
    assert!(m.read_exact_at(&mut b, 4).is_err());
    assert!(m.read_exact_at(&mut b, u64::MAX).is_err());
    assert!(m.write_all_at(b"x", u64::MAX).is_err());
    m.set_len(1).unwrap();
    m.sync().unwrap();
    assert_eq!(m.bytes(), [0]);
    assert!(m.bytes().is_empty(), "bytes() takes them");
    // files, through the platform's positioned reads and writes
    let path = scratch("rw");
    let f = std::fs::File::options().read(true).write(true).create(true).truncate(true).open(&path).unwrap();
    file_write_at(&f, b"hello", 3).unwrap();
    let mut got = [0u8; 4];
    file_read_at(&f, &mut got, 4).unwrap();
    assert_eq!(&got, b"ello");
    assert!(file_read_at(&f, &mut got, 6).is_err(), "reading past the end of a file");
    Storage::set_len(&f, 2).unwrap();
    assert!(f.read_exact_at(&mut got, 0).is_err());
    drop(f);
    let _ = std::fs::remove_file(path);
}

#[test]
fn parts_lists() {
    let data: Vec<u8> = (0..1000u32).map(|i| (i % 251) as u8).collect();
    let parts = split_parts(&data, 300);
    assert_eq!(parts.len(), 4);
    assert_eq!(split_parts(&data, 0).len(), data.len(), "a zero limit means one byte per part, not a hang");
    let named: Vec<(String, usize)> = parts.iter().enumerate().map(|(i, p)| (format!("w.{i:02}.bin"), p.len())).collect();
    let manifest = parts_manifest(data.len(), &named);
    assert_eq!(read_parts_manifest(&manifest).unwrap(), (data.len(), named.clone()));
    let read = |name: &str| -> io::Result<Vec<u8>> {
        let i: usize = name[2..4].parse().unwrap();
        Ok(parts[i].to_vec())
    };
    assert_eq!(join_parts(&manifest, read).unwrap(), data);
    // a short part, a missing part, a wrong total, an unsafe name, not a parts list at all
    assert!(join_parts(&manifest, |n| read(n).map(|mut v| {
        v.pop();
        v
    }))
    .unwrap_err()
    .to_string()
    .contains("parts list says"));
    assert!(join_parts(&manifest, |_| Err(io::ErrorKind::NotFound.into())).is_err());
    assert!(read_parts_manifest(&manifest.replacen("\"bytes\": 1000", "\"bytes\": 999", 1)).is_err());
    assert!(read_parts_manifest(&parts_manifest(3, &[("../etc/passwd".into(), 3)])).is_err());
    assert!(read_parts_manifest(&parts_manifest(3, &[("a/b".into(), 3)])).is_err());
    assert!(read_parts_manifest("{\"bytes\": 3}").is_err());
    assert!(read_parts_manifest(&manifest.replace("\"parts\"", "\"pieces\"")).is_err());
    assert!(read_parts_manifest(&manifest.replacen("\"bytes\": 300", "\"bytes\": x", 1)).is_err());
    // the committed web copy of the weights is a valid list whose parts are all there
    // (absent from the published crate, which leaves docs/ out)
    for kind in ["classifier", "regressor"] {
        let Ok(text) = std::fs::read_to_string(root().join(format!("docs/weights/{kind}-v2-half.json"))) else { continue };
        let (total, list) = read_parts_manifest(&text).unwrap();
        for (name, len) in &list {
            assert_eq!(std::fs::metadata(root().join("docs/weights").join(name)).unwrap().len() as usize, *len, "{name}");
        }
        assert_eq!(list.iter().map(|p| p.1).sum::<usize>(), total);
    }
}

#[test]
fn half_float_rounding_edges() {
    assert_eq!(f32_to_f16(f32::NAN) & 0x7e00, 0x7e00);
    assert_eq!(f32_to_f16(f32::NEG_INFINITY), 0xfbff, "infinity becomes the largest finite value, keeping its sign");
    assert_eq!(f32_to_f16(-0.0), 0x8000);
    assert_eq!(f32_to_f16(65520.0), 0x7bff, "rounds up past the largest finite value: clamped");
    assert_eq!(f32_to_f16(5.960_464_5e-8), 0x0001, "smallest subnormal");
    assert_eq!(f32_to_f16(2.98e-8), 0x0000, "below half the smallest subnormal");
    assert_eq!(f16_to_f32(0x8000).to_bits(), (-0.0f32).to_bits());
    assert!(f16_to_f32(0x7e00).is_nan());
}

proptest! {
    #[test]
    fn narrowing_is_nearest_and_monotone(a in -70000.0f32..70000.0, b in -70000.0f32..70000.0) {
        let (lo, hi) = if a <= b { (a, b) } else { (b, a) };
        let (hl, hh) = (f16_to_f32(f32_to_f16(lo)), f16_to_f32(f32_to_f16(hi)));
        prop_assert!(hl <= hh, "narrowing must keep order: {lo} -> {hl}, {hi} -> {hh}");
        if lo.abs() <= 65504.0 {
            let step = if lo.abs() < 6.1e-5 { 2.0f32.powi(-24) } else { lo.abs() * 2.0f32.powi(-10) };
            prop_assert!((hl - lo).abs() <= 0.5 * step * 1.0001, "{lo} -> {hl}");
        }
    }

    #[test]
    fn split_then_join_is_identity(len in 0usize..5000, max in 1usize..2000) {
        let data: Vec<u8> = (0..len).map(|i| (i * 7 % 256) as u8).collect();
        let parts = split_parts(&data, max);
        prop_assert!(parts.iter().all(|p| p.len() <= max && !p.is_empty()));
        let named: Vec<(String, usize)> = parts.iter().enumerate().map(|(i, p)| (format!("p{i}"), p.len())).collect();
        let joined = join_parts(&parts_manifest(len, &named), |n| Ok(parts[n[1..].parse::<usize>().unwrap()].to_vec())).unwrap();
        prop_assert_eq!(joined, data);
    }

    #[test]
    fn damaged_bundles_never_panic(flips in prop::collection::vec((0usize..2048, any::<u8>()), 1..8)) {
        let mut bytes = mem_bundle(&[("x", "float32", vec![8]), ("y", "int32", vec![3, 2]), ("z", "float16", vec![5])], |w| w.write_f32_at("x", 0, &[0.5; 8]).unwrap());
        for (at, v) in flips {
            let i = at % bytes.len();
            bytes[i] = v;
        }
        if let Ok(b) = Bundle::from_source(Box::new(Arc::new(bytes))) {
            for e in b.entries.clone() {
                let _ = (b.f32s(&e.name), b.i32s(&e.name), b.bytes(&e.name));
            }
        }
    }
}

// ---- sealed bundles ------------------------------------------------------------------------------------------

fn sealed_of(plain: &[u8]) -> (Vec<u8>, [u8; 32]) {
    let (p, s) = (scratch("plain"), scratch("sealed"));
    std::fs::write(&p, plain).unwrap();
    let key = tabicl_anywhere::crypt::new_key().unwrap();
    seal_file(&p, &s, &key).unwrap();
    let out = std::fs::read(&s).unwrap();
    let _ = (std::fs::remove_file(p), std::fs::remove_file(s));
    (out, key)
}

#[test]
fn sealing_edges() {
    const BLOCK: usize = 64 << 10;
    // empty, exactly one block, exactly two blocks
    for len in [0usize, 1, BLOCK, 2 * BLOCK] {
        let plain: Vec<u8> = (0..len).map(|i| (i % 253) as u8).collect();
        let (sealed, key) = sealed_of(&plain);
        assert_eq!(sealed.len(), 32 + len + len.div_ceil(BLOCK) * 16);
        let s = Sealed::open(Arc::new(sealed.clone()), &key).unwrap();
        let mut all = vec![0u8; len];
        s.read_exact_at(&mut all, 0).unwrap();
        assert_eq!(all, plain);
        s.read_exact_at(&mut [], 12345).unwrap(); // an empty read always succeeds
        assert_eq!(s.read_exact_at(&mut [0u8; 1], len as u64).unwrap_err().kind(), io::ErrorKind::UnexpectedEof);
        // sealing twice gives different bytes (a fresh random file id) that both open
        let (again, key2) = sealed_of(&plain);
        assert_ne!(again[8..24], sealed[8..24]);
        assert!(Sealed::open(Arc::new(again), &key2).is_ok());
    }
    assert!(Sealed::open(Arc::new(b"TIRSSEAL".to_vec()), &[0; 32]).is_err(), "header cut short");
    assert!(Sealed::open(Arc::new(vec![0u8; 64]), &[0; 32]).unwrap_err().to_string().contains("not a sealed"));
    // a block swapped for another block of the same file is refused (each block's nonce has its index)
    let plain = vec![9u8; 3 * BLOCK];
    let (mut sealed, key) = sealed_of(&plain);
    let (a, b) = (32, 32 + BLOCK + 16);
    let first: Vec<u8> = sealed[a..a + BLOCK + 16].to_vec();
    sealed.copy_within(b..b + BLOCK + 16, a);
    sealed[b..b + BLOCK + 16].copy_from_slice(&first);
    let s = Sealed::open(Arc::new(sealed.clone()), &key).unwrap();
    assert!(s.read_exact_at(&mut [0u8; 8], 0).is_err());
    // a cut-off file is refused when the missing part is read
    sealed.truncate(sealed.len() - 100);
    let s = Sealed::open(Arc::new(sealed), &key).unwrap();
    assert!(s.read_exact_at(&mut [0u8; 8], 3 * BLOCK as u64 - 8).is_err());
    let path = scratch("missing");
    assert!(seal_file(&path, scratch("out"), &key).is_err());
}

#[test]
fn key_files() {
    let path = scratch("key");
    let key: [u8; 32] = std::array::from_fn(|i| (i * 9) as u8);
    std::fs::write(&path, key).unwrap();
    assert_eq!(read_key(&path).unwrap(), key, "32 raw bytes");
    let hex: String = key.iter().map(|b| format!("{b:02X}")).collect();
    std::fs::write(&path, format!("  {hex}\n")).unwrap();
    assert_eq!(read_key(&path).unwrap(), key, "64 hex digits, any case, surrounding space ignored");
    for bad in ["abc", &hex[..62], &format!("{}zz", &hex[..62]), &format!("{hex}00")] {
        std::fs::write(&path, bad).unwrap();
        assert!(read_key(&path).is_err(), "{bad}");
    }
    let _ = std::fs::remove_file(&path);
    assert!(read_key(&path).is_err(), "missing file");
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(24))]
    #[test]
    fn sealed_ranges_read_back(len in 1usize..200_000, reads in prop::collection::vec((0usize..200_000, 0usize..70_000), 1..6)) {
        let plain: Vec<u8> = (0..len).map(|i| (i * 13 % 251) as u8).collect();
        let (sealed, key) = sealed_of(&plain);
        let s = Sealed::open(Arc::new(sealed), &key).unwrap();
        for (off, n) in reads {
            let off = off % len;
            let n = n.min(len - off);
            let mut got = vec![0u8; n];
            s.read_exact_at(&mut got, off as u64).unwrap();
            prop_assert_eq!(&got[..], &plain[off..off + n]);
        }
    }
}

// ---- published checkpoints -----------------------------------------------------------------------------------

fn tiny_ckpt() -> Vec<u8> {
    std::fs::read(root().join("tests/data/tiny-clf.ckpt")).unwrap()
}

#[test]
fn checkpoint_conversion() {
    let (ck, out) = (scratch("ck"), scratch("out"));
    std::fs::write(&ck, tiny_ckpt()).unwrap();
    let n = tabicl_anywhere::ckpt::convert(&ck, &out).unwrap();
    let b = Bundle::open(&out).unwrap();
    assert_eq!(b.entries.len(), n + 1, "every tensor plus the configuration");
    assert_eq!(std::fs::read(&out).unwrap(), tabicl_anywhere::ckpt::bundle_bytes(tiny_ckpt(), false).unwrap(), "file and memory paths agree");
    let cfg = String::from_utf8(b.bytes("config.json").unwrap()).unwrap();
    assert!(cfg.contains("\"embed_dim\": 8") && cfg.contains("\"col_ssmax\": \"qassmax-mlp-elementwise\"") && cfg.contains("\"col_affine\": false"), "{cfg}");
    // a bundle given to bundle_bytes passes through untouched
    let bytes = std::fs::read(&out).unwrap();
    assert_eq!(tabicl_anywhere::ckpt::bundle_bytes(bytes.clone(), false).unwrap(), bytes);
    let _ = (std::fs::remove_file(ck), std::fs::remove_file(out));
}

#[test]
fn halving_narrows_only_large_float_tensors() {
    let big: Vec<f32> = (0..5000).map(|i| (i as f32 * 0.37).sin() * 3.0).collect();
    let src = bytes_bundle(mem_bundle(&[("big", "float32", vec![50, 100]), ("small", "float32", vec![4]), ("ints", "int32", vec![5000])], |w| {
        w.write_f32_at("big", 0, &big).unwrap();
        w.write_f32_at("small", 0, &[0.1, 0.2, 0.3, 1e-9]).unwrap();
    }));
    let out = MemStore::new();
    assert_eq!(tabicl_anywhere::ckpt::halve_on(&src, Box::new(out.clone())).unwrap(), (1, 2));
    let half = bytes_bundle(out.bytes());
    assert_eq!(half.entry("big").unwrap().dtype, "float16");
    assert_eq!(half.entry("big").unwrap().dims, [50, 100]);
    assert_eq!(half.f32s("small").unwrap(), [0.1, 0.2, 0.3, 1e-9], "small tensors exact");
    assert_eq!(half.entry("ints").unwrap().dtype, "int32");
    let got = half.f32s("big").unwrap();
    for (a, b) in got.iter().zip(&big) {
        assert!((a - b).abs() <= b.abs() * 2.0f32.powi(-11) + 1e-7, "{a} vs {b}");
    }
}

#[test]
fn damaged_checkpoints_are_errors() {
    let good = tiny_ckpt();
    let err = |b: Vec<u8>| tabicl_anywhere::ckpt::bundle_bytes(b, false).unwrap_err().to_string();
    assert!(err(b"not a zip at all".to_vec()).contains("not a zip"));
    assert!(err(Vec::new()).contains("not a zip"));
    assert!(err(good[..good.len() / 2].to_vec()).contains("zip"), "cut in half: no end record");
    // the end record cut short right after its signature
    let at = good.windows(4).rposition(|w| w == b"PK\x05\x06").unwrap();
    assert!(err(good[..at + 6].to_vec()).contains("damaged"));
    // claim the members are compressed
    let mut comp = good.clone();
    let dir = good.windows(4).position(|w| w == b"PK\x01\x02").unwrap();
    comp[dir + 10] = 8;
    assert!(err(comp).contains("compressed"));
    // a central directory pointing past the end
    let mut far = good.clone();
    far[at + 16..at + 20].copy_from_slice(&u32::MAX.to_le_bytes());
    if let Some(z) = good.windows(4).rposition(|w| w == b"PK\x06\x06") {
        far[z + 48..z + 56].copy_from_slice(&u64::MAX.to_le_bytes()); // PyTorch writes the zip64 form too
    }
    assert!(tabicl_anywhere::ckpt::bundle_bytes(far, false).is_err());
    // a zip without data.pkl
    let mut renamed = good.clone();
    for i in 0..renamed.len() - 8 {
        if &renamed[i..i + 8] == b"data.pkl" {
            renamed[i..i + 8].copy_from_slice(b"dada.pkl");
        }
    }
    assert!(err(renamed).contains("data.pkl"));
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]
    /// Random damage anywhere in a real PyTorch checkpoint: an error or a bundle, never a panic. (Damage inside
    /// tensor data cannot be detected, since the format has no checksums; that converts to wrong numbers, which
    /// is why weights are downloaded from a known source.)
    #[test]
    fn damaged_checkpoints_never_panic(flips in prop::collection::vec((any::<prop::sample::Index>(), any::<u8>()), 1..6), cut in any::<prop::sample::Index>()) {
        let mut b = tiny_ckpt();
        for (at, v) in flips {
            let i = at.index(b.len());
            b[i] = v;
        }
        let _ = tabicl_anywhere::ckpt::bundle_bytes(b.clone(), false);
        b.truncate(cut.index(b.len()));
        let _ = tabicl_anywhere::ckpt::bundle_bytes(b, false);
    }
}

// ---- weights -------------------------------------------------------------------------------------------------

/// The tiny weight bundle rebuilt with `change` applied to its tensor list and configuration text.
fn altered(change: impl Fn(&mut Vec<(String, String, Vec<usize>, Vec<u8>)>)) -> Bundle {
    let src = tiny_weights("clf");
    let mut t: Vec<(String, String, Vec<usize>, Vec<u8>)> = src.entries.iter().map(|e| (e.name.clone(), e.dtype.clone(), e.dims.clone(), src.bytes(&e.name).unwrap())).collect();
    change(&mut t);
    let ents: Vec<(&str, &str, Vec<usize>)> = t.iter().map(|(n, d, s, _)| (n.as_str(), d.as_str(), s.clone())).collect();
    bytes_bundle(mem_bundle(&ents, |w| t.iter().for_each(|(n, _, _, b)| w.write_bytes(n, b).unwrap())))
}

fn with_config(t: &mut [(String, String, Vec<usize>, Vec<u8>)], f: impl Fn(String) -> String) {
    let c = t.iter_mut().find(|e| e.0 == "config.json").unwrap();
    c.3 = f(String::from_utf8(c.3.clone()).unwrap()).into_bytes();
    c.2 = vec![c.3.len()];
}

#[test]
fn model_loading() {
    let wb = tiny_weights("clf");
    let m = Model::load(&wb, Load::Fit).unwrap();
    assert_eq!((m.e, m.d, m.cls, m.max_classes, m.icl_layers, m.outputs()), (8, 16, 2, 10, 2, 10));
    assert!(m.col.iter().all(|c| c.a1.is_some() && !c.ind.is_empty()));
    let p = Model::load(&wb, Load::Predict).unwrap();
    assert!(p.col.iter().all(|c| c.a1.is_none()), "prediction needs no inducing-point blocks");
    assert_eq!(p.icl[0].in_w.len(), m.d * m.d, "only the query projection is kept for prediction");
    let r = Model::load(&tiny_weights("reg"), Load::Fit).unwrap();
    assert_eq!((r.max_classes, r.outputs()), (0, 99));
    // layer norms without biases load as zero biases
    let nb = altered(|t| t.retain(|e| !e.0.ends_with("norm1.bias") && e.0 != "icl_predictor.ln.bias" && e.0 != "row_interactor.out_ln.bias"));
    let m2 = Model::load(&nb, Load::Fit).unwrap();
    assert!(m2.lnb.iter().all(|v| *v == 0.0) && m2.row[0].n1b.iter().all(|v| *v == 0.0));
}

#[test]
fn unsupported_or_incomplete_weights_are_refused() {
    let err = |b: Bundle| Model::load(&b, Load::Fit).err().unwrap().to_string();
    assert!(err(altered(|t| with_config(t, |c| c.replace("\"activation\": \"gelu\"", "\"activation\": \"relu\"")))).contains("activation is relu"));
    assert!(err(altered(|t| with_config(t, |c| c.replace("\"norm_first\": true, ", "")))).contains("norm_first is (missing)"));
    assert!(err(altered(|t| with_config(t, |c| c.replace("\"icl_nhead\": 2", "\"icl_nhead\": \"two\"")))).contains("icl_nhead"));
    assert!(err(altered(|t| t.retain(|e| e.0 != "icl_predictor.decoder.2.bias"))).contains("decoder.2.bias"));
    assert!(err(altered(|t| t.retain(|e| e.0 != "config.json"))).contains("config.json"));
    assert!(err(altered(|t| with_config(t, |_| "{}".into()))).contains("(missing)"));
}

// ---- CSV -----------------------------------------------------------------------------------------------------

#[test]
fn csv_parsing() {
    assert_eq!(parse("a,b\r\n1,2\r\n"), [["a", "b"], ["1", "2"]], "Windows line ends");
    assert_eq!(parse("a,b\n1,2"), [["a", "b"], ["1", "2"]], "no final line end");
    assert_eq!(parse("a\n\n1\n\n"), [["a"], ["1"]], "blank lines skipped");
    assert_eq!(parse("\"x,\"\"y\"\"\nz\",2\n"), [["x,\"y\"\nz", "2"]], "quotes, doubled quotes, a line break inside");
    assert_eq!(parse("a,,c\n"), [["a", "", "c"]]);
    assert_eq!(parse("a,\n"), [["a", ""]]);
    assert!(parse("").is_empty());
}

#[test]
fn csv_schema_edges() {
    let rows = parse("y,n,t,gone\n10,1,b,\n9,NA,a,?\n10,3,b,null\n2, 4 ,c,nan\n");
    let (s, x, y) = Schema::learn(&rows, "y").unwrap();
    assert_eq!(s.labels, ["2", "9", "10"], "numeric labels sort as numbers");
    assert_eq!(y, [2, 1, 2, 0]);
    assert_eq!(s.columns[0], ("t".to_string(), Some(vec!["a".to_string(), "b".into(), "c".into()])), "categorical first");
    assert_eq!(s.columns[1].0, "n");
    assert_eq!(s.columns[2], ("gone".to_string(), None), "an all-missing column is numeric (all NaN)");
    assert_eq!(&x[..2], [1.0, 1.0], "row 1: t = b (index 1), n = 1");
    assert!(x[2].is_nan() && x[4].is_nan() && x[9] == 2.0 && x[10] == 4.0, "{x:?}");
    let (s2, _, _) = Schema::learn(&parse("y,v\nb,1\na,2\n"), "y").unwrap();
    assert_eq!(s2.labels, ["a", "b"], "text labels sort as text");
    assert_eq!(Schema::from_text(&s.to_text()), s);
    // errors
    let e = |r: io::Result<(Schema, Vec<f64>, Vec<usize>)>| r.unwrap_err().to_string();
    assert!(e(Schema::learn(&[], "y")).contains("empty"));
    assert!(e(Schema::learn(&parse("a,b\n1,2\n"), "y")).contains("no column named y"));
    assert!(e(Schema::learn(&parse("a,y\n1,2\n3\n"), "y")).contains("row 3 has 1 fields"));
    assert!(e(Schema::learn(&parse("a,y\n1,\n"), "y")).contains("missing values"));
    // features of new tables
    assert!(s.features(&parse("t,n\nb,1\n")).unwrap_err().to_string().contains("no column named gone"));
    assert!(s.features(&parse("t,n,gone\nb,one,1\n")).unwrap_err().to_string().contains("'one' is not a number"));
    let f = s.features(&parse("gone,n,t,extra\n,,zzz,1\n \n")).unwrap();
    assert_eq!(f[0], -1.0, "an unseen category");
    assert!(f[1].is_nan() && f[2].is_nan());
    assert!(f[3..].iter().all(|v| v.is_nan() || *v == -1.0), "a short row: its missing cells are missing");
    assert!(s.features(&[]).is_err());
}

#[test]
fn csv_numeric_target() {
    let (s, x, y) = Schema::learn_numeric(&parse("a,y\n1,2.5\n2,-1e3\n"), "y").unwrap();
    assert!(s.labels.is_empty());
    assert_eq!((x, y), (vec![1.0, 2.0], vec![2.5, -1000.0]));
    for bad in ["a,y\n1,x\n", "a,y\n1,\n", "a,y\n1,inf\n", "a,y\n1\n"] {
        assert!(Schema::learn_numeric(&parse(bad), "y").unwrap_err().to_string().contains("not a number"), "{bad}");
    }
    assert!(Schema::learn_numeric(&parse("a,b\n1,2\n"), "y").is_err());
    assert!(Schema::learn_numeric(&[], "y").is_err());
}

#[test]
fn csv_describe() {
    let d = describe(&parse("a,b\n1,x\n2,y\n,x\n3,z\n4,w\n5,v\n")).unwrap();
    assert_eq!(d[0], tabicl_anywhere::csv::ColumnInfo { name: "a".into(), numeric: true, distinct: 5, missing: 1, examples: vec!["1".into(), "2".into(), "3".into(), "4".into()] });
    assert!(!d[1].numeric && d[1].distinct == 5 && d[1].examples.len() == 4);
    assert!(describe(&[]).is_err());
    assert!(describe(&parse("a,b\n1\n")).is_err());
    assert!(describe(&parse("a, a\n1,2\n")).unwrap_err().to_string().contains("both named 'a'"));
}

proptest! {
    /// Any table of fields, written out as CSV with quoting where needed, parses back to the same fields.
    #[test]
    fn csv_round_trip(rows in prop::collection::vec(prop::collection::vec("[a-z0-9 ,\"\n]{0,6}", 3), 1..8)) {
        let quote = |f: &String| if f.contains([',', '"', '\n']) || f.is_empty() { format!("\"{}\"", f.replace('"', "\"\"")) } else { f.clone() };
        let text: String = rows.iter().map(|r| r.iter().map(quote).collect::<Vec<_>>().join(",") + "\n").collect();
        prop_assert_eq!(parse(&text), rows);
    }
}

// ---- preprocessing -------------------------------------------------------------------------------------------

#[test]
fn preprocessing_edges() {
    // columns: varying, constant, all missing, varying with gaps
    let n = 6;
    let x: Vec<f64> = (0..n).flat_map(|r| [r as f64, 5.0, f64::NAN, if r % 2 == 0 { f64::NAN } else { r as f64 * 10.0 }]).collect();
    let (p, t) = Prep::fit(&x, n, 4, &[false, true], 4.0);
    assert_eq!(p.keep, [0, 3], "constant and all-missing columns are dropped");
    assert_eq!(p.impute[1], 5.0);
    assert_eq!(p.impute[2], 0.0, "all missing: imputed with 0");
    assert_eq!(p.impute[3], 30.0, "the mean of the cells that are there");
    for (v, tv) in t.iter().enumerate() {
        let col: Vec<f64> = (0..n).map(|r| tv[r * 2]).collect();
        let m = col.iter().sum::<f64>() / n as f64;
        assert!(m.abs() < 1e-9, "variant {v}: centred");
        // the stored numbers reproduce the training table exactly, row by row
        let mut row = [0.0; 2];
        for r in 0..n {
            p.apply(v, &x[r * 4..(r + 1) * 4], &mut row);
            assert!((row[0] - tv[r * 2]).abs() < 1e-12 && (row[1] - tv[r * 2 + 1]).abs() < 1e-12);
        }
    }
    // nothing varies: the reference keeps the first column
    let (p, t) = Prep::fit(&[1.0, 2.0, 3.0, 4.0], 2, 2, &[false], 4.0);
    assert_eq!(p.keep, [0, 1]);
    let (p, t2) = Prep::fit(&[1.0, 3.0, 1.0, 3.0], 2, 2, &[false], 4.0);
    assert_eq!(p.keep, [0], "{t:?}");
    assert!(t2[0].iter().all(|v| v.is_finite()));
    // a single row: everything kept, nothing infinite
    let (p, t) = Prep::fit(&[1.0, f64::NAN, 3.0], 1, 3, &[false, true], 4.0);
    assert_eq!(p.keep, [0, 1, 2]);
    assert!(t.iter().flatten().all(|v| v.is_finite()), "{t:?}");
    // extreme values are pulled in, not passed through
    let mut big: Vec<f64> = (0..50).map(|i| (i % 7) as f64).collect();
    big[10] = 1e12;
    let (p, t) = Prep::fit(&big, 50, 1, &[false], 4.0);
    assert!(t[0][10] < 30.0, "an outlier is squashed: {}", t[0][10]);
    let mut out = [0.0];
    p.apply(0, &[1e300], &mut out);
    assert!(out[0].is_finite() && out[0] < 200.0);
    p.apply(0, &[f64::NAN], &mut out);
    assert!(out[0].is_finite(), "a missing cell becomes the training mean");
}

#[test]
fn power_transform() {
    for l in [-1.5, 0.0, 0.5, 1.0, 2.0, 3.0] {
        assert!(yeo_johnson(0.0, l).abs() < 1e-15, "zero maps to zero");
        assert!((yeo_johnson(1e-9, l) - yeo_johnson(-1e-9, l) - 2e-9).abs() < 1e-15, "slope 1 at zero, both sides");
        let mut last = f64::MIN;
        for i in -50..50 {
            let v = yeo_johnson(i as f64 * 0.3, l);
            assert!(v > last, "monotone for lambda {l}");
            last = v;
        }
    }
    assert!((yeo_johnson(3.0, 1.0) - 3.0).abs() < 1e-12, "lambda 1 is the identity");
    assert_eq!(yeo_johnson_lambda(&[0.0; 5]), 1.0);
    let skewed: Vec<f64> = (1..200).map(|i| (i as f64 / 40.0).exp()).collect();
    assert!(yeo_johnson_lambda(&skewed) < 0.5, "a right tail asks for a lambda below 1");
    let neg: Vec<f64> = skewed.iter().map(|v| -v).collect();
    assert!(yeo_johnson_lambda(&neg) > 1.5, "a left tail, above 1");
    let mixed: Vec<f64> = (0..100).map(|i| i as f64 - 30.0).collect();
    assert!(yeo_johnson_lambda(&mixed).is_finite());
}

#[test]
fn python_random_and_ensembles() {
    let mut r = PyRandom::new(7);
    for n in [1usize, 2, 3, 1000, u32::MAX as usize] {
        assert!(r.below(n) < n);
    }
    let mut empty: [u8; 0] = [];
    r.shuffle(&mut empty);
    for n in [1usize, 2, 5, 9] {
        let sq = latin_orders(n, 42);
        assert_eq!(sq.len(), n);
        for i in 0..n {
            let mut row = sq[i].clone();
            row.sort();
            assert_eq!(row, (0..n).collect::<Vec<_>>(), "each order is a permutation");
            let mut col: Vec<usize> = sq.iter().map(|o| o[i]).collect();
            col.sort();
            assert_eq!(col, (0..n).collect::<Vec<_>>(), "and together they form a Latin square");
        }
    }
    let e = ensemble(5, 3, 2, 8, 42);
    assert_eq!(e.len(), 8);
    assert!(e.iter().all(|m| m.norm < 2 && m.features.len() == 5 && m.classes.len() == 3));
    let distinct: std::collections::HashSet<String> = e.iter().map(|m| format!("{m:?}")).collect();
    assert_eq!(distinct.len(), 8, "members differ");
    let one = ensemble(5, 3, 2, 1, 42);
    assert_eq!(one, vec![Member { norm: 0, features: (0..5).collect(), classes: (0..3).collect() }], "one member: the identity");
    assert_eq!(ensemble(1, 1, 1, 8, 42).len(), 1, "a table with one arrangement has one member however many are asked");
    let c = ColNorm { mean: 1.0, scale: 2.0, lambda: 0.5, pmean: 0.1, pscale: 0.9, lower: -3.0, upper: 3.0 };
    assert_eq!(ColNorm::from_array(&c.to_array()).to_array(), c.to_array());
}

proptest! {
    /// Fitting transforms the training table; applying the stored numbers to the same rows must give the same table.
    #[test]
    fn preprocessing_apply_matches_fit(n in 1usize..40, h in 1usize..5, seed in any::<u64>(), missing in 0.0f64..0.5) {
        let mut s = seed | 1;
        let mut next = || { s ^= s << 13; s ^= s >> 7; s ^= s << 17; (s >> 11) as f64 / (1u64 << 53) as f64 };
        let x: Vec<f64> = (0..n * h).map(|i| if next() < missing { f64::NAN } else { (next() - 0.5) * 10.0f64.powi((i % 4) as i32) }).collect();
        let (p, t) = Prep::fit(&x, n, h, &[false, true], 4.0);
        let k = p.keep.len();
        let mut row = vec![0.0; k];
        for (v, tv) in t.iter().enumerate() {
            for r in 0..n {
                p.apply(v, &x[r * h..(r + 1) * h], &mut row);
                for j in 0..k {
                    prop_assert!(row[j].is_finite());
                    prop_assert!((row[j] - tv[r * k + j]).abs() <= 1e-9 * (1.0 + tv[r * k + j].abs()), "variant {v} row {r} col {j}: {} vs {}", row[j], tv[r * k + j]);
                }
            }
        }
    }

    #[test]
    fn minimiser_finds_parabola_minimum(c in -5.0f64..5.0, w in 0.1f64..10.0) {
        let x = minimize_bounded(|x| w * (x - c) * (x - c), -10.0, 10.0, 1e-9);
        prop_assert!((x - c).abs() < 1e-6, "{x} vs {c}");
        // a minimum outside the bounds ends at the nearer bound
        let x = minimize_bounded(|x| (x - 20.0).powi(2), -1.0, 1.0, 1e-9);
        prop_assert!((x - 1.0).abs() < 1e-6);
    }
}

/// The published checkpoints, read by our own reader, give exactly the bundle the Python export wrote, and the
/// half-size bundle matches the committed web copy. Needs ref/ckpt and fixtures/ (skipped otherwise).
#[test]
fn published_checkpoints_convert_exactly() {
    for (ck, kind) in [("tabicl-classifier-v2-20260212.ckpt", "classifier"), ("tabicl-regressor-v2-20260212.ckpt", "regressor")] {
        let (Ok(bytes), Some(want)) = (std::fs::read(root().join("ref/ckpt").join(ck)), fixture(&format!("{kind}-v2"))) else {
            eprintln!("ref/ckpt or fixtures missing, skipping {kind}");
            continue;
        };
        let ours = tabicl_anywhere::ckpt::bundle_bytes(bytes, false).unwrap();
        let ob = bytes_bundle(ours.clone());
        assert_eq!(ob.entries.len(), want.entries.len());
        for e in &want.entries {
            assert_eq!(ob.bytes(&e.name).unwrap(), want.bytes(&e.name).unwrap(), "{kind}: {}", e.name);
        }
        let half = tabicl_anywhere::ckpt::bundle_bytes(ours, true).unwrap();
        if let Some(docs) = docs_weights(kind) {
            let db: Vec<u8> = docs.entries.iter().flat_map(|e| docs.bytes(&e.name).unwrap()).collect();
            let hb = bytes_bundle(half);
            let hv: Vec<u8> = hb.entries.iter().flat_map(|e| hb.bytes(&e.name).unwrap()).collect();
            assert!(db == hv, "{kind}: the half-size weights differ from docs/weights");
        }
    }
}
