//! Tensor bundle: the one file format used for weights, fitted caches and test vectors.
//!
//! ```text
//!   bytes 0..8    magic "TIRSBNDL"
//!   bytes 8..16   u64 little-endian: length of the index in bytes
//!   index         utf-8 text, one line per tensor:  name dtype offset nbytes dim0 dim1 ...
//!   data          each tensor starts on a 64-byte boundary (offset counts from the start of the file)
//! ```
//!
//! Reading never maps the file: callers ask for a range and it is copied into a buffer they own, so memory use is
//! whatever buffers the caller keeps, however large the file.

use std::collections::HashMap;
use std::fs::File;
use std::io;
use std::sync::{Arc, Mutex};
use std::path::Path;

#[cfg(not(target_endian = "little"))]
compile_error!("BrowserTabICL stores little-endian floats and reads them in place; big-endian targets are not supported");

const MAGIC: &[u8; 8] = b"TIRSBNDL";

#[derive(Clone, Debug)]
pub struct Entry {
    pub name: String,
    pub dtype: String,
    pub offset: u64,
    pub nbytes: u64,
    pub dims: Vec<usize>,
}

impl Entry {
    pub fn len(&self) -> usize {
        self.dims.iter().product()
    }
}

/// Anything that can hand back a byte range: a local file, bytes already in memory, or (implemented by the caller)
/// a ranged request to object storage such as S3. Bundles are laid out in the order prediction reads them, and
/// prediction asks for large ranges, strictly forwards within each pass, so a slow source costs few round trips.
pub trait ReadAt: Send + Sync {
    fn read_exact_at(&self, buf: &mut [u8], offset: u64) -> io::Result<()>;
}

/// Positioned read on a file, on any platform (an error where there are no files, as in a browser).
pub fn file_read_at(f: &File, buf: &mut [u8], offset: u64) -> io::Result<()> {
    #[cfg(unix)]
    return std::os::unix::fs::FileExt::read_exact_at(f, buf, offset);
    #[cfg(windows)]
    {
        let (mut done, mut at) = (0, offset);
        while done < buf.len() {
            let n = std::os::windows::fs::FileExt::seek_read(f, &mut buf[done..], at)?;
            if n == 0 {
                return Err(io::ErrorKind::UnexpectedEof.into());
            }
            done += n;
            at += n as u64;
        }
        return Ok(());
    }
    #[cfg(not(any(unix, windows)))]
    {
        let _ = (f, buf, offset);
        Err(io::Error::new(io::ErrorKind::Unsupported, "no file access on this platform"))
    }
}

/// Positioned write on a file, on any platform.
pub fn file_write_at(f: &File, data: &[u8], offset: u64) -> io::Result<()> {
    #[cfg(unix)]
    return std::os::unix::fs::FileExt::write_all_at(f, data, offset);
    #[cfg(windows)]
    {
        let (mut done, mut at) = (0, offset);
        while done < data.len() {
            let n = std::os::windows::fs::FileExt::seek_write(f, &data[done..], at)?;
            done += n;
            at += n as u64;
        }
        return Ok(());
    }
    #[cfg(not(any(unix, windows)))]
    {
        let _ = (f, data, offset);
        Err(io::Error::new(io::ErrorKind::Unsupported, "no file access on this platform"))
    }
}

impl ReadAt for File {
    fn read_exact_at(&self, buf: &mut [u8], offset: u64) -> io::Result<()> {
        file_read_at(self, buf, offset)
    }
}

/// Somewhere a bundle can be written: positioned writes (from several threads at once) and reading back.
pub trait Storage: ReadAt {
    fn write_all_at(&self, data: &[u8], offset: u64) -> io::Result<()>;
    fn set_len(&self, len: u64) -> io::Result<()>;
    fn sync(&self) -> io::Result<()> {
        Ok(())
    }
}

impl Storage for File {
    fn write_all_at(&self, data: &[u8], offset: u64) -> io::Result<()> {
        file_write_at(self, data, offset)
    }
    fn set_len(&self, len: u64) -> io::Result<()> {
        File::set_len(self, len)
    }
    fn sync(&self) -> io::Result<()> {
        self.sync_all()
    }
}

/// A bundle written into memory: what fitting uses where there are no files (a browser), or when the caller
/// wants the bytes to put somewhere of its own. Keep a clone of the `Arc` and call `bytes()` afterwards.
#[derive(Default)]
pub struct MemStore(Mutex<Vec<u8>>);

impl MemStore {
    pub fn new() -> Arc<MemStore> {
        Arc::new(MemStore::default())
    }
    /// Take the bytes written so far, leaving the store empty.
    pub fn bytes(&self) -> Vec<u8> {
        std::mem::take(&mut *self.0.lock().unwrap())
    }
}

impl ReadAt for Arc<MemStore> {
    fn read_exact_at(&self, buf: &mut [u8], offset: u64) -> io::Result<()> {
        let all = self.0.lock().unwrap();
        let end = offset as usize + buf.len();
        if end > all.len() {
            return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "read past the end of the bundle bytes"));
        }
        buf.copy_from_slice(&all[offset as usize..end]);
        Ok(())
    }
}

impl Storage for Arc<MemStore> {
    fn write_all_at(&self, data: &[u8], offset: u64) -> io::Result<()> {
        let mut all = self.0.lock().unwrap();
        let end = offset as usize + data.len();
        if end > all.len() {
            all.resize(end, 0);
        }
        all[offset as usize..end].copy_from_slice(data);
        Ok(())
    }
    fn set_len(&self, len: u64) -> io::Result<()> {
        self.0.lock().unwrap().resize(len as usize, 0);
        Ok(())
    }
}

impl<T: AsRef<[u8]> + Send + Sync> ReadAt for Arc<T> {
    fn read_exact_at(&self, buf: &mut [u8], offset: u64) -> io::Result<()> {
        let all: &[u8] = (**self).as_ref();
        let end = offset as usize + buf.len();
        if end > all.len() {
            return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "read past the end of the bundle bytes"));
        }
        buf.copy_from_slice(&all[offset as usize..end]);
        Ok(())
    }
}

/// The value of a 16-bit float (IEEE 754 binary16), exactly.
pub fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h as u32) >> 15) << 31;
    let (exp, man) = (((h >> 10) & 0x1f) as u32, (h & 0x3ff) as u32);
    match exp {
        // zero and subnormals: man * 2^-24, exact in 32 bits
        0 => f32::from_bits(sign | (man as f32 * (1.0 / 16_777_216.0)).to_bits()),
        31 => f32::from_bits(sign | 0x7f80_0000 | (man << 13)),
        _ => f32::from_bits(sign | ((exp + 112) << 23) | (man << 13)),
    }
}

/// The nearest 16-bit float (ties to even). Values too large for 16 bits become the largest finite one rather
/// than infinity, so a narrowed weight never turns a sum into infinity; NaN stays NaN.
pub fn f32_to_f16(x: f32) -> u16 {
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let (exp, man) = (((b >> 23) & 0xff) as i32, b & 0x7f_ffff);
    if exp == 255 {
        return sign | if man != 0 { 0x7e00 } else { 0x7bff };
    }
    let e = exp - 112;
    if e >= 31 {
        return sign | 0x7bff;
    }
    // `kept` are the bits that stay, `rest` the bits dropped, `half` the value of the first dropped bit
    let nearest = |kept: u32, rest: u32, half: u32| kept + (rest > half || (rest == half && kept & 1 == 1)) as u32;
    if e <= 0 {
        if e < -10 {
            return sign;
        }
        let (m, shift) = (man | 0x80_0000, (14 - e) as u32);
        return sign | nearest(m >> shift, m & ((1 << shift) - 1), 1 << (shift - 1)) as u16;
    }
    let v = nearest(((e as u32) << 10) | (man >> 13), man & 0x1fff, 0x1000);
    sign | v.min(0x7bff) as u16
}

/// Cut `bytes` into pieces of at most `max` bytes (hosts limit file sizes: GitHub to 100 MB, others lower).
/// Joining is plain concatenation in order.
pub fn split_parts(bytes: &[u8], max: usize) -> Vec<&[u8]> {
    bytes.chunks(max.max(1)).collect()
}

/// The manifest that sits next to the parts and tells a reader what to fetch: total length and each part's file
/// name (relative to the manifest) and length. JSON, written by hand to need no dependency.
pub fn parts_manifest(total: usize, parts: &[(String, usize)]) -> String {
    let list: Vec<String> = parts.iter().map(|(n, b)| format!("    {{\"file\": \"{n}\", \"bytes\": {b}}}")).collect();
    format!("{{\n  \"format\": \"browsertabicl-parts-1\",\n  \"bytes\": {total},\n  \"parts\": [\n{}\n  ]\n}}\n", list.join(",\n"))
}

pub struct Bundle {
    pub file: Box<dyn ReadAt>,
    pub entries: Vec<Entry>,
    by_name: HashMap<String, usize>,
}

fn bad(msg: String) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg)
}

impl Bundle {
    pub fn open(path: impl AsRef<Path>) -> io::Result<Bundle> {
        let path = path.as_ref();
        let file = File::open(path).map_err(|e| io::Error::new(e.kind(), format!("{}: {e}", path.display())))?;
        Bundle::from_source(Box::new(file)).map_err(|e| io::Error::new(e.kind(), format!("{}: {e}", path.display())))
    }

    /// Open a bundle held anywhere `ReadAt` can reach. Costs two small reads (header, then index).
    pub fn from_source(file: Box<dyn ReadAt>) -> io::Result<Bundle> {
        let mut head = [0u8; 16];
        file.read_exact_at(&mut head, 0)?;
        if &head[..8] != MAGIC {
            return Err(bad("not a BrowserTabICL bundle".into()));
        }
        let mut index = vec![0u8; u64::from_le_bytes(head[8..].try_into().unwrap()) as usize];
        file.read_exact_at(&mut index, 16)?;
        let text = String::from_utf8(index).map_err(|e| bad(e.to_string()))?;
        let mut entries = Vec::new();
        for line in text.lines() {
            let mut p = line.split(' ');
            let mut next = || p.next().ok_or_else(|| bad(format!("short index line: {line}")));
            let (name, dtype) = (next()?.to_string(), next()?.to_string());
            let num = |s: &str| s.parse::<u64>().map_err(|e| bad(format!("{line}: {e}")));
            let (offset, nbytes) = (num(next()?)?, num(next()?)?);
            let dims = p.map(|s| num(s).map(|v| v as usize)).collect::<io::Result<Vec<_>>>()?;
            entries.push(Entry { name, dtype, offset, nbytes, dims });
        }
        let by_name = entries.iter().enumerate().map(|(i, e)| (e.name.clone(), i)).collect();
        Ok(Bundle { file, entries, by_name })
    }

    pub fn has(&self, name: &str) -> bool {
        self.by_name.contains_key(name)
    }

    pub fn entry(&self, name: &str) -> io::Result<&Entry> {
        self.by_name.get(name).map(|&i| &self.entries[i]).ok_or_else(|| bad(format!("bundle has no tensor named {name}")))
    }

    /// Copy `out.len()` floats of tensor `name`, starting `skip` floats in.
    /// A tensor stored as 16-bit floats (a half-size weight file) is widened as it is read.
    pub fn read_f32_at(&self, name: &str, skip: usize, out: &mut [f32]) -> io::Result<()> {
        let e = self.entry(name)?;
        if e.dtype == "float16" {
            if (skip + out.len()) * 2 > e.nbytes as usize {
                return Err(bad(format!("{name}: wanted {} float16 from {skip}, have {} bytes", out.len(), e.nbytes)));
            }
            let mut raw = vec![0u8; out.len() * 2];
            self.file.read_exact_at(&mut raw, e.offset + 2 * skip as u64)?;
            for (o, b) in out.iter_mut().zip(raw.chunks_exact(2)) {
                *o = f16_to_f32(u16::from_le_bytes([b[0], b[1]]));
            }
            return Ok(());
        }
        if e.dtype != "float32" || (skip + out.len()) * 4 > e.nbytes as usize {
            return Err(bad(format!("{name}: wanted {} float32 from {skip}, have {} bytes of {}", out.len(), e.nbytes, e.dtype)));
        }
        self.file.read_exact_at(bytemuck::cast_slice_mut(out), e.offset + 4 * skip as u64)
    }

    /// Copy raw bytes of tensor `name`, starting `skip` bytes in.
    pub fn read_bytes_at(&self, name: &str, skip: usize, out: &mut [u8]) -> io::Result<()> {
        let e = self.entry(name)?;
        if skip + out.len() > e.nbytes as usize {
            return Err(bad(format!("{name}: read past the end")));
        }
        self.file.read_exact_at(out, e.offset + skip as u64)
    }

    pub fn f32s(&self, name: &str) -> io::Result<Vec<f32>> {
        let mut v = vec![0.0f32; self.entry(name)?.len()];
        self.read_f32_at(name, 0, &mut v)?;
        Ok(v)
    }

    pub fn i32s(&self, name: &str) -> io::Result<Vec<i32>> {
        let e = self.entry(name)?;
        if e.dtype != "int32" {
            return Err(bad(format!("{name}: is {}, wanted int32", e.dtype)));
        }
        let mut v = vec![0i32; e.len()];
        self.file.read_exact_at(bytemuck::cast_slice_mut(&mut v), e.offset)?;
        Ok(v)
    }

    pub fn bytes(&self, name: &str) -> io::Result<Vec<u8>> {
        let e = self.entry(name)?;
        let mut v = vec![0u8; e.nbytes as usize];
        self.file.read_exact_at(&mut v, e.offset)?;
        Ok(v)
    }
}

/// Writes a bundle whose tensor shapes are all known up front; the data can then be filled in any order and in
/// pieces, which is what lets fitting write a cache far bigger than memory.
pub struct BundleWriter {
    pub file: Box<dyn Storage>,
    entries: Vec<Entry>,
    by_name: HashMap<String, usize>,
}

impl BundleWriter {
    pub fn create(path: impl AsRef<Path>, tensors: &[(&str, &str, Vec<usize>)]) -> io::Result<BundleWriter> {
        let file = File::options().read(true).write(true).create(true).truncate(true).open(path)?;
        BundleWriter::create_on(Box::new(file), tensors)
    }

    /// As `create`, on any storage (for instance a `MemStore`).
    pub fn create_on(file: Box<dyn Storage>, tensors: &[(&str, &str, Vec<usize>)]) -> io::Result<BundleWriter> {
        let size = |d: &str| match d {
            "float32" | "int32" => 4,
            "float16" => 2,
            "uint8" | "int8" => 1,
            "float64" | "int64" => 8,
            _ => panic!("unknown dtype {d}"),
        };
        let line = |n: &str, d: &str, off: u64, nb: u64, dims: &[usize]| {
            let mut s = format!("{n} {d} {off} {nb}");
            for v in dims {
                s += &format!(" {v}");
            }
            s + "\n"
        };
        let sizes: Vec<u64> = tensors.iter().map(|(_, d, dims)| (dims.iter().product::<usize>() * size(d)) as u64).collect();
        let guess: usize = tensors.iter().zip(&sizes).map(|((n, d, dims), nb)| line(n, d, 10u64.pow(15), *nb, dims).len()).sum();
        let mut pos = ((guess + 16 + 63) / 64 * 64) as u64;
        let mut entries = Vec::new();
        let mut index = String::new();
        for ((n, d, dims), nb) in tensors.iter().zip(&sizes) {
            index += &line(n, d, pos, *nb, dims);
            entries.push(Entry { name: n.to_string(), dtype: d.to_string(), offset: pos, nbytes: *nb, dims: dims.clone() });
            pos = (pos + nb + 63) / 64 * 64;
        }
        file.set_len(pos)?;
        let mut head = MAGIC.to_vec();
        head.extend((index.len() as u64).to_le_bytes());
        head.extend(index.as_bytes());
        file.write_all_at(&head, 0)?;
        let by_name = entries.iter().enumerate().map(|(i, e)| (e.name.clone(), i)).collect();
        Ok(BundleWriter { file, entries, by_name })
    }

    pub fn entry(&self, name: &str) -> &Entry {
        &self.entries[self.by_name[name]]
    }

    /// Write `data` into tensor `name`, starting `skip` floats in. Safe to call from several threads at once.
    pub fn write_f32_at(&self, name: &str, skip: usize, data: &[f32]) -> io::Result<()> {
        let e = self.entry(name);
        assert!(e.dtype == "float32" && (skip + data.len()) * 4 <= e.nbytes as usize, "{name}: write past the end");
        self.file.write_all_at(bytemuck::cast_slice(data), e.offset + 4 * skip as u64)
    }

    pub fn write_bytes(&self, name: &str, data: &[u8]) -> io::Result<()> {
        self.write_bytes_at(name, 0, data)
    }

    pub fn write_bytes_at(&self, name: &str, skip: usize, data: &[u8]) -> io::Result<()> {
        let e = self.entry(name);
        assert!((skip + data.len()) as u64 <= e.nbytes, "{name}: write past the end");
        self.file.write_all_at(data, e.offset + skip as u64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn half_floats() {
        // every 16-bit float widens and narrows back to itself
        for h in 0..=u16::MAX {
            let f = f16_to_f32(h);
            if f.is_nan() {
                assert!(f16_to_f32(f32_to_f16(f)).is_nan());
            } else if f.is_infinite() {
                assert_eq!(f32_to_f16(f), (h & 0x8000) | 0x7bff);
            } else {
                assert_eq!(f32_to_f16(f), h, "{h:#06x} = {f}");
            }
        }
        assert_eq!(f16_to_f32(0x3c00), 1.0);
        assert_eq!(f16_to_f32(0xc000), -2.0);
        assert_eq!(f16_to_f32(0x0001), 2.0f32.powi(-24));
        assert_eq!(f16_to_f32(0x7bff), 65504.0);
        // ties go to the even neighbour; 1 + 2^-11 lies exactly between 1 and the next 16-bit float
        assert_eq!(f32_to_f16(1.0 + 2.0f32.powi(-11)), 0x3c00);
        assert_eq!(f32_to_f16(1.0 + 3.0 * 2.0f32.powi(-11)), 0x3c02);
        assert_eq!(f32_to_f16(1e9), 0x7bff);
        assert_eq!(f32_to_f16(1e-9), 0);
        // narrowing is never further off than half a step: check against a brute-force nearest on a sweep
        let mut x = 1.0e-7f32;
        while x < 7.0e4 {
            let got = f16_to_f32(f32_to_f16(x));
            let step = if x < 6.2e-5 { 2.0f32.powi(-24) } else { x * 2.0f32.powi(-10) };
            assert!((got - x).abs() <= step * 0.5 + f32::EPSILON * x || x > 65504.0, "{x} -> {got}");
            x *= 1.000_37;
        }
    }

    #[test]
    fn parts_round_trip() {
        let data: Vec<u8> = (0..100_003u32).map(|i| (i * 31 % 251) as u8).collect();
        let parts = split_parts(&data, 14_000);
        assert_eq!(parts.len(), 8);
        assert!(parts.iter().all(|p| p.len() <= 14_000));
        assert_eq!(parts.concat(), data);
    }
}
