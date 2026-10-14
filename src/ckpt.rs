//! Read a published TabICL checkpoint (a PyTorch `.ckpt` file) and write it as a weight bundle, without Python.
//!
//! Such a file is a zip archive, stored uncompressed, holding `data.pkl` (a Python pickle describing a dictionary
//! `{"config": {...}, "state_dict": {name: tensor}}`) and one raw little-endian blob per tensor storage. The
//! pickle is a small stack-machine program; this module runs just the instructions PyTorch emits and understands
//! just the three constructors that appear (ordered dictionary, tensor storage reference, tensor). Nothing in the
//! pickle is executed as code, so an untrusted file can at worst be rejected.

use crate::store::{f32_to_f16, Bundle, BundleWriter, ReadAt, Storage};
use std::collections::HashMap;
use std::fs::File;
use std::io;
use std::path::Path;

fn bad(msg: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg.into())
}

#[derive(Clone, Debug)]
enum V {
    None,
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(String),
    Tuple(Vec<V>),
    List(Vec<V>),
    Dict(Vec<(V, V)>),
    Global(String),
    /// A tensor's storage: blob key and element type.
    Storage(String, String),
    Tensor {
        key: String,
        dtype: String,
        offset: usize,
        shape: Vec<usize>,
    },
    Mark,
    Other,
}

fn run_pickle(p: &[u8]) -> io::Result<V> {
    let mut stack: Vec<V> = Vec::new();
    let mut memo: HashMap<u32, V> = HashMap::new();
    let mut i = 0;
    let take = |i: &mut usize, n: usize| -> io::Result<&[u8]> {
        let s = p.get(*i..*i + n).ok_or_else(|| bad("pickle ends early"))?;
        *i += n;
        Ok(s)
    };
    let pop_mark = |stack: &mut Vec<V>| -> io::Result<Vec<V>> {
        let at = stack.iter().rposition(|v| matches!(v, V::Mark)).ok_or_else(|| bad("pickle: no mark"))?;
        let items = stack.split_off(at + 1);
        stack.pop();
        Ok(items)
    };
    loop {
        let op = take(&mut i, 1)?[0];
        match op {
            0x80 => i += 1, // PROTO
            b'.' => return stack.pop().ok_or_else(|| bad("pickle: empty at stop")),
            b'}' => stack.push(V::Dict(Vec::new())),
            b']' => stack.push(V::List(Vec::new())),
            b')' => stack.push(V::Tuple(Vec::new())),
            b'(' => stack.push(V::Mark),
            b'N' => stack.push(V::None),
            0x88 => stack.push(V::Bool(true)),
            0x89 => stack.push(V::Bool(false)),
            b'K' => stack.push(V::Int(take(&mut i, 1)?[0] as i64)),
            b'M' => stack.push(V::Int(u16::from_le_bytes(take(&mut i, 2)?.try_into().unwrap()) as i64)),
            b'J' => stack.push(V::Int(i32::from_le_bytes(take(&mut i, 4)?.try_into().unwrap()) as i64)),
            0x8a => {
                let n = take(&mut i, 1)?[0] as usize;
                let b = take(&mut i, n)?;
                let mut v = 0i64;
                for (k, byte) in b.iter().enumerate().take(8) {
                    v |= (*byte as i64) << (8 * k);
                }
                if n > 0 && n < 8 && b[n - 1] & 0x80 != 0 {
                    v -= 1i64 << (8 * n);
                }
                stack.push(V::Int(v));
            }
            b'G' => stack.push(V::Float(f64::from_be_bytes(take(&mut i, 8)?.try_into().unwrap()))),
            b'X' | 0x8c | b'U' => {
                let n = if op == b'X' { u32::from_le_bytes(take(&mut i, 4)?.try_into().unwrap()) as usize } else { take(&mut i, 1)?[0] as usize };
                stack.push(V::Str(String::from_utf8_lossy(take(&mut i, n)?).into_owned()));
            }
            b'c' => {
                let rest = &p[i..];
                let a = rest.iter().position(|b| *b == b'\n').ok_or_else(|| bad("pickle: bad global"))?;
                let b = rest[a + 1..].iter().position(|b| *b == b'\n').ok_or_else(|| bad("pickle: bad global"))?;
                stack.push(V::Global(format!("{}.{}", String::from_utf8_lossy(&rest[..a]), String::from_utf8_lossy(&rest[a + 1..a + 1 + b]))));
                i += a + b + 2;
            }
            b'q' => {
                let k = take(&mut i, 1)?[0] as u32;
                memo.insert(k, stack.last().cloned().ok_or_else(|| bad("pickle: put on empty"))?);
            }
            b'r' => {
                let k = u32::from_le_bytes(take(&mut i, 4)?.try_into().unwrap());
                memo.insert(k, stack.last().cloned().ok_or_else(|| bad("pickle: put on empty"))?);
            }
            0x94 => {
                let k = memo.len() as u32;
                memo.insert(k, stack.last().cloned().ok_or_else(|| bad("pickle: memoize on empty"))?);
            }
            b'h' => {
                let k = take(&mut i, 1)?[0] as u32;
                stack.push(memo.get(&k).cloned().ok_or_else(|| bad("pickle: unknown memo"))?);
            }
            b'j' => {
                let k = u32::from_le_bytes(take(&mut i, 4)?.try_into().unwrap());
                stack.push(memo.get(&k).cloned().ok_or_else(|| bad("pickle: unknown memo"))?);
            }
            0x85..=0x87 => {
                let n = (op - 0x84) as usize;
                let at = stack.len().checked_sub(n).ok_or_else(|| bad("pickle: short stack"))?;
                let items = stack.split_off(at);
                stack.push(V::Tuple(items));
            }
            b't' => {
                let items = pop_mark(&mut stack)?;
                stack.push(V::Tuple(items));
            }
            b'l' => {
                let items = pop_mark(&mut stack)?;
                stack.push(V::List(items));
            }
            b'a' | b'e' => {
                let items = if op == b'a' { vec![stack.pop().ok_or_else(|| bad("pickle: short stack"))?] } else { pop_mark(&mut stack)? };
                if let Some(V::List(l)) = stack.last_mut() {
                    l.extend(items);
                }
            }
            b's' | b'u' => {
                let items = if op == b's' {
                    let at = stack.len().checked_sub(2).ok_or_else(|| bad("pickle: short stack"))?;
                    stack.split_off(at)
                } else {
                    pop_mark(&mut stack)?
                };
                if let Some(V::Dict(d)) = stack.last_mut() {
                    let mut it = items.into_iter();
                    while let (Some(k), Some(v)) = (it.next(), it.next()) {
                        d.push((k, v));
                    }
                }
            }
            b'Q' => {
                // persistent id: ('storage', <storage class>, key, location, element count)
                let v = match stack.pop() {
                    Some(V::Tuple(t)) if t.len() >= 3 => match (&t[1], &t[2]) {
                        (V::Global(g), V::Str(key)) => {
                            let dtype = match g.rsplit('.').next().unwrap_or("") {
                                "FloatStorage" => "float32",
                                "DoubleStorage" => "float64",
                                "HalfStorage" => "float16",
                                "LongStorage" => "int64",
                                "IntStorage" => "int32",
                                other => return Err(bad(format!("checkpoint uses storage type {other}, which is not handled"))),
                            };
                            V::Storage(key.clone(), dtype.to_string())
                        }
                        _ => V::Other,
                    },
                    _ => V::Other,
                };
                stack.push(v);
            }
            b'R' | 0x81 => {
                let args = stack.pop().ok_or_else(|| bad("pickle: short stack"))?;
                let f = stack.pop().ok_or_else(|| bad("pickle: short stack"))?;
                let v = match (&f, &args) {
                    (V::Global(g), _) if g == "collections.OrderedDict" => V::Dict(Vec::new()),
                    (V::Global(g), V::Tuple(a)) if g == "torch._utils._rebuild_tensor_v2" && a.len() >= 3 => match (&a[0], &a[1], &a[2]) {
                        (V::Storage(key, dtype), V::Int(off), V::Tuple(shape)) => {
                            let shape: Vec<usize> = shape.iter().map(|d| if let V::Int(n) = d { usize::try_from(*n).unwrap_or(usize::MAX) } else { 0 }).collect();
                            // only plain, densely packed tensors are expected; check the strides say so
                            if let Some(V::Tuple(strides)) = a.get(3) {
                                let mut want = 1i64;
                                for (d, s) in shape.iter().zip(strides).rev() {
                                    if !matches!(s, V::Int(n) if *n == want) && *d > 1 {
                                        return Err(bad("checkpoint holds a tensor that is not stored contiguously"));
                                    }
                                    want *= *d as i64;
                                }
                            }
                            V::Tensor { key: key.clone(), dtype: dtype.clone(), offset: usize::try_from(*off).map_err(|_| bad("checkpoint holds a tensor at a negative offset"))?, shape }
                        }
                        _ => V::Other,
                    },
                    (V::Global(g), V::Tuple(a)) if g == "torch._utils._rebuild_parameter" && !a.is_empty() => a[0].clone(),
                    _ => V::Other,
                };
                stack.push(v);
            }
            b'b' => {
                stack.pop(); // BUILD: extra attributes (ordered-dict metadata); not needed
            }
            other => return Err(bad(format!("pickle: instruction 0x{other:02x} is not handled"))),
        }
    }
}

fn json(v: &V) -> String {
    match v {
        V::None => "null".into(),
        V::Bool(b) => b.to_string(),
        V::Int(n) => n.to_string(),
        // as Python writes them: a float always has a decimal point (0.0, not 0)
        V::Float(f) => format!("{f:?}"),
        V::Str(s) => format!("\"{}\"", s.replace('\\', "\\\\").replace('"', "\\\"")),
        V::Tuple(t) | V::List(t) => format!("[{}]", t.iter().map(json).collect::<Vec<_>>().join(", ")),
        V::Dict(d) => format!("{{{}}}", d.iter().map(|(k, v)| format!("{}: {}", json(k), json(v))).collect::<Vec<_>>().join(", ")),
        _ => "null".into(),
    }
}

/// Where each stored (uncompressed) member of a zip archive starts and how long it is. Every field is checked
/// against the bytes really there, so a damaged archive is an error, never a panic or a huge allocation.
fn zip_members(f: &dyn ReadAt, len: u64) -> io::Result<HashMap<String, (u64, u64)>> {
    let tail_len = len.min(66_000) as usize;
    let mut tail = vec![0u8; tail_len];
    f.read_exact_at(&mut tail, len - tail_len as u64)?;
    let short = || bad("zip directory is damaged");
    let field = |b: &[u8], o: usize, n: usize| -> io::Result<u64> {
        let s = b.get(o..o + n).ok_or_else(short)?;
        Ok(s.iter().rev().fold(0u64, |a, v| (a << 8) | *v as u64))
    };
    let u16at = |b: &[u8], o: usize| field(b, o, 2);
    let u32at = |b: &[u8], o: usize| field(b, o, 4);
    let u64at = |b: &[u8], o: usize| field(b, o, 8);
    let at = tail.windows(4).rposition(|w| w == b"PK\x05\x06").ok_or_else(|| bad("not a zip archive (so not a PyTorch checkpoint)"))?;
    let (mut count, mut dir_len, mut dir_at) = (u16at(&tail, at + 10)?, u32at(&tail, at + 12)?, u32at(&tail, at + 16)?);
    if let Some(z) = tail.windows(4).rposition(|w| w == b"PK\x06\x06") {
        // zip64 end record
        (count, dir_len, dir_at) = (u64at(&tail, z + 32)?, u64at(&tail, z + 40)?, u64at(&tail, z + 48)?);
    }
    if dir_at.checked_add(dir_len).is_none_or(|end| end > len) {
        return Err(short());
    }
    let mut dir = vec![0u8; dir_len as usize];
    f.read_exact_at(&mut dir, dir_at)?;
    let mut out = HashMap::new();
    let mut p = 0usize;
    for _ in 0..count {
        if dir.get(p..p + 4) != Some(b"PK\x01\x02") {
            return Err(short());
        }
        if u16at(&dir, p + 10)? != 0 {
            return Err(bad("checkpoint archive is compressed; PyTorch checkpoints are expected to be stored"));
        }
        let (mut size, name_len, extra_len, comment_len, mut local) =
            (u32at(&dir, p + 24)?, u16at(&dir, p + 28)? as usize, u16at(&dir, p + 30)? as usize, u16at(&dir, p + 32)? as usize, u32at(&dir, p + 42)?);
        let name = String::from_utf8_lossy(dir.get(p + 46..p + 46 + name_len).ok_or_else(short)?).into_owned();
        // zip64 extra field carries the sizes and offset that did not fit 32 bits
        let mut e = p + 46 + name_len;
        let extra_end = e + extra_len;
        while e + 4 <= extra_end {
            let (id, sz) = (u16at(&dir, e)?, u16at(&dir, e + 2)? as usize);
            if id == 1 {
                let mut o = e + 4;
                if u32at(&dir, p + 24)? == 0xffff_ffff {
                    size = u64at(&dir, o)?;
                    o += 8;
                }
                if u32at(&dir, p + 20)? == 0xffff_ffff {
                    o += 8;
                }
                if local == 0xffff_ffff {
                    local = u64at(&dir, o)?;
                }
            }
            e += 4 + sz;
        }
        let mut head = [0u8; 30];
        f.read_exact_at(&mut head, local)?;
        let data = local + 30 + u16at(&head, 26)? + u16at(&head, 28)?;
        if data.checked_add(size).is_none_or(|end| end > len) {
            return Err(bad(format!("zip member {name} runs past the end of the archive")));
        }
        out.insert(name, (data, size));
        p = extra_end + comment_len;
    }
    Ok(out)
}

/// Convert `ckpt` to a weight bundle at `out`. Returns how many tensors were written.
pub fn convert(ckpt: impl AsRef<Path>, out: impl AsRef<Path>) -> io::Result<usize> {
    let f = File::open(ckpt)?;
    let len = f.metadata()?.len();
    let store = File::options().read(true).write(true).create(true).truncate(true).open(out)?;
    convert_on(&f, len, Box::new(store))
}

/// As `convert`, from any byte source of `len` bytes to any storage (so a browser can convert a checkpoint it
/// has fetched without touching a file system).
pub fn convert_on(f: &dyn ReadAt, len: u64, out: Box<dyn Storage>) -> io::Result<usize> {
    let members = zip_members(f, len)?;
    let (pkl_name, &(at, len)) = members.iter().find(|(n, _)| n.ends_with("/data.pkl") || n.as_str() == "data.pkl").ok_or_else(|| bad("archive has no data.pkl"))?;
    let root = pkl_name.strip_suffix("data.pkl").unwrap().to_string();
    let mut pkl = vec![0u8; len as usize];
    f.read_exact_at(&mut pkl, at)?;
    let V::Dict(top) = run_pickle(&pkl)? else { return Err(bad("checkpoint is not a dictionary")) };
    let find = |k: &str| top.iter().find(|(key, _)| matches!(key, V::Str(s) if s == k)).map(|(_, v)| v);
    let config = json(find("config").ok_or_else(|| bad("checkpoint has no model configuration"))?);
    let Some(V::Dict(state)) = find("state_dict") else { return Err(bad("checkpoint has no state_dict")) };
    let mut tensors = Vec::new();
    for (k, v) in state {
        if let (V::Str(name), V::Tensor { key, dtype, offset, shape }) = (k, v) {
            let &(data, size) = members.get(&format!("{root}data/{key}")).ok_or_else(|| bad(format!("archive is missing the data for {name}")))?;
            let width = if dtype.ends_with("64") {
                8
            } else if dtype.ends_with("16") {
                2
            } else {
                4
            };
            let n: usize = shape.iter().try_fold(1usize, |a, d| a.checked_mul(*d)).ok_or_else(|| bad(format!("{name}: impossible shape")))?;
            if offset.checked_add(n).and_then(|v| v.checked_mul(width)).is_none_or(|end| end as u64 > size) {
                return Err(bad(format!("{name}: data is shorter than its shape says")));
            }
            tensors.push((name.clone(), dtype.clone(), shape.clone(), data + (offset * width) as u64, n * width));
        }
    }
    let mut ents: Vec<(&str, &str, Vec<usize>)> = tensors.iter().map(|(n, d, s, _, _)| (n.as_str(), d.as_str(), s.clone())).collect();
    ents.push(("config.json", "uint8", vec![config.len()]));
    let w = BundleWriter::create_on(out, &ents)?;
    let mut buf = Vec::new();
    for (name, _, _, at, bytes) in &tensors {
        buf.resize(*bytes, 0u8);
        f.read_exact_at(&mut buf, *at)?;
        w.write_bytes(name, &buf)?;
    }
    w.write_bytes("config.json", config.as_bytes())?;
    Ok(tensors.len())
}

/// Tensors with fewer numbers than this stay at full precision in a half-size bundle: biases, normalisation
/// weights and other small tables are a rounding error of the file's size and exact is simpler to reason about.
pub const HALF_MIN_LEN: usize = 4096;

/// Rewrite the weight bundle `src` with its large 32-bit tensors as 16-bit floats (about half the size).
/// `Model::load` widens them again on reading. Returns (tensors narrowed, tensors copied as they were).
pub fn halve_on(src: &Bundle, out: Box<dyn Storage>) -> io::Result<(usize, usize)> {
    let narrow = |e: &crate::store::Entry| e.dtype == "float32" && e.len() >= HALF_MIN_LEN;
    let ents: Vec<(&str, &str, Vec<usize>)> = src.entries.iter().map(|e| (e.name.as_str(), if narrow(e) { "float16" } else { e.dtype.as_str() }, e.dims.clone())).collect();
    let w = BundleWriter::create_on(out, &ents)?;
    let mut count = (0, 0);
    for e in &src.entries {
        if narrow(e) {
            let wide = src.f32s(&e.name)?;
            let half: Vec<u8> = wide.iter().flat_map(|v| f32_to_f16(*v).to_le_bytes()).collect();
            w.write_bytes(&e.name, &half)?;
            count.0 += 1;
        } else {
            w.write_bytes(&e.name, &src.bytes(&e.name)?)?;
            count.1 += 1;
        }
    }
    w.file.sync()?;
    Ok(count)
}

/// A weight bundle from the bytes of either a published checkpoint or a bundle, optionally narrowed to 16 bits.
/// This is what both the command line and the browser use to prepare weights.
pub fn bundle_bytes(bytes: Vec<u8>, half: bool) -> io::Result<Vec<u8>> {
    let bytes = if bytes.starts_with(b"TIRSBNDL") {
        bytes
    } else {
        let len = bytes.len() as u64;
        let out = crate::store::MemStore::new();
        convert_on(&std::sync::Arc::new(bytes), len, Box::new(out.clone()))?;
        out.bytes()
    };
    if !half {
        return Ok(bytes);
    }
    let src = Bundle::from_source(Box::new(std::sync::Arc::new(bytes)))?;
    let out = crate::store::MemStore::new();
    halve_on(&src, Box::new(out.clone()))?;
    Ok(out.bytes())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    fn run(p: &[u8]) -> io::Result<String> {
        run_pickle(p).map(|v| json(&v))
    }

    #[test]
    fn pickle_instructions() {
        // PROTO 2; a dict built from a mark with every kind of value PyTorch's pickles contain
        let mut p = vec![0x80, 2, b'}', b'('];
        p.extend([0x8c, 1, b'a', b'K', 7]); // short string key, one-byte int
        p.extend([0x8c, 1, b'b', b'M', 0x34, 0x12]); // two-byte int
        p.extend([0x8c, 1, b'c', b'J', 0xff, 0xff, 0xff, 0xff]); // four-byte int, negative
        p.extend([0x8c, 1, b'd', 0x8a, 2, 0x00, 0x80]); // LONG1, negative (two bytes, top bit set)
        p.extend([0x8c, 1, b'e', 0x8a, 1, 0x05]); // LONG1, positive
        p.extend([0x8c, 1, b'f', b'G']);
        p.extend(1.5f64.to_be_bytes()); // binary float
        p.extend([b'X', 2, 0, 0, 0, b'q', b'"', b'N']); // long string key, None value
        p.extend([b'U', 1, b'g', 0x88]); // short string key, True
        p.extend([b'U', 1, b'h', 0x89]); // False
        p.extend([b'U', 1, b'i', b'(', b'K', 1, b'K', 2, b't']); // tuple from a mark
        p.extend([b'U', 1, b'j', b']', b'K', 1, b'a', b'(', b'K', 2, b'K', 3, b'e']); // list: append, appends
        p.extend([b'U', 1, b'k', b'(', b'K', 4, b'l']); // list from a mark
        p.extend([b'U', 1, b'l', b'(', b')', b'K', 1, 0x85, b'K', 1, b'K', 2, 0x86, b'K', 1, b'K', 2, b'K', 3, 0x87, b't']); // tuples
        p.push(b'u'); // SETITEMS
        p.extend([b'U', 1, b'm', b'K', 9, b's']); // SETITEM
        p.push(b'.');
        let got = run(&p).unwrap();
        assert_eq!(
            got,
            r#"{"a": 7, "b": 4660, "c": -1, "d": -32768, "e": 5, "f": 1.5, "q\"": null, "g": true, "h": false, "i": [1, 2], "j": [1, 2, 3], "k": [4], "l": [[], [1], [1, 2], [1, 2, 3]], "m": 9}"#
        );
    }

    #[test]
    fn pickle_memo_globals_and_tensors() {
        // _rebuild_tensor_v2(storage, offset 0, shape (2, 3), strides (3, 1)), the storage given by a persistent
        // id ('storage', FloatStorage, '0', 'cpu', 6): how PyTorch pickles a tensor
        let mut q = vec![0x80, 2];
        q.extend(b"ctorch._utils\n_rebuild_tensor_v2\n");
        q.extend([b'(', b'(', 0x8c, 7]);
        q.extend(b"storage");
        q.extend(b"ctorch\nFloatStorage\n");
        q.extend([0x8c, 1, b'0', 0x8c, 3]);
        q.extend(b"cpu");
        q.extend([b'K', 6, b't', b'Q', b'K', 0, b'K', 2, b'K', 3, 0x86, b'K', 3, b'K', 1, 0x86, b't', b'R']);
        q.push(b'.');
        match run_pickle(&q).unwrap() {
            V::Tensor { key, dtype, offset, shape } => assert_eq!((key.as_str(), dtype.as_str(), offset, shape), ("0", "float32", 0, vec![2, 3])),
            other => panic!("{other:?}"),
        }
        let swap = |from: &[u8], to: &[u8]| -> Vec<u8> {
            let at = q.windows(from.len()).position(|w| w == from).unwrap();
            [&q[..at], to, &q[at + from.len()..]].concat()
        };
        // strides that are not contiguous are refused
        assert!(run(&swap(&[b'K', 3, b'K', 1, 0x86], &[b'K', 1, b'K', 2, 0x86])).unwrap_err().to_string().contains("contiguously"));
        // an unknown storage type is refused
        let half = swap(b"FloatStorage", b"BFloat16Storage");
        assert!(run(&half).unwrap_err().to_string().contains("BFloat16Storage"));
        // memo instructions work on their own
        let mut m = vec![0x80, 2, b'K', 5, b'q', 3, b'h', 3, 0x94, b'r', 9, 0, 0, 0, b'j', 9, 0, 0, 0, 0x87, b'.'];
        assert_eq!(run(&m).unwrap(), "[5, 5, 5]");
        // OrderedDict() and _rebuild_parameter(x, ...) and BUILD
        m = b"\x80\x02ccollections\nOrderedDict\n)R}b.".to_vec();
        assert_eq!(run(&m).unwrap(), "{}");
        m = b"\x80\x02ctorch._utils\n_rebuild_parameter\nK\x04\x85R.".to_vec();
        assert_eq!(run(&m).unwrap(), "4");
        m = b"\x80\x02cfoo\nbar\nK\x04\x85R.".to_vec();
        assert_eq!(run(&m).unwrap(), "null", "an unknown constructor becomes an opaque value, never a call");
    }

    #[test]
    fn broken_pickles_are_errors() {
        for (p, what) in [
            (&b""[..], "ends early"),
            (b"\x80\x02.", "empty at stop"),
            (b"\x80\x02K", "ends early"),
            (b"\x80\x02t.", "no mark"),
            (b"\x80\x02\x86.", "short stack"),
            (b"\x80\x02a.", "short stack"),
            (b"\x80\x02s.", "short stack"),
            (b"\x80\x02R.", "short stack"),
            (b"\x80\x02h\x07.", "unknown memo"),
            (b"\x80\x02q\x00.", "put on empty"),
            (b"\x80\x02\x94.", "memoize on empty"),
            (b"\x80\x02cno-newline", "bad global"),
            (b"\x80\x02X\xff\xff\xff\xff", "ends early"),
            (b"\x80\x02Z.", "0x5a is not handled"),
        ] {
            let e = run(p).unwrap_err().to_string();
            assert!(e.contains(what), "{p:?}: {e}");
        }
    }

    /// A stored zip archive written by hand, optionally in the zip64 form (sizes and offsets in extra fields).
    fn zip(files: &[(&str, &[u8])], zip64: bool) -> Vec<u8> {
        let mut out = Vec::new();
        let mut dir = Vec::new();
        for (name, data) in files {
            let at = out.len() as u64;
            out.extend(b"PK\x03\x04");
            out.extend([0u8; 22]);
            out.extend((name.len() as u16).to_le_bytes());
            out.extend(0u16.to_le_bytes());
            out.extend(name.as_bytes());
            out.extend(*data);
            dir.extend(b"PK\x01\x02");
            dir.extend([0u8; 6]);
            dir.extend(0u16.to_le_bytes()); // stored
            dir.extend([0u8; 8]);
            let (csize, size, local) = if zip64 { (u32::MAX, u32::MAX, u32::MAX) } else { (data.len() as u32, data.len() as u32, at as u32) };
            dir.extend(csize.to_le_bytes());
            dir.extend(size.to_le_bytes());
            dir.extend((name.len() as u16).to_le_bytes());
            dir.extend((if zip64 { 28u16 } else { 0 }).to_le_bytes());
            dir.extend([0u8; 10]); // comment length, disk, attributes
            dir.extend(local.to_le_bytes());
            dir.extend(name.as_bytes());
            if zip64 {
                dir.extend(1u16.to_le_bytes());
                dir.extend(24u16.to_le_bytes());
                dir.extend((data.len() as u64).to_le_bytes()); // uncompressed
                dir.extend((data.len() as u64).to_le_bytes()); // compressed
                dir.extend(at.to_le_bytes());
            }
        }
        let dir_at = out.len() as u64;
        out.extend(&dir);
        if zip64 {
            out.extend(b"PK\x06\x06");
            out.extend(44u64.to_le_bytes());
            out.extend([0u8; 20]);
            out.extend((files.len() as u64).to_le_bytes());
            out.extend((dir.len() as u64).to_le_bytes());
            out.extend(dir_at.to_le_bytes());
            out.extend(b"PK\x06\x07");
            out.extend([0u8; 16]);
        }
        out.extend(b"PK\x05\x06");
        out.extend([0u8; 4]);
        out.extend((files.len() as u16).to_le_bytes());
        out.extend((files.len() as u16).to_le_bytes());
        out.extend((dir.len() as u32).to_le_bytes());
        out.extend((dir_at as u32).to_le_bytes());
        out.extend(0u16.to_le_bytes());
        out
    }

    #[test]
    fn zip_archives_plain_and_zip64() {
        let pkl: &[u8] = b"\x80\x02}(\x8c\x06config}\x8c\x0astate_dict}u.";
        for z64 in [false, true] {
            let z = zip(&[("m/data.pkl", pkl), ("m/version", b"3")], z64);
            let members = zip_members(&Arc::new(z.clone()), z.len() as u64).unwrap();
            let (at, len) = members["m/version"];
            assert_eq!(&z[at as usize..(at + len) as usize], b"3", "zip64 {z64}");
            // a checkpoint with a configuration and no tensors converts to a bundle holding just the configuration
            let out = crate::store::MemStore::new();
            assert_eq!(convert_on(&Arc::new(z), 0, Box::new(out.clone())).err().map(|_| ()), Some(()), "a zero length is not an archive");
            let z = zip(&[("m/data.pkl", pkl)], z64);
            assert_eq!(convert_on(&Arc::new(z.clone()), z.len() as u64, Box::new(out.clone())).unwrap(), 0);
            assert_eq!(Bundle::from_source(Box::new(Arc::new(out.bytes()))).unwrap().bytes("config.json").unwrap(), b"{}");
        }
        // not a dictionary, no configuration, no state_dict, a tensor without its data
        for (p, what) in [(&b"\x80\x02K\x01."[..], "not a dictionary"), (b"\x80\x02}.", "no model configuration"), (b"\x80\x02}(\x8c\x06config}u.", "no state_dict")] {
            let z = zip(&[("data.pkl", p)], false);
            let e = convert_on(&Arc::new(z.clone()), z.len() as u64, Box::new(crate::store::MemStore::new())).unwrap_err().to_string();
            assert!(e.contains(what), "{e}");
        }
    }
}
