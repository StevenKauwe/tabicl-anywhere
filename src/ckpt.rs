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
    Tensor { key: String, dtype: String, offset: usize, shape: Vec<usize> },
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
            0x80 => i += 1,                               // PROTO
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
            0x85 | 0x86 | 0x87 => {
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
                            let shape: Vec<usize> = shape.iter().map(|d| if let V::Int(n) = d { *n as usize } else { 0 }).collect();
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
                            V::Tensor { key: key.clone(), dtype: dtype.clone(), offset: *off as usize, shape }
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
        V::Float(f) => format!("{f}"),
        V::Str(s) => format!("\"{}\"", s.replace('\\', "\\\\").replace('"', "\\\"")),
        V::Tuple(t) | V::List(t) => format!("[{}]", t.iter().map(json).collect::<Vec<_>>().join(", ")),
        V::Dict(d) => format!("{{{}}}", d.iter().map(|(k, v)| format!("{}: {}", json(k), json(v))).collect::<Vec<_>>().join(", ")),
        _ => "null".into(),
    }
}

/// Where each stored (uncompressed) member of a zip archive starts and how long it is.
fn zip_members(f: &dyn ReadAt, len: u64) -> io::Result<HashMap<String, (u64, u64)>> {
    let tail_len = len.min(66_000) as usize;
    let mut tail = vec![0u8; tail_len];
    f.read_exact_at(&mut tail, len - tail_len as u64)?;
    let at = tail.windows(4).rposition(|w| w == b"PK\x05\x06").ok_or_else(|| bad("not a zip archive (so not a PyTorch checkpoint)"))?;
    let u16at = |b: &[u8], o: usize| u16::from_le_bytes(b[o..o + 2].try_into().unwrap()) as u64;
    let u32at = |b: &[u8], o: usize| u32::from_le_bytes(b[o..o + 4].try_into().unwrap()) as u64;
    let u64at = |b: &[u8], o: usize| u64::from_le_bytes(b[o..o + 8].try_into().unwrap());
    let (mut count, mut dir_len, mut dir_at) = (u16at(&tail, at + 10), u32at(&tail, at + 12), u32at(&tail, at + 16));
    if let Some(z) = tail.windows(4).rposition(|w| w == b"PK\x06\x06") {
        // zip64 end record
        (count, dir_len, dir_at) = (u64at(&tail, z + 32), u64at(&tail, z + 40), u64at(&tail, z + 48));
    }
    let mut dir = vec![0u8; dir_len as usize];
    f.read_exact_at(&mut dir, dir_at)?;
    let mut out = HashMap::new();
    let mut p = 0usize;
    for _ in 0..count {
        if dir.get(p..p + 4) != Some(b"PK\x01\x02") {
            return Err(bad("zip directory is damaged"));
        }
        if u16at(&dir, p + 10) != 0 {
            return Err(bad("checkpoint archive is compressed; PyTorch checkpoints are expected to be stored"));
        }
        let (mut size, name_len, extra_len, comment_len, mut local) = (u32at(&dir, p + 24), u16at(&dir, p + 28) as usize, u16at(&dir, p + 30) as usize, u16at(&dir, p + 32) as usize, u32at(&dir, p + 42));
        let name = String::from_utf8_lossy(&dir[p + 46..p + 46 + name_len]).into_owned();
        // zip64 extra field carries the sizes and offset that did not fit 32 bits
        let mut e = p + 46 + name_len;
        let extra_end = e + extra_len;
        while e + 4 <= extra_end {
            let (id, sz) = (u16at(&dir, e), u16at(&dir, e + 2) as usize);
            if id == 1 {
                let mut o = e + 4;
                if u32at(&dir, p + 24) == 0xffff_ffff {
                    size = u64at(&dir, o);
                    o += 8;
                }
                if u32at(&dir, p + 20) == 0xffff_ffff {
                    o += 8;
                }
                if local == 0xffff_ffff {
                    local = u64at(&dir, o);
                }
            }
            e += 4 + sz;
        }
        let mut head = [0u8; 30];
        f.read_exact_at(&mut head, local)?;
        let data = local + 30 + u16at(&head, 26) + u16at(&head, 28);
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
            let width = if dtype.ends_with("64") { 8 } else if dtype.ends_with("16") { 2 } else { 4 };
            let n: usize = shape.iter().product();
            if ((offset + n) * width) as u64 > size {
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
