//! Optional encryption of a bundle at rest, built so that streaming still works.
//!
//! The plain bundle is cut into blocks of 64 KiB and each block is sealed on its own with XChaCha20-Poly1305
//! (authenticated: a wrong key or a flipped bit is an error, never silently wrong numbers). Blocks keep their
//! size plus a 16-byte tag, so the position of any byte is known without an index, and reading a range means
//! fetching and opening only the blocks that cover it. `Sealed` does that behind the `ReadAt` interface, which
//! is all the rest of the crate sees; it works the same over a file, bytes in memory, or object storage.
//!
//! ```text
//!   bytes 0..8     magic "TIRSSEAL"
//!   bytes 8..24    file id (random; first 16 bytes of every block's nonce)
//!   bytes 24..32   u64 little-endian: length of the plain bundle
//!   then           blocks of (64 KiB + 16), the last one shorter; block i uses nonce = file id || i
//! ```
//!
//! The key is 32 random bytes that the caller keeps; nothing derived from it is stored in the file.

use crate::store::ReadAt;
use chacha20poly1305::aead::{AeadInPlace, KeyInit};
use chacha20poly1305::{Tag, XChaCha20Poly1305, XNonce};
use std::fs::File;
use std::io::{self, Read, Write};
use std::path::Path;

const MAGIC: &[u8; 8] = b"TIRSSEAL";
const BLOCK: usize = 64 << 10;
const TAG: usize = 16;
const HEAD: usize = 32;

fn bad(msg: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg.to_string())
}

fn nonce(id: &[u8; 16], block: u64) -> XNonce {
    let mut n = [0u8; 24];
    n[..16].copy_from_slice(id);
    n[16..].copy_from_slice(&block.to_le_bytes());
    n.into()
}

/// Read a key file: 32 raw bytes, or 64 hexadecimal characters.
pub fn read_key(path: impl AsRef<Path>) -> io::Result<[u8; 32]> {
    let raw = std::fs::read(path)?;
    if raw.len() == 32 {
        return Ok(raw.try_into().unwrap());
    }
    let text = String::from_utf8_lossy(&raw);
    let hex = text.trim();
    if hex.len() == 64 && hex.bytes().all(|b| b.is_ascii_hexdigit()) {
        let mut key = [0u8; 32];
        for (i, k) in key.iter_mut().enumerate() {
            *k = u8::from_str_radix(&hex[2 * i..2 * i + 2], 16).unwrap();
        }
        return Ok(key);
    }
    Err(bad("a key file must hold 32 raw bytes or 64 hexadecimal characters"))
}

/// A fresh random key from the operating system.
pub fn new_key() -> io::Result<[u8; 32]> {
    let mut key = [0u8; 32];
    getrandom::getrandom(&mut key).map_err(|e| io::Error::other(e.to_string()))?;
    Ok(key)
}

/// Seal the plain bundle at `plain` into `sealed`, a block at a time (memory use: one block).
pub fn seal_file(plain: impl AsRef<Path>, sealed: impl AsRef<Path>, key: &[u8; 32]) -> io::Result<()> {
    let mut src = File::open(plain)?;
    let len = src.metadata()?.len();
    let mut id = [0u8; 16];
    getrandom::getrandom(&mut id).map_err(|e| io::Error::other(e.to_string()))?;
    let cipher = XChaCha20Poly1305::new(key.into());
    let mut out = io::BufWriter::new(File::create(sealed)?);
    out.write_all(MAGIC)?;
    out.write_all(&id)?;
    out.write_all(&len.to_le_bytes())?;
    let mut buf = vec![0u8; BLOCK];
    let mut left = len;
    let mut block = 0u64;
    while left > 0 {
        let n = (left as usize).min(BLOCK);
        src.read_exact(&mut buf[..n])?;
        let tag = cipher.encrypt_in_place_detached(&nonce(&id, block), MAGIC, &mut buf[..n]).map_err(|_| bad("sealing failed"))?;
        out.write_all(&buf[..n])?;
        out.write_all(&tag)?;
        left -= n as u64;
        block += 1;
    }
    out.flush()
}

/// A sealed bundle seen as its plain bytes.
pub struct Sealed<R: ReadAt> {
    inner: R,
    cipher: XChaCha20Poly1305,
    id: [u8; 16],
    len: u64,
}

impl<R: ReadAt> Sealed<R> {
    pub fn open(inner: R, key: &[u8; 32]) -> io::Result<Sealed<R>> {
        let mut head = [0u8; HEAD];
        inner.read_exact_at(&mut head, 0)?;
        if &head[..8] != MAGIC {
            return Err(bad("not a sealed BrowserTabICL bundle"));
        }
        Ok(Sealed { inner, cipher: XChaCha20Poly1305::new(key.into()), id: head[8..24].try_into().unwrap(), len: u64::from_le_bytes(head[24..32].try_into().unwrap()) })
    }
}

impl<R: ReadAt> ReadAt for Sealed<R> {
    fn read_exact_at(&self, buf: &mut [u8], offset: u64) -> io::Result<()> {
        if buf.is_empty() {
            return Ok(());
        }
        let end = offset + buf.len() as u64;
        if end > self.len {
            return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "read past the end of the sealed bundle"));
        }
        let (first, last) = (offset / BLOCK as u64, (end - 1) / BLOCK as u64);
        let plain_len = |b: u64| (self.len - b * BLOCK as u64).min(BLOCK as u64) as usize;
        // one request to the source for every block touched
        let stored = |b: u64| HEAD as u64 + b * (BLOCK + TAG) as u64;
        let total = (stored(last) + (plain_len(last) + TAG) as u64 - stored(first)) as usize;
        let mut raw = vec![0u8; total];
        self.inner.read_exact_at(&mut raw, stored(first))?;
        for b in first..=last {
            let at = (stored(b) - stored(first)) as usize;
            let n = plain_len(b);
            let (body, tag) = raw[at..at + n + TAG].split_at_mut(n);
            self.cipher
                .decrypt_in_place_detached(&nonce(&self.id, b), MAGIC, body, Tag::from_slice(tag))
                .map_err(|_| bad("sealed bundle did not open: wrong key, or the data has been altered"))?;
            let block_start = b * BLOCK as u64;
            let (from, to) = (offset.max(block_start), end.min(block_start + n as u64));
            buf[(from - offset) as usize..(to - offset) as usize].copy_from_slice(&body[(from - block_start) as usize..(to - block_start) as usize]);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn seal_then_read_ranges() {
        let dir = std::env::temp_dir();
        let (p, s) = (dir.join(format!("tirs-seal-{}.plain", std::process::id())), dir.join(format!("tirs-seal-{}.sealed", std::process::id())));
        let plain: Vec<u8> = (0..3 * BLOCK + 12345).map(|i| (i * 31 % 251) as u8).collect();
        std::fs::write(&p, &plain).unwrap();
        let key = new_key().unwrap();
        seal_file(&p, &s, &key).unwrap();
        let bytes = Arc::new(std::fs::read(&s).unwrap());
        assert_eq!(bytes.len(), HEAD + plain.len() + 4 * TAG);
        let sealed = Sealed::open(bytes.clone(), &key).unwrap();
        for (off, len) in [(0usize, 10usize), (BLOCK - 3, 7), (BLOCK, BLOCK), (5, 3 * BLOCK), (3 * BLOCK + 12000, 345), (0, plain.len())] {
            let mut got = vec![0u8; len];
            sealed.read_exact_at(&mut got, off as u64).unwrap();
            assert_eq!(got, plain[off..off + len], "range {off}+{len}");
        }
        // a wrong key and a flipped bit are both refused
        let mut other = key;
        other[0] ^= 1;
        assert!(Sealed::open(bytes.clone(), &other).unwrap().read_exact_at(&mut [0u8; 4], 0).is_err());
        let mut tampered = (*bytes).clone();
        tampered[HEAD + 100] ^= 1;
        assert!(Sealed::open(Arc::new(tampered), &key).unwrap().read_exact_at(&mut [0u8; 4], 0).is_err());
        let _ = (std::fs::remove_file(p), std::fs::remove_file(s));
    }
}
