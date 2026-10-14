# Security policy

## Reporting a vulnerability

Please report security problems privately, through GitHub's
[private vulnerability reporting](https://github.com/StevenKauwe/browsertabicl/security/advisories/new)
("Report a vulnerability" on the repository's Security tab), not in a public issue. Include what you did, what
happened, and the version or commit. You should hear back within a week. Fixes are released as soon as they are
ready and credited to you unless you prefer otherwise.

Only the latest release (and `main`) receives security fixes.

## What BrowserTabICL defends against, and what it does not

### Reading untrusted files

Weight files, fitted models (bundles) and PyTorch checkpoints are parsed by safe Rust code: the core crate
forbids `unsafe`. The checkpoint reader runs only the handful of pickle instructions PyTorch writes and never
executes code from a file. A damaged or hostile file is meant to produce an error; a crash, a hang, or an attempt
to allocate an absurd amount of memory is a bug worth reporting. These readers are tested with random damage
(`tests/formats.rs`).

Files carry no checksums: damage inside the numbers of an unsealed file is not detected and gives wrong answers.
Get weights from a source you trust.

### Sealed (encrypted) models

`browsertabicl seal` and `crypt::Sealed` encrypt a fitted model at rest so that it can still be streamed:

* The model is cut into blocks of 64 KiB, each encrypted and authenticated on its own with XChaCha20-Poly1305
  (a 256-bit key; a 192-bit nonce made of a random 128-bit file id and the block's number).
* Reading checks every block it touches. A wrong key, a changed byte, or a block moved to another position
  or taken from another sealed file is refused with an error; it never yields silently wrong numbers.
* The key is 32 random bytes from the operating system (`browsertabicl keygen`). Nothing derived from it is
  stored in the sealed file. `keygen` writes the key file readable by its owner only (on Unix) and never
  overwrites an existing one.

What it does **not** protect:

* **The length of the model** (stored in the clear in the header), and from it roughly the training table's size.
* **Truncation at a block boundary.** The header (file id and length) is not itself authenticated, so someone
  who can write the file can cut it at a 64 KiB boundary and change the stated length; the blocks that remain
  still open. In practice the model's index then points past the end and opening fails, but this is not a
  cryptographic guarantee. (Binding the header into every block is planned for a later, incompatible version of
  the sealed format.)
* **Which blocks are read.** Someone watching the storage sees which ranges a prediction fetches.
* **Memory.** Decrypted pieces are in memory while a prediction uses them. The warm tier can overwrite them
  with zeros when they are evicted (`zero_on_free`), but copies made elsewhere (the operating system, the
  graphics card) are not erased.
* **The weights** (they are public) and **the rows you predict** (they are not stored).
* **The key itself.** Keep the key file somewhere safe and separate from the sealed model: without it a sealed
  model cannot be read, by anyone.

### The web page

The page at https://stevenkauwe.github.io/browsertabicl/ runs entirely in the visitor's browser. Tables and
fitted models stay on the visitor's machine (browser storage); nothing is uploaded. It fetches only its own files
and, if configured to, the published weights from Hugging Face.

## Dependencies

Dependencies are checked weekly and on every change for known vulnerabilities, unmaintained or yanked crates and
licences (`cargo deny`, `cargo audit`; see `deny.toml` and `.github/workflows/audit.yml`).
