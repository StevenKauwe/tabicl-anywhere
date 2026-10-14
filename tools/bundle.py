"""Tensor bundle: the one file format tabicl-rs uses for weights, caches and test vectors.

  bytes 0..8    magic "TIRSBNDL"
  bytes 8..16   u64 little-endian: length of the index in bytes
  index         utf-8 text, one line per tensor:  name dtype offset nbytes dim0 dim1 ...
  data          each tensor starts on a 64-byte boundary (offset counts from the start of the file)
"""
import numpy as np, struct

def write(path, tensors):
    items = [(k, np.ascontiguousarray(v)) for k, v in tensors.items()]
    # the index length depends on the offsets, so lay out with a generous fixed-size index
    lines = lambda offs: "".join(f"{k} {v.dtype.name} {o} {v.nbytes} {' '.join(map(str, v.shape))}\n" for (k, v), o in zip(items, offs))
    guess = len(lines([10**15] * len(items)).encode()) + 16
    start = (guess + 63) // 64 * 64
    offs, pos = [], start
    for _, v in items:
        offs.append(pos)
        pos = (pos + v.nbytes + 63) // 64 * 64
    index = lines(offs).encode()
    with open(path, "wb") as f:
        f.write(b"TIRSBNDL" + struct.pack("<Q", len(index)) + index)
        for (_, v), o in zip(items, offs):
            f.seek(o)
            f.write(v.astype(v.dtype.newbyteorder("<"), copy=False).tobytes())
        f.truncate(pos)
