//! An optional in-memory tier in front of the streamed cache, with strict ownership.
//!
//! The first prediction against a fitted model streams it from its source as always. With a `Warm` attached,
//! the pieces that go past are kept (up to a byte budget), and later predictions use them in place: no read and
//! no copy. That removes most of a single prediction's time; the price is the memory the budget allows.
//!
//! Ownership rules, for running many tenants in one process:
//!
//! * every entry belongs to exactly one `(tenant, model)`; a handle can only ever see its own tenant's entries;
//! * each tenant has its own byte budget, on top of the overall one, so one tenant cannot crowd out the rest
//!   beyond its share (within the overall budget, the least recently used entry of any tenant goes first);
//! * entries are handed out as counted references: eviction removes the cache's reference, and the memory is
//!   freed (and, if asked, overwritten with zeros first) only when the last prediction using it lets go;
//! * `evict_tenant` and `evict_model` drop everything a tenant or model holds, immediately.

use std::collections::HashMap;
use std::io;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::sync::{Arc, Mutex};

/// Who an entry belongs to and which one it is: tenant, model, a name within the model, a number within the name.
pub(crate) type Key = (Arc<str>, Arc<str>, String, u64);

/// The bookkeeping of a tier with ownership, shared by this one and the graphics-memory tier
/// (`gpu::VramTier`): what is held for whom, how many bytes in all and per tenant, what was used last, and the
/// one rule for making room. What an entry *is* (a piece of cache here, a model on the card there) and what
/// letting go of it means are the tier's own business: evicted values are handed back to the caller.
pub(crate) struct Ledger<V> {
    entries: HashMap<Key, (V, usize, u64)>,
    bytes: usize,
    by_tenant: HashMap<Arc<str>, usize>,
    clock: u64,
}

impl<V> Default for Ledger<V> {
    fn default() -> Ledger<V> {
        Ledger { entries: HashMap::new(), bytes: 0, by_tenant: HashMap::new(), clock: 0 }
    }
}

impl<V> Ledger<V> {
    /// The entry, marked as just used.
    pub fn get(&mut self, key: &Key) -> Option<&mut V> {
        self.clock += 1;
        let now = self.clock;
        self.entries.get_mut(key).map(|e| {
            e.2 = now;
            &mut e.0
        })
    }

    pub fn remove(&mut self, key: &Key) -> Option<V> {
        let (v, n, _) = self.entries.remove(key)?;
        self.bytes -= n;
        if let Some(t) = self.by_tenant.get_mut(&key.0) {
            *t -= n;
        }
        Some(v)
    }

    /// Hold `value` (`bytes` large) under `key`, first making room within both budgets: while the tenant would
    /// be over its share its own least recently used entry goes, otherwise the least recently used of anyone's.
    /// Only entries `may_go` allows are ever removed. Returns what was removed; or, if room cannot be made (the
    /// entry alone is over the tenant's budget, or nothing more may go), gives `value` back with nothing held.
    pub fn admit(&mut self, key: Key, bytes: usize, value: V, budget: usize, tenant_budget: usize, may_go: impl Fn(&V) -> bool) -> Result<Vec<V>, (V, Vec<V>)> {
        let mut out = Vec::new();
        if bytes > tenant_budget.min(budget) {
            return Err((value, out));
        }
        loop {
            let mine = self.by_tenant.get(&key.0).copied().unwrap_or(0);
            let over_tenant = mine + bytes > tenant_budget;
            if !over_tenant && self.bytes + bytes <= budget {
                break;
            }
            let victim = self.entries.iter().filter(|(k, e)| (!over_tenant || k.0 == key.0) && may_go(&e.0)).min_by_key(|(_, e)| e.2).map(|(k, _)| k.clone());
            match victim {
                Some(k) => out.extend(self.remove(&k)),
                None => return Err((value, out)),
            }
        }
        self.clock += 1;
        self.bytes += bytes;
        *self.by_tenant.entry(key.0.clone()).or_insert(0) += bytes;
        out.extend(self.remove_quietly(&key));
        self.entries.insert(key, (value, bytes, self.clock));
        Ok(out)
    }

    /// Remove an entry that `admit` is about to replace (its bytes are given back first).
    fn remove_quietly(&mut self, key: &Key) -> Option<V> {
        self.entries.contains_key(key).then(|| self.remove(key)).flatten()
    }

    pub fn contains(&self, key: &Key) -> bool {
        self.entries.contains_key(key)
    }

    /// Remove everything `tenant` holds (for `model` only, if given).
    pub fn evict(&mut self, tenant: &str, model: Option<&str>) -> Vec<V> {
        let keys: Vec<Key> = self.entries.keys().filter(|k| &*k.0 == tenant && model.is_none_or(|m| &*k.1 == m)).cloned().collect();
        keys.iter().filter_map(|k| self.remove(k)).collect()
    }

    /// Every entry with its owner and size.
    #[cfg_attr(not(feature = "gpu"), allow(dead_code))]
    pub fn iter(&self) -> impl Iterator<Item = (&Key, &V, usize)> {
        self.entries.iter().map(|(k, e)| (k, &e.0, e.1))
    }

    /// Bytes held now, in all and by `tenant`.
    pub fn held(&self, tenant: &str) -> (usize, usize) {
        (self.bytes, self.by_tenant.get(tenant).copied().unwrap_or(0))
    }
}

/// A cached piece. Zeroes itself when the last reference goes, if the cache was created with `zero_on_free`.
pub struct Piece {
    pub words: Vec<u32>,
    zero: bool,
}

impl Drop for Piece {
    fn drop(&mut self) {
        if self.zero {
            self.words.fill(0);
        }
    }
}

/// The shared tier. Create one per process (or per pool) and hand out `WarmHandle`s.
pub struct Warm {
    budget: usize,
    tenant_budget: usize,
    zero_on_free: bool,
    state: Mutex<Ledger<Arc<Piece>>>,
    pub hits: AtomicU64,
    pub misses: AtomicU64,
    pub evictions: AtomicU64,
}

impl Warm {
    /// `budget`: bytes in all. `tenant_budget`: bytes any one tenant may hold. `zero_on_free`: overwrite evicted
    /// data with zeros before the memory is returned.
    pub fn new(budget: usize, tenant_budget: usize, zero_on_free: bool) -> Arc<Warm> {
        Arc::new(Warm { budget, tenant_budget: tenant_budget.min(budget), zero_on_free, state: Mutex::default(), hits: AtomicU64::new(0), misses: AtomicU64::new(0), evictions: AtomicU64::new(0) })
    }

    /// A handle through which one tenant's predictions against one fitted model use the tier.
    pub fn handle(self: &Arc<Warm>, tenant: &str, model: &str) -> WarmHandle {
        WarmHandle { warm: self.clone(), tenant: tenant.into(), model: model.into() }
    }

    /// Bytes held now, in all and by `tenant`.
    pub fn held(&self, tenant: &str) -> (usize, usize) {
        self.state.lock().unwrap().held(tenant)
    }

    /// Drop everything `tenant` holds (for `model` only, if given).
    pub fn evict(&self, tenant: &str, model: Option<&str>) {
        let gone = self.state.lock().unwrap().evict(tenant, model);
        self.evictions.fetch_add(gone.len() as u64, Relaxed);
    }
}

/// One tenant's view of the tier for one fitted model.
#[derive(Clone)]
pub struct WarmHandle {
    warm: Arc<Warm>,
    tenant: Arc<str>,
    model: Arc<str>,
}

impl std::fmt::Debug for WarmHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "warm tier for tenant {} model {}", self.tenant, self.model)
    }
}

impl WarmHandle {
    /// The piece named `(name, offset)`: from the tier if present, otherwise produced by `load` and kept if the
    /// budgets allow (evicting least recently used entries to make room).
    pub fn get(&self, name: &str, offset: u64, load: impl FnOnce() -> io::Result<Vec<u32>>) -> io::Result<Arc<Piece>> {
        let key: Key = (self.tenant.clone(), self.model.clone(), name.to_string(), offset);
        if let Some(p) = self.warm.state.lock().unwrap().get(&key) {
            self.warm.hits.fetch_add(1, Relaxed);
            return Ok(p.clone());
        }
        self.warm.misses.fetch_add(1, Relaxed);
        // load without holding the lock: other tenants are not kept waiting on this one's storage
        let piece = Arc::new(Piece { words: load()?, zero: self.warm.zero_on_free });
        let n = piece.words.len() * 4;
        let mut s = self.warm.state.lock().unwrap();
        if s.contains(&key) {
            return Ok(piece); // another thread of this tenant loaded it meanwhile
        }
        // an entry in use elsewhere may still go: its memory is freed when its last user lets go
        let gone = match s.admit(key, n, piece.clone(), self.warm.budget, self.warm.tenant_budget, |_| true) {
            Ok(gone) => gone,
            Err((_, gone)) => gone,
        };
        self.warm.evictions.fetch_add(gone.len() as u64, Relaxed);
        Ok(piece)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn budgets_and_isolation() {
        let warm = Warm::new(4000, 2400, true);
        let (a, b) = (warm.handle("alice", "m"), warm.handle("bob", "m"));
        let load = |v: u32| move || Ok(vec![v; 200]); // 800 bytes each
        for i in 0..3 {
            a.get("kv", i, load(1)).unwrap();
        }
        assert_eq!(warm.held("alice"), (2400, 2400));
        // a fourth piece pushes out alice's own oldest, not anyone else's
        a.get("kv", 3, load(1)).unwrap();
        assert_eq!(warm.held("alice").1, 2400);
        assert_eq!(warm.misses.load(Relaxed), 4);
        a.get("kv", 3, || panic!("must be a hit")).unwrap();
        // bob has the same names and sees none of alice's data
        let p = b.get("kv", 3, load(2)).unwrap();
        assert_eq!(p.words[0], 2);
        assert_eq!(warm.held("bob"), (3200, 800));
        // filling bob up respects the overall budget by evicting the least recently used
        b.get("kv", 4, load(2)).unwrap();
        b.get("kv", 5, load(2)).unwrap();
        assert!(warm.held("bob").0 <= 4000);
        // a piece in use survives eviction until released
        let held = a.get("kv", 3, load(1)).unwrap();
        warm.evict("alice", None);
        assert_eq!(warm.held("alice").1, 0);
        assert_eq!(held.words[0], 1);
    }

    fn key(t: &str, n: u64) -> Key {
        (t.into(), "m".into(), "kv".into(), n)
    }

    #[test]
    fn ledger_rules() {
        let mut l: Ledger<u32> = Ledger::default();
        // an entry bigger than a tenant's share is refused and handed back
        assert_eq!(l.admit(key("a", 0), 11, 7, 100, 10, |_| true), Err((7, vec![])));
        assert_eq!(l.admit(key("a", 1), 6, 1, 20, 10, |_| true), Ok(vec![]));
        // over the tenant's share: its own oldest goes
        assert_eq!(l.admit(key("b", 1), 6, 2, 20, 10, |_| true), Ok(vec![]));
        assert_eq!(l.admit(key("a", 2), 6, 3, 20, 10, |_| true), Ok(vec![1]));
        // over the overall budget: the least recently used of anyone's (b's, since a's 3 is newer)
        l.get(&key("a", 2));
        assert_eq!(l.admit(key("c", 1), 9, 4, 20, 10, |_| true), Ok(vec![2]));
        assert_eq!(l.held("a"), (15, 6));
        // nothing may go: refused, and the entries considered stay
        assert_eq!(l.admit(key("d", 1), 9, 5, 20, 10, |_| false), Err((5, vec![])));
        assert!(l.contains(&key("a", 2)) && l.contains(&key("c", 1)));
        // replacing an entry gives the old one back and counts its bytes once
        assert_eq!(l.admit(key("a", 2), 4, 6, 20, 10, |_| true), Ok(vec![3]));
        assert_eq!(l.held("a"), (13, 4));
        assert_eq!(*l.get(&key("a", 2)).unwrap(), 6);
        assert!(l.get(&key("a", 99)).is_none());
        assert_eq!(l.iter().map(|(_, _, n)| n).sum::<usize>(), 13);
        assert_eq!(l.evict("c", Some("other model")), Vec::<u32>::new());
        assert_eq!(l.evict("c", None), vec![4]);
        assert_eq!(l.remove(&key("a", 2)), Some(6));
        assert_eq!(l.remove(&key("a", 2)), None);
        assert_eq!(l.held("a"), (0, 0));
    }

    #[test]
    fn concurrent_loads_of_one_piece() {
        let warm = Warm::new(1 << 20, 1 << 20, false);
        let h = warm.handle("t", "m");
        assert!(format!("{h:?}").contains("tenant t"));
        std::thread::scope(|s| {
            for _ in 0..8 {
                let h = h.clone();
                s.spawn(move || assert_eq!(h.get("x", 0, || Ok(vec![5; 10])).unwrap().words, [5; 10]));
            }
        });
        assert_eq!(warm.held("t").1, 40, "held once however many threads loaded it");
        assert_eq!(warm.hits.load(Relaxed) + warm.misses.load(Relaxed), 8);
        // a load that fails is passed on and nothing is kept
        let e = h.get("y", 0, || Err(io::Error::other("storage down"))).err().unwrap();
        assert_eq!(e.to_string(), "storage down");
        assert_eq!(warm.held("t").1, 40);
    }
}
