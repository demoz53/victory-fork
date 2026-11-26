//! A fast hasher for the maps the recording threads look into on every
//! bind (`VICTORY_FAST_HASH=1`; off by default: the standard SipHash).
//!
//! The profiler put 3.5-3.8% of each recording thread's running time in
//! `core::hash::sip` (open world): the maps keyed by integers, pointers and
//! small tuples that a draw's binds look into (a list's refreshed ranges,
//! textures and writes, its tables made afresh and root view slots; the
//! texture staging maps; a heap's slot spans). SipHash keeps a map safe
//! from keys chosen by an attacker, which these are not.
//!
//! With the switch those maps hash with a multiply-rotate hash in the
//! manner of FxHash (`rustc-hash`'s second form: add, multiply by an odd
//! constant, rotate at the end so that the low bits the table indexes by
//! depend on all of the key; written here, no crate added). The choice is
//! made at start-up and costs one branch per hashed word
//! ([`tests::hashing_costs`] measures both).

use std::collections::hash_map::{DefaultHasher, RandomState};
use std::hash::{BuildHasher, Hasher};

/// `HashMap` with the hasher [`on`] picks.
pub(crate) type Map<K, V> = std::collections::HashMap<K, V, Pick>;
/// `HashSet` with the hasher [`on`] picks.
pub(crate) type Set<T> = std::collections::HashSet<T, Pick>;

/// Whether the maps of this module hash fast (`VICTORY_FAST_HASH=1`).
pub(crate) fn on() -> bool {
    #[cfg(test)]
    {
        let forced = TEST_FAST.with(std::cell::Cell::get);
        if forced != 0 {
            return forced == 2;
        }
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        let on = std::env::var_os("VICTORY_FAST_HASH").is_some_and(|v| v == "1");
        if on {
            tracing::info!("VICTORY_FAST_HASH=1: the recording path's maps hash with a multiply-rotate hash, not SipHash");
        }
        on
    })
}

#[cfg(test)]
thread_local! {
    /// Tests, for maps made on the calling thread: 0 as the switch says,
    /// 1 SipHash, 2 fast.
    pub(crate) static TEST_FAST: std::cell::Cell<u8> = const { std::cell::Cell::new(0) };
}

/// The hasher of a map, picked when the map is made.
#[derive(Clone)]
pub(crate) enum Pick {
    Sip(RandomState),
    Fast,
}

impl Default for Pick {
    fn default() -> Self {
        if on() { Pick::Fast } else { Pick::Sip(RandomState::new()) }
    }
}

impl BuildHasher for Pick {
    type Hasher = Picked;

    #[inline]
    fn build_hasher(&self) -> Picked {
        match self {
            Pick::Sip(state) => Picked::Sip(state.build_hasher()),
            Pick::Fast => Picked::Fast(0),
        }
    }
}

/// A hash being made.
pub(crate) enum Picked {
    Sip(DefaultHasher),
    Fast(u64),
}

/// The fast hash's multiplier (odd; that of `rustc-hash` 2).
const K: u64 = 0xf135_7aea_2e62_a9c5;

#[inline]
fn add(hash: &mut u64, word: u64) {
    *hash = hash.wrapping_add(word).wrapping_mul(K);
}

impl Hasher for Picked {
    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        match self {
            Picked::Sip(h) => h.write(bytes),
            Picked::Fast(h) => {
                let mut chunks = bytes.chunks_exact(8);
                for chunk in &mut chunks {
                    add(h, u64::from_le_bytes(chunk.try_into().unwrap_or_default()));
                }
                let rest = chunks.remainder();
                if !rest.is_empty() {
                    let mut last = [0u8; 8];
                    last[..rest.len()].copy_from_slice(rest);
                    add(h, u64::from_le_bytes(last) ^ (rest.len() as u64) << 59);
                }
            }
        }
    }

    #[inline]
    fn write_u8(&mut self, n: u8) {
        self.write_u64(u64::from(n));
    }

    #[inline]
    fn write_u16(&mut self, n: u16) {
        self.write_u64(u64::from(n));
    }

    #[inline]
    fn write_u32(&mut self, n: u32) {
        match self {
            Picked::Sip(h) => h.write_u32(n),
            Picked::Fast(h) => add(h, u64::from(n)),
        }
    }

    #[inline]
    fn write_u64(&mut self, n: u64) {
        match self {
            Picked::Sip(h) => h.write_u64(n),
            Picked::Fast(h) => add(h, n),
        }
    }

    #[inline]
    fn write_usize(&mut self, n: usize) {
        match self {
            Picked::Sip(h) => h.write_usize(n),
            Picked::Fast(h) => add(h, n as u64),
        }
    }

    #[inline]
    fn finish(&self) -> u64 {
        match self {
            Picked::Sip(h) => h.finish(),
            // The low bits index the table: the rotation brings the
            // product's best-mixed high bits there (pointers' low bits are
            // zero).
            Picked::Fast(h) => h.rotate_left(26),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Either hasher: a map finds what it holds, and the fast one spreads
    /// aligned pointers over the low bits.
    #[test]
    fn both_hashers_find_their_keys() {
        for forced in [1, 2] {
            TEST_FAST.set(forced);
            let mut map: Map<(usize, usize, usize), u32> = Map::default();
            let mut set: Set<[u32; 14]> = Set::default();
            for i in 0..10_000usize {
                map.insert((0x2_0000_0000 + i * 256, i, 7), i as u32);
                let mut key = [0u32; 14];
                key[i % 14] = i as u32;
                set.insert(key);
            }
            for i in 0..10_000usize {
                assert_eq!(map.get(&(0x2_0000_0000 + i * 256, i, 7)), Some(&(i as u32)));
            }
            assert!(set.len() > 9_000);
            TEST_FAST.set(0);
        }
        TEST_FAST.set(2);
        let pick = Pick::default();
        let low: std::collections::BTreeSet<u64> = (0..4096u64)
            .map(|i| {
                let mut h = pick.build_hasher();
                h.write_usize(0x7ff0_0000_0000 + (i as usize) * 4096);
                h.finish() & 0xfff
            })
            .collect();
        assert!(low.len() > 2_000, "page-aligned pointers fall in {} of 4096 buckets", low.len());
        TEST_FAST.set(0);
    }

    /// The micro-measurement behind the switch: inserts and lookups of the
    /// recording path's commonest key shape, SipHash against fast
    /// (printed; `--nocapture`).
    #[test]
    #[ignore = "a measurement: cargo test -p victory-gpu --profile diag --lib fasthash -- --ignored --nocapture"]
    fn hashing_costs() {
        const N: usize = 200_000;
        let mut line = Vec::new();
        for (forced, name) in [(1, "SipHash"), (2, "fast")] {
            TEST_FAST.set(forced);
            let mut set: Set<(usize, usize, usize)> = Set::default();
            set.reserve(N);
            let t = std::time::Instant::now();
            for i in 0..N {
                set.insert((0x2_0000_0000 + i * 64, i * 16, 256));
            }
            let mut found = 0;
            for _ in 0..4 {
                for i in 0..N {
                    found += usize::from(set.contains(&(0x2_0000_0000 + i * 64, i * 16, 256)));
                }
            }
            let ns = t.elapsed().as_nanos() as f64 / (5 * N) as f64;
            line.push(format!("{name} {ns:.1} ns ({found})"));
        }
        TEST_FAST.set(0);
        println!("an insert or lookup of a (usize, usize, usize) key: {}", line.join(", "));
    }
}
