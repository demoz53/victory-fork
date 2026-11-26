//! Emits `VICTORY_TRANSLATOR_FINGERPRINT` (`victory_shader::TRANSLATOR_FINGERPRINT`): a hash of
//! everything at build time that can change what this crate translates a
//! console shader to. `victory-gpu` keys its translated-shader cache on it
//! instead of on the whole executable, so two builds with the same
//! translator share one cache and a build with another translator never
//! reads its entries.
//!
//! What goes in, and why (when in doubt a thing is included: a key that
//! changes too often costs a translation pass; one that changes too seldom
//! serves wrong shaders):
//! - every file under `src/`, by its path and bytes: the translator, its
//!   tables and its tests (tests cannot change output, but telling them
//!   apart from code is a rule that could go wrong);
//! - this crate's `Cargo.toml` and this script;
//! - the workspace's `Cargo.toml` (profiles, dependency versions and
//!   features) and `Cargo.lock` (the exact version of every crate linked
//!   in: this crate's own dependencies are `thiserror` and `zlib-rs`);
//! - the compiler: `rust-toolchain.toml` and `rustc -vV` (version, commit,
//!   host, LLVM);
//! - how this crate is built: target, profile, optimisation level, debug
//!   assertions, panic strategy, target features, extra compiler flags, and
//!   the crate features that are on (`diagnostics`, `host`).
//!
//! What stays out: the other crates' sources. They reach a translation
//! only through the container's bytes and through settings, and
//! `victory-gpu`'s `shader_cache` puts both in each entry's key by value.

use std::path::{Path, PathBuf};
use std::{env, fs};

/// FNV-1a, 128 bits (std has no wide hash, and a build script should not
/// need a dependency): the inputs are our own files, so only accidental
/// collisions matter, and 128 bits leave none to expect.
struct Fnv(u128);

impl Fnv {
    const BASIS: u128 = 0x6c62_272e_07bb_0142_62b8_2175_6295_c58d;
    const PRIME: u128 = 0x0000_0000_0100_0000_0000_0000_0000_013b;

    fn bytes(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 ^= u128::from(b);
            self.0 = self.0.wrapping_mul(Self::PRIME);
        }
    }

    /// One named input. Name and value go in with their lengths, so no two
    /// different lists of inputs feed the hash the same bytes.
    fn part(&mut self, name: &str, value: &[u8]) {
        self.bytes(&(name.len() as u64).to_le_bytes());
        self.bytes(name.as_bytes());
        self.bytes(&(value.len() as u64).to_le_bytes());
        self.bytes(value);
    }
}

/// Every file under `dir`, at any depth.
fn files(dir: &Path, out: &mut Vec<PathBuf>) {
    let entries = fs::read_dir(dir).unwrap_or_else(|e| panic!("cannot list {}: {e}", dir.display()));
    for entry in entries {
        let path = entry
            .unwrap_or_else(|e| panic!("cannot list {}: {e}", dir.display()))
            .path();
        if path.is_dir() {
            files(&path, out);
        } else {
            out.push(path);
        }
    }
}

fn main() {
    let here = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").expect("cargo sets CARGO_MANIFEST_DIR"));
    let workspace = here.join("../..");
    let mut hash = Fnv(Fnv::BASIS);

    // The sources, in a fixed order, under names that do not depend on
    // where the worktree is.
    let mut sources = Vec::new();
    files(&here.join("src"), &mut sources);
    sources.sort();
    for path in &sources {
        let name = path
            .strip_prefix(&here)
            .expect("a file under the crate")
            .to_string_lossy()
            .replace('\\', "/");
        let bytes = fs::read(path).unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
        hash.part(&name, &bytes);
    }
    // (a folder: cargo looks at every file in it)
    println!("cargo:rerun-if-changed=src");

    // Manifests, the lock file and the toolchain file. One that is not
    // there counts as empty.
    for (name, path) in [
        ("Cargo.toml", here.join("Cargo.toml")),
        ("build.rs", here.join("build.rs")),
        ("workspace Cargo.toml", workspace.join("Cargo.toml")),
        ("Cargo.lock", workspace.join("Cargo.lock")),
        ("rust-toolchain.toml", workspace.join("rust-toolchain.toml")),
    ] {
        println!("cargo:rerun-if-changed={}", path.display());
        hash.part(name, &fs::read(&path).unwrap_or_default());
    }

    // The compiler that builds the translator.
    let rustc = env::var_os("RUSTC").unwrap_or_else(|| "rustc".into());
    let version = std::process::Command::new(&rustc)
        .arg("-vV")
        .output()
        .unwrap_or_else(|e| panic!("cannot run {} -vV: {e}", Path::new(&rustc).display()));
    assert!(version.status.success(), "rustc -vV failed");
    hash.part("rustc -vV", &version.stdout);

    // How this crate is built (cargo sets these for the build script).
    for name in [
        "TARGET",
        "PROFILE",
        "OPT_LEVEL",
        "DEBUG",
        "CARGO_CFG_DEBUG_ASSERTIONS",
        "CARGO_CFG_PANIC",
        "CARGO_CFG_TARGET_FEATURE",
        "CARGO_ENCODED_RUSTFLAGS",
    ] {
        let value = env::var_os(name);
        hash.part(name, value.as_deref().map_or(&b"<unset>"[..], |v| v.as_encoded_bytes()));
    }
    let mut features: Vec<String> = env::vars_os()
        .filter_map(|(name, _)| name.into_string().ok())
        .filter(|name| name.starts_with("CARGO_FEATURE_"))
        .collect();
    features.sort();
    hash.part("features", features.join(",").as_bytes());

    println!("cargo:rustc-env=VICTORY_TRANSLATOR_FINGERPRINT={:032x}", hash.0);
}
