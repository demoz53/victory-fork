//! Translated shaders kept on disk (decision 0003): under the configured
//! cache folder, keyed by the container's content and by what decides its
//! translation, so a changed translator never serves an old translation.
//! Entries go through `victory-pipeline`'s content-addressed cache, which
//! writes them atomically; an entry that does not read back is translated
//! again.
//!
//! # The key
//!
//! Every entry's key holds, by value:
//! - the **translator**: `victory_shader::TRANSLATOR_FINGERPRINT`, a
//!   build-time hash of the `victory-shader` crate's sources, manifests,
//!   lock file, compiler and build settings (its `build.rs` lists them),
//!   and the BLAKE3 of this file, which holds the entries' encoding and the
//!   rest of the key. Until 2026-11-25 this was the BLAKE3 of the whole
//!   executable, so every build translated every shader again, whatever it
//!   had changed. `VICTORY_SHADER_CACHE_KEY=exe` brings that back;
//! - the **settings** the translator reads from this process: the loop
//!   limit, the view heap's limit and map, the group barriers' mode, the
//!   clean-up passes, the ray-tracing fallback, and the value of every
//!   environment switch in [`SWITCHES`];
//! - the **program's own decisions**: the host's wave width, the
//!   deep-skips and wide-first choices, the group base, and clean-up passes
//!   given to it alone ([`opt_key`]);
//! - the **container's bytes**.
//!
//! Nothing else reaches `victory_shader::emit::translate`: the other
//! crates hand it those bytes and those settings and no more. A new
//! setting or switch that changes translated code MUST be added here
//! (a test fails when the translator reads a switch this file does not
//! account for).
//!
//! Each run key (translator and settings) has its own folder under the
//! cache folder, `k-<16 hex digits>`, with a `key.txt` saying what it is
//! made of: `victory shader-cache --list` shows them, so the folders of
//! translators no longer built can be told apart and removed.
//!
//! The entry is our own encoding of [`HostShader`]: a field added to it
//! must be added to `encode` and `decode` too (this file's hash is in the
//! key, so entries written before the change are not read).

use std::path::PathBuf;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use victory_pipeline::{Cache, Converter};
use victory_shader::emit::{HostShader, StaticSampler, UserData};
use victory_shader::resource::RootIndexedView;
use victory_shader::{HwStage, RootType};

/// The folder translated shaders are kept under ([`use_shader_cache`]).
static ROOT: OnceLock<PathBuf> = OnceLock::new();
/// This run's cache: its key's folder under [`ROOT`], opened at the first
/// translation (the key needs the host's view heap limit).
static CACHE: OnceLock<Option<Cache>> = OnceLock::new();
static HITS: AtomicU64 = AtomicU64::new(0);
static MISSES: AtomicU64 = AtomicU64::new(0);

/// Keeps translated shaders under `dir` from now on.
pub fn use_shader_cache(dir: PathBuf) {
    let _ = ROOT.set(dir);
}

fn cache() -> Option<&'static Cache> {
    CACHE
        .get_or_init(|| {
            let root = ROOT.get()?;
            let (key, made_of) = build();
            let dir = root.join(key_folder(key));
            match Cache::open(dir.clone()) {
                Ok(cache) => {
                    // What the folder's key is made of, for whoever tidies
                    // the cache folder (`victory shader-cache --list`).
                    let note = dir.join(KEY_NOTE);
                    if !note.exists() {
                        let _ = std::fs::write(&note, made_of);
                    }
                    Some(cache)
                }
                Err(e) => {
                    tracing::warn!(error = %e, "no shader cache");
                    None
                }
            }
        })
        .as_ref()
}

/// The file in a key's folder that says what the key is made of.
pub const KEY_NOTE: &str = "key.txt";

/// The name of a run key's folder under the cache folder.
fn key_folder(key: &[u8; 32]) -> String {
    let hex: String = key[..8].iter().map(|b| format!("{b:02x}")).collect();
    format!("k-{hex}")
}

/// This run's folder name under the cache folder and what its key is made
/// of (`victory shader-cache`). Probes the host for the view heap limit, as
/// the first translation does.
pub fn run_key() -> (String, String) {
    let (key, made_of) = build();
    (key_folder(key), made_of.clone())
}

/// The translation of `container` through the cache, as a run makes it
/// (`victory shader-cache --warm`).
pub fn translate_cached(container: &[u8]) -> Result<HostShader, String> {
    host_shader(container)
}

/// Shaders found in the cache and translated since the start
/// (`victory shader-cache --warm`).
pub fn hits_and_misses() -> (u64, u64) {
    counts()
}

/// Shaders found in the cache and translated since the start.
pub(crate) fn counts() -> (u64, u64) {
    (HITS.load(Ordering::Relaxed), MISSES.load(Ordering::Relaxed))
}

/// The environment switches the translator reads that change translated
/// code for every program (the per-program ones are in each program's own
/// key, [`program_key`] and [`wide_first_key`]): name and value of each go
/// into the run's key. Without one here, a run with the switch set would
/// be served what a run without it had cached. A switch added to the
/// translator goes here (a test below reads the translator's sources and
/// fails when one is missing).
///
/// The order, and `VICTORY_WAVE_LOCAL` being here twice, are kept as they
/// were: with `VICTORY_SHADER_CACHE_KEY=exe` the key is, byte for byte,
/// what it was while the executable was the identity.
const SWITCHES: [&str; 52] = [
    "VICTORY_WAVE_WIDE",
    "VICTORY_WAVE_DEEP_SKIPS",
    "VICTORY_WAVE_LDS_REGIONS",
    "VICTORY_WAVE_LOCAL",
    "VICTORY_LOCAL_LDS_SYNC",
    "VICTORY_WAVE_OWN_BITS",
    "VICTORY_F16_NEAREST",
    // (read by `pipeline.rs`: which programs get which clean-up passes; the
    // passes a program gets are in its own key too, [`opt_key`])
    "VICTORY_SHADER_OPT_RANGE",
    "VICTORY_SHADER_OPT_IN",
    "VICTORY_SHADER_OPT_STAGE",
    "VICTORY_MESH_EDGE_FLAGS",
    "VICTORY_FIRSTLANE_OWN",
    "VICTORY_STRICT_JOINS",
    "VICTORY_MESH_OUTPUT_STRICT",
    "VICTORY_WAVE_LOCAL",
    "VICTORY_FIRST_HALF",
    "VICTORY_LANE_LOOP_FIX",
    "VICTORY_LOOP_EXITS",
    // (test/loops2: the loops a lane branch leaves are its region)
    "VICTORY_AFTER_LOOP",
    // (test/loops2: the lane fixes, each on unless "0")
    "VICTORY_PERMUTE_HALVES",
    "VICTORY_SHORT_WAVE_LAST",
    "VICTORY_LANE_MODULO",
    "VICTORY_THREAD_TICKET",
    "VICTORY_VCC_LOOP_POLARITY",
    "VICTORY_SKIP_DESCRIPTORS",
    "VICTORY_DIV_FIXUP",
    "VICTORY_WAVE_EXACT",
    "VICTORY_WAVE_WIDE_LANE_RESOURCES",
    "VICTORY_EARLY_DEPTH",
    "VICTORY_PIXEL_HELPERS",
    "VICTORY_LDS_FIT",
    "VICTORY_HALF_BITS",
    "VICTORY_HEAP_HALF_BITS",
    "VICTORY_SWIZZLE_MODES",
    "VICTORY_NORM_EXACT",
    "VICTORY_NAN_GUARD",
    "VICTORY_CUBE_GRAD_LOD",
    "VICTORY_LOCAL_ANY_MASKS",
    "VICTORY_HEAP_HANDLE_LOCAL",
    "VICTORY_HEAP_BELOW_ZERO",
    // Which programs run a console thread group as several host groups, and
    // so what their dispatches' z is multiplied by (test/split).
    "VICTORY_GROUP_SPLIT",
    // Constant buffers through tables read as raw buffers (test/tables).
    "VICTORY_TABLE_CONSTANTS_IN_PLACE",
    // Mesh shaders declare a payload under it (`crate::launch`; test/meshclamp).
    "VICTORY_MESH_CLAMP_IN_DRAW",
    // (test/small3: SDWA WORD_0 UNUSED_SEXT sign-extends; on unless "0")
    "VICTORY_SDWA_SEXT",
    // (test/shader3: NaN told by its bits in float compares, Corgansm/victory-corgi 8b6671cc2)
    "VICTORY_NAN_EXACT",
    // (test/shader3: the tiled cascade shadow programs' exact form, fxceless PR #1002)
    "VICTORY_CASCADES_EXACT",
    // (test/shader4b: wave64 fixes; OWN_BITS_SCALAR, OWN_BITS_LATCH and MASK_PAIR on unless "0",
    // LDS_ATOMICS_READ and SELECT_LANES opt-in)
    "VICTORY_OWN_BITS_SCALAR",
    "VICTORY_OWN_BITS_LATCH",
    "VICTORY_LDS_ATOMICS_READ",
    "VICTORY_MASK_PAIR",
    "VICTORY_SELECT_LANES",
    // (test/shader5: a comparison sampler's reduction replaced, not ORed into;
    // on unless "0": the static samplers an entry holds)
    "VICTORY_COMPARE_REDUCTION",
];

/// The translation settings of this process that are in the run's key by
/// value (each is read by the translator from a global it or `victory-gpu`
/// sets once, before the first translation).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Settings {
    /// `VICTORY_SHADER_LOOP_LIMIT` (`emit::loop_limit`).
    loop_limit: u32,
    /// The view heap's null slot, which differs by host GPU
    /// (`emit::view_heap_limit`): a translation cached without the clamp
    /// must not be served.
    view_limit: u32,
    /// Which group barriers wait (`VICTORY_GROUP_SYNC`, `emit::group_sync`).
    group_sync: u8,
    /// The run's clean-up passes (`VICTORY_SHADER_OPT`,
    /// `emit::run_shader_opt`).
    shader_opt: u32,
    /// The heap map bindless views are looked up in (`VICTORY_HEAP_MAP`,
    /// `VICTORY_HEAP_WINDOW`, `emit::view_heap_map`); `u32::MAX` twice
    /// without one.
    heap_map: [u32; 2],
    /// Whether ray-tracing node tests all miss (`VICTORY_RT_MISS` and
    /// `VICTORY_BVH_BUILDER`, `emit::rt_miss`).
    rt_miss: bool,
}

impl Settings {
    /// This process's. Probes the host for its view heap limit first.
    fn now() -> Self {
        use victory_shader::emit;
        crate::descriptor::host::probe_view_limit();
        Self {
            loop_limit: emit::loop_limit(),
            view_limit: emit::view_heap_limit(),
            group_sync: emit::group_sync(),
            shader_opt: emit::run_shader_opt(),
            heap_map: emit::view_heap_map().map_or([u32::MAX; 2], |(w, s)| [w, s]),
            rt_miss: emit::rt_miss(),
        }
    }
}

/// The settings and the switches' values (`env` gives a switch's) as the
/// bytes that go into the run's key after the build's identity. Pure.
fn settings_bytes(s: &Settings, env: &dyn Fn(&str) -> Option<std::ffi::OsString>) -> Vec<u8> {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&s.loop_limit.to_le_bytes());
    bytes.extend_from_slice(&s.view_limit.to_le_bytes());
    bytes.push(s.group_sync);
    for name in SWITCHES {
        bytes.extend_from_slice(name.as_bytes());
        bytes.push(b'=');
        if let Some(value) = env(name) {
            bytes.extend_from_slice(value.as_encoded_bytes());
        }
        bytes.push(0);
    }
    bytes.extend_from_slice(&s.shader_opt.to_le_bytes());
    bytes.extend_from_slice(&s.view_limit.to_le_bytes());
    for word in s.heap_map {
        bytes.extend_from_slice(&word.to_le_bytes());
    }
    bytes.push(u8::from(s.rt_miss));
    bytes
}

/// What a build is known by in the key (`VICTORY_SHADER_CACHE_KEY`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Identity {
    /// The translator's fingerprint and this file's hash: every build with
    /// the same translator shares its entries. The default.
    Translator,
    /// The BLAKE3 of the whole executable, as until 2026-11-25
    /// (`VICTORY_SHADER_CACHE_KEY=exe`): no build shares another's entries.
    Exe,
}

impl Identity {
    /// From `VICTORY_SHADER_CACHE_KEY`'s value: `exe` and nothing else asks
    /// for the executable.
    fn from_switch(value: Option<&std::ffi::OsStr>) -> Self {
        if value.is_some_and(|v| v == "exe") {
            Self::Exe
        } else {
            Self::Translator
        }
    }
}

/// The translator's identity: what `victory-shader`'s build script hashed
/// (`victory_shader::TRANSLATOR_FINGERPRINT`) and the BLAKE3 of this file
/// (the entries' encoding, and what else goes into a key).
fn translator_identity(fingerprint: &str) -> Vec<u8> {
    let mut id = b"victory translator ".to_vec();
    id.extend_from_slice(fingerprint.as_bytes());
    id.extend_from_slice(b" entries ");
    id.extend_from_slice(blake3::hash(include_bytes!("shader_cache.rs")).as_bytes());
    id
}

/// A run's key: the build's identity, then its settings. With the
/// executable's bytes as the identity this is the key as it always was.
fn run_key_of(identity: &[u8], settings: &[u8]) -> [u8; 32] {
    let mut h = blake3::Hasher::new();
    h.update(identity);
    h.update(settings);
    *h.finalize().as_bytes()
}

/// The run's key and what it is made of (a few lines of text, kept in the
/// key's folder): the build's identity ([`Identity`]) and the translation
/// settings this process runs with ([`Settings`], [`SWITCHES`]).
fn build() -> &'static ([u8; 32], String) {
    static BUILD: OnceLock<([u8; 32], String)> = OnceLock::new();
    BUILD.get_or_init(|| {
        let settings = Settings::now();
        if settings.loop_limit != victory_shader::emit::DEFAULT_LOOP_LIMIT {
            tracing::warn!(
                limit = settings.loop_limit,
                "VICTORY_SHADER_LOOP_LIMIT: translated loops stop sooner"
            );
        }
        if settings.rt_miss {
            tracing::warn!(
                "VICTORY_RT_MISS=1: every ray-tracing node test misses, so every traced ray sees the sky"
            );
        } else if std::env::var_os("VICTORY_RT_MISS").is_some_and(|v| v == "1") {
            tracing::warn!("VICTORY_RT_MISS=1 is ignored: VICTORY_BVH_BUILDER=1 asks for the structure builder");
        }
        let bytes = settings_bytes(&settings, &|name| std::env::var_os(name));
        let set: Vec<String> = SWITCHES
            .iter()
            .filter_map(|&name| Some(format!("{name}={}", std::env::var_os(name)?.to_string_lossy())))
            .collect();
        let mode = Identity::from_switch(std::env::var_os("VICTORY_SHADER_CACHE_KEY").as_deref());
        let (identity, known_by) = match mode {
            Identity::Translator => (
                Ok(translator_identity(victory_shader::TRANSLATOR_FINGERPRINT)),
                format!(
                    "translator {} (victory-shader's build fingerprint), entries {}",
                    victory_shader::TRANSLATOR_FINGERPRINT,
                    &blake3::hash(include_bytes!("shader_cache.rs")).to_hex()[..16]
                ),
            ),
            Identity::Exe => {
                let exe = std::env::current_exe().and_then(std::fs::read);
                let known_by = exe.as_ref().map_or_else(
                    |e| format!("executable (unreadable: {e})"),
                    |bytes| format!("executable {} (VICTORY_SHADER_CACHE_KEY=exe)", &blake3::hash(bytes).to_hex()[..16]),
                );
                (exe, known_by)
            }
        };
        let made_of = format!(
            "identity: {known_by}\nsettings: {settings:?}\nswitches set: {}\nfirst used by: {}\n",
            if set.is_empty() { "none".to_owned() } else { set.join(" ") },
            std::env::current_exe().map_or_else(|_| "?".to_owned(), |p| p.display().to_string()),
        );
        match identity {
            Ok(identity) => {
                let key = run_key_of(&identity, &bytes);
                tracing::info!(folder = %key_folder(&key), identity = %known_by, "shader cache key");
                (key, made_of)
            }
            Err(e) => {
                // Without it no entry can be trusted: a fresh key per run.
                tracing::warn!(error = %e, "cannot read our executable; shader cache unused");
                let now = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default();
                (*blake3::hash(&now.as_nanos().to_le_bytes()).as_bytes(), made_of)
            }
        }
    })
}

/// The translation of a console shader container, from the cache when
/// there is one.
pub(crate) fn host_shader(container: &[u8]) -> Result<HostShader, String> {
    host_shader_from_group(container, 0)
}

/// [`host_shader`] with `base` added to the thread group's id x
/// (`victory_shader::emit::with_group_base`; 0: as it is): the program of
/// the second parts of indirect dispatches over 65,535 groups in x
/// (`VICTORY_INDIRECT_SPLIT`). The base is in the entry's key.
pub(crate) fn host_shader_from_group(container: &[u8], base: u32) -> Result<HostShader, String> {
    use victory_shader::emit;
    // The hash the log names the shader by (`pipeline.rs`), which the
    // per-program switches take.
    let hash = blake3::hash(container).to_hex()[..16].to_owned();
    let translate = |c: &[u8]| {
        emit::for_program(&hash, || emit::with_group_base(base, || emit::translate(c))).map_err(|e| e.to_string())
    };
    match cache() {
        Some(cache) => {
            // (`wave64_retried` second: it parses the container, and only a
            // program a list names, or with `VICTORY_WAVE_DEEP_SKIPS=0`
            // any, gets there)
            let off = !emit::deep_skips_allowed(Some(&hash)) && emit::wave64_retried(container);
            let wide_first = emit::wave_wide_first(Some(&hash)) && emit::wave64_retried(container);
            // (`VICTORY_WAVE_PAIRS` and its lists: the answer for this program, when it is
            // one the paired form can matter for)
            let pairs = match emit::wave_pairs_for(Some(&hash)) {
                emit::WavePairs::Off => emit::WavePairs::Off,
                mode if emit::wave64_retried(container) => mode,
                _ => emit::WavePairs::Off,
            };
            let key = pairs_key(&group_key(&wide_first_key(&program_key(&build().0, off), wide_first), base), pairs);
            // (inside `with_shader_opt`, when `pipeline.rs` gives this program its own passes)
            let key = opt_key(&key, emit::shader_opt(), emit::run_shader_opt());
            cached(cache, &key, emit::host_uses_wave32(), container, translate)
        }
        None => translate(container),
    }
}

/// `key` mixed with the program's answer to `VICTORY_WAVE_PAIRS`
/// (`victory_shader::emit::wave_pairs_for`) when the paired form is tried for
/// it; as it is otherwise, so the switch and its lists are in no key.
fn pairs_key(key: &[u8; 32], pairs: victory_shader::emit::WavePairs) -> [u8; 32] {
    let how: &[u8] = match pairs {
        victory_shader::emit::WavePairs::Off => return *key,
        victory_shader::emit::WavePairs::BeforeWide => b"VICTORY_WAVE_PAIRS=1 for this program",
        victory_shader::emit::WavePairs::First => b"VICTORY_WAVE_PAIRS=first for this program",
    };
    let mut h = blake3::Hasher::new();
    h.update(key);
    h.update(how);
    *h.finalize().as_bytes()
}

/// `key` mixed with the per-program decision of `VICTORY_WAVE_WIDE_FIRST`
/// (`victory_shader::emit::wave_wide_first`) when the program is listed; as
/// it is otherwise, so the list is in no key.
fn wide_first_key(key: &[u8; 32], listed: bool) -> [u8; 32] {
    if !listed {
        return *key;
    }
    let mut h = blake3::Hasher::new();
    h.update(key);
    h.update(b"VICTORY_WAVE_WIDE_FIRST lists this program");
    *h.finalize().as_bytes()
}

/// `key` mixed with the clean-up passes this translation runs with
/// (`passes`, `victory_shader::emit::shader_opt`) when they are not the
/// run's (`run`, which is in the run's key): `pipeline.rs` gives a program
/// its own with `VICTORY_SHADER_OPT_RANGE`, `_IN` and `_STAGE`. Those
/// switches' values are in the run's key, but which programs they pick is
/// decided outside the translator, so what was picked is in the key too.
fn opt_key(key: &[u8; 32], passes: u32, run: u32) -> [u8; 32] {
    if passes == run {
        return *key;
    }
    let mut h = blake3::Hasher::new();
    h.update(key);
    h.update(b"clean-up passes for this program");
    h.update(&passes.to_le_bytes());
    *h.finalize().as_bytes()
}

/// `key` mixed with a group base other than 0 ([`host_shader_from_group`]);
/// with 0, `key` as it is.
fn group_key(key: &[u8; 32], base: u32) -> [u8; 32] {
    if base == 0 {
        return *key;
    }
    let mut h = blake3::Hasher::new();
    h.update(key);
    h.update(b"group id x from");
    h.update(&base.to_le_bytes());
    *h.finalize().as_bytes()
}

/// The key of one program's entries: the build's, mixed with the
/// per-program decision when the program is kept out of the wave64 form
/// "without skips, scalars followed" (`VICTORY_WAVE_DEEP_SKIPS_NOT`,
/// `_ONLY`: `victory_shader::emit::deep_skips_allowed`). Only for a wave64
/// compute program on a host of 32-lane waves, the only ones it can change.
/// So the lists are in no key: changing one translates again only the
/// programs whose decision it changes, and a program allowed the form keeps
/// the key it had without any list.
fn program_key(build: &[u8; 32], deep_skips_off: bool) -> [u8; 32] {
    if !deep_skips_off {
        return *build;
    }
    let mut h = blake3::Hasher::new();
    h.update(build);
    h.update(b"VICTORY_WAVE_DEEP_SKIPS off for this program");
    *h.finalize().as_bytes()
}

/// The cache entry's converter: the build, host wave mode and container in, the
/// encoded translation out. `ran` tells a miss from a hit.
struct Translate<F> {
    translate: F,
    ran: AtomicBool,
}

impl<F: Fn(&[u8]) -> Result<HostShader, String> + Send + Sync> Converter for Translate<F> {
    fn id(&self) -> &'static str {
        "shader.dxil"
    }
    fn version(&self) -> u32 {
        // Version 1 did not distinguish translations made for 32-lane hosts.
        2
    }
    fn convert(&self, input: &[u8]) -> anyhow::Result<Vec<u8>> {
        self.ran.store(true, Ordering::Relaxed);
        let shader = (self.translate)(&input[33..]).map_err(anyhow::Error::msg)?;
        Ok(encode(&shader))
    }
}

fn cached(
    cache: &Cache,
    build: &[u8; 32],
    wave32: bool,
    container: &[u8],
    translate: impl Fn(&[u8]) -> Result<HostShader, String> + Send + Sync,
) -> Result<HostShader, String> {
    let input = [&build[..], &[u8::from(wave32)], container].concat();
    let converter = Translate {
        translate,
        ran: AtomicBool::new(false),
    };
    for _ in 0..2 {
        let path = cache
            .get_or_convert(&converter, &input)
            .map_err(|e| format!("{e:#}"))?;
        if let Some(shader) = std::fs::read(&path).ok().and_then(|b| decode(&b)) {
            let (counter, what) = if converter.ran.load(Ordering::Relaxed) {
                (&MISSES, "translated and cached")
            } else {
                (&HITS, "from the cache")
            };
            counter.fetch_add(1, Ordering::Relaxed);
            tracing::debug!(path = %path.display(), "shader {what}");
            return Ok(shader);
        }
        tracing::debug!(path = %path.display(), "unreadable shader cache entry; translating again");
        let _ = std::fs::remove_file(&path);
    }
    MISSES.fetch_add(1, Ordering::Relaxed);
    (converter.translate)(container)
}

const MAGIC: &[u8; 4] = b"VSH1";
const STAGES: [HwStage; 8] = [
    HwStage::Ls,
    HwStage::Hs,
    HwStage::Es,
    HwStage::Gs,
    HwStage::Vs,
    HwStage::Ps,
    HwStage::Ms,
    HwStage::Cs,
];

fn root_code(r: RootType) -> u32 {
    match r {
        RootType::Constants => 0,
        RootType::Descriptor => 1,
        RootType::Table => 2,
        RootType::ImmediateSampler => 3,
        RootType::ResourceHeap => 4,
        RootType::SamplerHeap => 5,
        RootType::Other(x) => 6 + x,
    }
}

fn root_type(code: u32) -> RootType {
    match code {
        0 => RootType::Constants,
        1 => RootType::Descriptor,
        2 => RootType::Table,
        3 => RootType::ImmediateSampler,
        4 => RootType::ResourceHeap,
        5 => RootType::SamplerHeap,
        x => RootType::Other(x - 6),
    }
}

/// Little-endian words and length-prefixed lists.
fn encode(s: &HostShader) -> Vec<u8> {
    let mut out = MAGIC.to_vec();
    let word = |out: &mut Vec<u8>, v: u32| out.extend(v.to_le_bytes());
    let stage = STAGES.iter().position(|&x| x == s.stage).unwrap_or(0) as u32;
    word(&mut out, stage);
    word(&mut out, s.dxil.len() as u32);
    out.extend(&s.dxil);
    word(&mut out, s.user_data.len() as u32);
    for u in &s.user_data {
        for v in [u.index, u32::from(u.slot), root_code(u.root), u.dword] {
            word(&mut out, v);
        }
    }
    word(&mut out, s.samplers.len() as u32);
    for sampler in &s.samplers {
        for &v in sampler {
            word(&mut out, v);
        }
    }
    word(&mut out, s.static_samplers.len() as u32);
    for t in &s.static_samplers {
        let [u, v, w] = t.address;
        let words = [
            t.filter,
            u,
            v,
            w,
            t.mip_lod_bias.to_bits(),
            t.max_anisotropy,
            t.comparison,
            t.border,
            t.min_lod.to_bits(),
            t.max_lod.to_bits(),
        ];
        for v in words {
            word(&mut out, v);
        }
    }
    for slots in [&s.root_cbvs, &s.root_srvs, &s.root_uavs] {
        word(&mut out, slots.len() as u32);
        out.extend(slots);
    }
    word(&mut out, s.root_indexed_views.len() as u32);
    for view in &s.root_indexed_views {
        for value in [
            u32::from(view.table_slot),
            u32::from(view.constant_slot),
            view.constant_dword,
            view.byte_offset,
            u32::from(view.shift),
        ] {
            word(&mut out, value);
        }
    }
    word(&mut out, s.half_views.len() as u32);
    for &(slot, dword) in &s.half_views {
        word(&mut out, u32::from(slot));
        word(&mut out, dword);
    }
    word(&mut out, s.group_split);
    out
}

pub(crate) fn decode(bytes: &[u8]) -> Option<HostShader> {
    struct Reader<'a>(&'a [u8]);
    impl<'a> Reader<'a> {
        fn take(&mut self, n: usize) -> Option<&'a [u8]> {
            let (head, rest) = self.0.split_at_checked(n)?;
            self.0 = rest;
            Some(head)
        }
        fn word(&mut self) -> Option<u32> {
            Some(u32::from_le_bytes(self.take(4)?.try_into().ok()?))
        }
        fn list(&mut self) -> Option<Vec<u8>> {
            let n = self.word()? as usize;
            Some(self.take(n)?.to_vec())
        }
    }
    let mut r = Reader(bytes);
    if r.take(4)? != MAGIC {
        return None;
    }
    let stage = *STAGES.get(r.word()? as usize)?;
    let dxil = r.list()?;
    let n = r.word()? as usize;
    let mut user_data = Vec::with_capacity(n.min(256));
    for _ in 0..n {
        user_data.push(UserData {
            index: r.word()?,
            slot: u8::try_from(r.word()?).ok()?,
            root: root_type(r.word()?),
            dword: r.word()?,
        });
    }
    let n = r.word()? as usize;
    let mut samplers = Vec::with_capacity(n.min(256));
    for _ in 0..n {
        samplers.push([r.word()?, r.word()?, r.word()?, r.word()?]);
    }
    let n = r.word()? as usize;
    let mut static_samplers = Vec::with_capacity(n.min(256));
    for _ in 0..n {
        static_samplers.push(StaticSampler {
            filter: r.word()?,
            address: [r.word()?, r.word()?, r.word()?],
            mip_lod_bias: f32::from_bits(r.word()?),
            max_anisotropy: r.word()?,
            comparison: r.word()?,
            border: r.word()?,
            min_lod: f32::from_bits(r.word()?),
            max_lod: f32::from_bits(r.word()?),
        });
    }
    let root_cbvs = r.list()?;
    let root_srvs = r.list()?;
    let root_uavs = r.list()?;
    let n = r.word()? as usize;
    let mut root_indexed_views = Vec::with_capacity(n.min(256));
    for _ in 0..n {
        root_indexed_views.push(RootIndexedView {
            table_slot: u8::try_from(r.word()?).ok()?,
            constant_slot: u8::try_from(r.word()?).ok()?,
            constant_dword: r.word()?,
            byte_offset: r.word()?,
            shift: u8::try_from(r.word()?).ok().filter(|&s| s < 32)?,
        });
    }
    let n = r.word()? as usize;
    let mut half_views = Vec::with_capacity(n.min(256));
    for _ in 0..n {
        half_views.push((u8::try_from(r.word()?).ok()?, r.word()?));
    }
    let group_split = r.word().filter(|n| (1..=victory_shader::emit::MAX_GROUP_SPLIT).contains(n))?;
    r.0.is_empty().then_some(HostShader {
        stage,
        dxil,
        user_data,
        samplers,
        static_samplers,
        root_cbvs,
        root_srvs,
        root_uavs,
        root_indexed_views,
        half_views,
        group_split,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn shader() -> HostShader {
        HostShader {
            stage: HwStage::Ps,
            dxil: b"DXBC and the rest".to_vec(),
            user_data: vec![UserData {
                index: 8,
                slot: 1,
                root: RootType::Table,
                dword: 0,
            }],
            samplers: vec![[0x400, 0xc0_0000, 0x18f0_0000, 0]],
            static_samplers: vec![StaticSampler::from_words([
                0x400,
                0xc0_0000,
                0x18f0_0000,
                0,
            ])],
            root_cbvs: vec![2],
            root_srvs: vec![0, 3],
            root_uavs: vec![1],
            root_indexed_views: vec![RootIndexedView {
                table_slot: 1,
                constant_slot: 2,
                constant_dword: 3,
                byte_offset: 32,
                shift: 5,
            }],
            half_views: vec![(0, 8), (4, 0)],
            // (a compute program split in two host groups a console group would say 2;
            // any value in range comes back as stored)
            group_split: 2,
        }
    }

    #[test]
    fn translations_come_back_from_the_cache() {
        let dir = tempfile::tempdir().unwrap();
        let cache = Cache::open(dir.path()).unwrap();
        let build = [7u8; 32];
        let container = b"a console container".as_slice();
        let runs = AtomicU64::new(0);
        let translate = |_: &[u8]| {
            runs.fetch_add(1, Ordering::Relaxed);
            Ok(shader())
        };
        assert_eq!(cached(&cache, &build, false, container, translate), Ok(shader()));
        assert_eq!(cached(&cache, &build, false, container, translate), Ok(shader()));
        assert_eq!(runs.load(Ordering::Relaxed), 1, "the second came from disk");

        // Another build translates again.
        assert_eq!(cached(&cache, &[8; 32], false, container, translate), Ok(shader()));
        assert_eq!(runs.load(Ordering::Relaxed), 2);

        // A damaged entry is a miss, translated and stored again.
        for e in walk(dir.path()) {
            std::fs::write(e, b"VSH1 broken").unwrap();
        }
        assert_eq!(cached(&cache, &build, false, container, translate), Ok(shader()));
        assert_eq!(runs.load(Ordering::Relaxed), 3);
        assert_eq!(cached(&cache, &build, false, container, translate), Ok(shader()));
        assert_eq!(runs.load(Ordering::Relaxed), 3);

        // Failures are not stored.
        let fail = |_: &[u8]| Err::<HostShader, _>("unsupported".to_owned());
        assert!(cached(&cache, &build, false, b"other", fail).is_err());
    }

    #[test]
    fn reopening_the_cache_preserves_each_host_wave_translation() {
        let dir = tempfile::tempdir().unwrap();
        let input = b"synthetic wave reduction";
        let runs = AtomicU64::new(0);
        for wave32 in [false, true, false, true] {
            let cache = Cache::open(dir.path()).unwrap();
            let mut expected = shader();
            expected.stage = HwStage::Cs;
            expected.dxil.push(if wave32 { 32 } else { 64 });
            let result = cached(&cache, &[7; 32], wave32, input, |container| {
                assert_eq!(container, input);
                runs.fetch_add(1, Ordering::Relaxed);
                Ok(expected.clone())
            });
            assert_eq!(result, Ok(expected));
        }
        assert_eq!(runs.load(Ordering::Relaxed), 2);
    }

    fn walk(dir: &std::path::Path) -> Vec<PathBuf> {
        let mut files = Vec::new();
        for e in std::fs::read_dir(dir).unwrap().flatten() {
            if e.path().is_dir() {
                files.extend(walk(&e.path()));
            } else {
                files.push(e.path());
            }
        }
        files
    }

    #[test]
    fn a_program_kept_out_of_the_deep_skips_form_has_a_key_of_its_own() {
        let build = [7u8; 32];
        assert_eq!(program_key(&build, false), build, "allowed: the build's key, as without lists");
        let off = program_key(&build, true);
        assert_ne!(off, build);
        assert_eq!(program_key(&build, true), off, "the same decision, the same key");
        assert_ne!(program_key(&[8; 32], true), off, "and still the build's");

        // So only the program whose decision changed is translated again.
        let dir = tempfile::tempdir().unwrap();
        let cache = Cache::open(dir.path()).unwrap();
        let runs = AtomicU64::new(0);
        let translate = |_: &[u8]| {
            runs.fetch_add(1, Ordering::Relaxed);
            Ok(shader())
        };
        let [a, b] = [b"program a".as_slice(), b"program b"];
        for c in [a, b] {
            cached(&cache, &program_key(&build, false), false, c, translate).unwrap();
        }
        assert_eq!(runs.load(Ordering::Relaxed), 2);
        // a listed: a translated again, b from the cache
        cached(&cache, &program_key(&build, true), false, a, translate).unwrap();
        cached(&cache, &program_key(&build, false), false, b, translate).unwrap();
        assert_eq!(runs.load(Ordering::Relaxed), 3);
        // the list emptied again: both from the cache
        for c in [a, b] {
            cached(&cache, &program_key(&build, false), false, c, translate).unwrap();
        }
        assert_eq!(runs.load(Ordering::Relaxed), 3);
    }

    #[test]
    fn a_program_listed_for_the_wave_wide_form_first_has_a_key_of_its_own() {
        let build = [7u8; 32];
        assert_eq!(wide_first_key(&build, false), build);
        let listed = wide_first_key(&build, true);
        assert_ne!(listed, build);
        assert_ne!(listed, program_key(&build, true), "not the deep-skips decision's key");
        assert_eq!(wide_first_key(&build, true), listed);
    }

    /// `VICTORY_INDIRECT_SPLIT`: the program translated with a group base
    /// is an entry of its own, beside the one without.
    #[test]
    fn a_group_base_has_a_key_of_its_own() {
        let build = [7u8; 32];
        assert_eq!(group_key(&build, 0), build, "no base: the key as it was");
        let based = group_key(&build, 65_535);
        assert_ne!(based, build);
        assert_eq!(group_key(&build, 65_535), based);
        assert_ne!(group_key(&build, 1), based, "another base, another key");
        assert_ne!(group_key(&program_key(&build, true), 65_535), based, "and still the program's");
        let dir = tempfile::tempdir().unwrap();
        let cache = Cache::open(dir.path()).unwrap();
        let runs = AtomicU64::new(0);
        let translate = |_: &[u8]| {
            runs.fetch_add(1, Ordering::Relaxed);
            Ok(shader())
        };
        for key in [build, based, build, based] {
            cached(&cache, &key, false, b"program", translate).unwrap();
        }
        assert_eq!(runs.load(Ordering::Relaxed), 2, "each translated once");
    }

    #[test]
    fn entries_round_trip_and_reject_damage() {
        let bytes = encode(&shader());
        assert_eq!(decode(&bytes), Some(shader()));
        assert_eq!(decode(&bytes[..bytes.len() - 1]), None);
        assert_eq!(decode(&[bytes.as_slice(), &[0]].concat()), None);
        assert_eq!(decode(b"VSH0"), None);
    }

    fn no_switches(_: &str) -> Option<std::ffi::OsString> {
        None
    }

    fn settings() -> Settings {
        Settings {
            loop_limit: 1024,
            view_limit: 999_999,
            group_sync: 1,
            shader_opt: 63,
            heap_map: [u32::MAX; 2],
            rt_miss: false,
        }
    }

    /// A run's key with the translator `fingerprint`, `settings`, and one
    /// switch set (or none).
    fn key(fingerprint: &str, settings: &Settings, switch: Option<(&str, &str)>) -> [u8; 32] {
        let env = |name: &str| {
            switch
                .filter(|(n, _)| *n == name)
                .map(|(_, value)| std::ffi::OsString::from(value))
        };
        run_key_of(&translator_identity(fingerprint), &settings_bytes(settings, &env))
    }

    #[test]
    fn the_same_translator_and_settings_give_the_same_key_and_any_change_another() {
        const A: &str = "0123456789abcdef0123456789abcdef";
        let base = key(A, &settings(), None);
        // Two executables with one translator: one key, so they share entries.
        assert_eq!(base, key(A, &settings(), None));
        assert_eq!(
            base,
            run_key_of(&translator_identity(A), &settings_bytes(&settings(), &no_switches))
        );
        // Another translator.
        assert_ne!(base, key("1123456789abcdef0123456789abcdef", &settings(), None));
        // A switch set, each value of it, and each switch.
        assert_ne!(base, key(A, &settings(), Some(("VICTORY_LOOP_EXITS", "1"))));
        assert_ne!(
            key(A, &settings(), Some(("VICTORY_LOOP_EXITS", "0"))),
            key(A, &settings(), Some(("VICTORY_LOOP_EXITS", "1")))
        );
        assert_ne!(
            key(A, &settings(), Some(("VICTORY_LOOP_EXITS", "1"))),
            key(A, &settings(), Some(("VICTORY_NORM_EXACT", "1")))
        );
        for name in SWITCHES {
            assert_ne!(base, key(A, &settings(), Some((name, "1"))), "{name}");
            // (an empty value is keyed as unset: Windows has no empty variables)
            assert_eq!(base, key(A, &settings(), Some((name, ""))), "{name}");
        }
        // A switch that is not the translator's leaves the key alone.
        assert_eq!(base, key(A, &settings(), Some(("VICTORY_CAPTURE", "somewhere"))));
        // Each setting.
        for changed in [
            Settings { loop_limit: 16, ..settings() },
            Settings { view_limit: 1, ..settings() },
            Settings { group_sync: 0, ..settings() },
            Settings { shader_opt: 0, ..settings() },
            Settings { heap_map: [4096, 7], ..settings() },
            Settings { rt_miss: true, ..settings() },
        ] {
            assert_ne!(base, key(A, &changed, None), "{changed:?}");
        }
    }

    #[test]
    fn the_executable_is_the_identity_only_when_its_word_asks_for_it() {
        use std::ffi::OsStr;
        assert_eq!(Identity::from_switch(None), Identity::Translator);
        assert_eq!(Identity::from_switch(Some(OsStr::new("exe"))), Identity::Exe);
        for other in ["", "0", "1", "EXE", "translator"] {
            assert_eq!(
                Identity::from_switch(Some(OsStr::new(other))),
                Identity::Translator,
                "{other:?}"
            );
        }
        // With the executable's bytes as the identity the key is the hash of
        // those bytes and the settings, as it was before the translator's
        // fingerprint: no two executables share a key.
        let settings = settings_bytes(&settings(), &no_switches);
        let exe = b"MZ and the rest of an executable".as_slice();
        assert_eq!(
            run_key_of(exe, &settings),
            *blake3::hash(&[exe, &settings[..]].concat()).as_bytes()
        );
        assert_ne!(
            run_key_of(exe, &settings),
            run_key_of(b"MZ and the rest of another executable", &settings)
        );
    }

    #[test]
    fn this_build_has_a_translator_fingerprint_and_the_identity_holds_it() {
        let f = victory_shader::TRANSLATOR_FINGERPRINT;
        assert_eq!(f.len(), 32, "{f}");
        assert!(f.bytes().all(|b| b.is_ascii_hexdigit()), "{f}");
        let id = translator_identity(f);
        assert!(id.windows(f.len()).any(|w| w == f.as_bytes()));
        // And this file's hash (the entries' encoding), after it.
        assert!(id.ends_with(blake3::hash(include_bytes!("shader_cache.rs")).as_bytes()));
    }

    #[test]
    fn passes_given_to_one_program_are_in_its_key() {
        let key = [3u8; 32];
        assert_eq!(opt_key(&key, 63, 63), key);
        assert_ne!(opt_key(&key, 0, 63), key);
        assert_ne!(opt_key(&key, 0, 63), opt_key(&key, 1, 63));
    }

    #[test]
    fn a_run_keys_folder_is_named_by_the_keys_first_bytes() {
        let mut key = [0u8; 32];
        key[0] = 0xab;
        key[7] = 0x01;
        key[8] = 0xff;
        assert_eq!(key_folder(&key), "k-ab00000000000001");
    }

    /// The safety net for the key: every environment switch the translator's
    /// sources name (a `"VICTORY_..."` string) is either in [`SWITCHES`], or
    /// listed here with the reason it need not be. A switch added to the
    /// translator and to neither fails this test, instead of serving a run
    /// with it the shaders cached without it.
    #[test]
    fn every_switch_the_translator_reads_is_accounted_for() {
        // In the run's key by the value the translator makes of them (`Settings`).
        const BY_VALUE: [&str; 6] = [
            // (not a switch: the translator's fingerprint, which its build
            // script sets at compile time and the key starts with)
            "VICTORY_TRANSLATOR_FINGERPRINT",
            "VICTORY_SHADER_LOOP_LIMIT",
            "VICTORY_GROUP_SYNC",
            "VICTORY_SHADER_OPT",
            "VICTORY_RT_MISS",
            "VICTORY_BVH_BUILDER",
        ];
        // In each program's key by what they decide for it (`program_key`,
        // `wide_first_key`).
        const PER_PROGRAM: [&str; 6] = [
            "VICTORY_WAVE_DEEP_SKIPS_NOT",
            "VICTORY_WAVE_DEEP_SKIPS_ONLY",
            "VICTORY_WAVE_WIDE_FIRST",
            // (test/pairs: the paired form and its lists, by each program's
            // answer, `pairs_key`)
            "VICTORY_WAVE_PAIRS",
            "VICTORY_WAVE_PAIRS_NOT",
            "VICTORY_WAVE_PAIRS_ONLY",
        ];
        // Not part of a translation: lines printed and files dumped beside
        // it, the validator and the host device of the offline checks, and
        // what only tests read.
        const NOT_OUTPUT: [&str; 14] = [
            "VICTORY_WAVE_WHY",
            // (test/shader5: an ignored test's folder of containers)
            "VICTORY_SAMPLER_SWEEP",
            "VICTORY_DESC_WHY",
            "VICTORY_EMIT_DUMP",
            "VICTORY_SHADER_CENSUS",
            "VICTORY_DXIL",
            "VICTORY_D3D12_DEBUG",
            "VICTORY_SHADER_ADAPTER",
            "VICTORY_WARP",
            "VICTORY_CONSOLE_DXC",
            "VICTORY_PC_DXC",
            "VICTORY_DIFF_ONLY",
            "VICTORY_REF_PS",
            "VICTORY_REF_VS",
        ];
        fn sources(dir: &std::path::Path, out: &mut Vec<std::path::PathBuf>) {
            for entry in std::fs::read_dir(dir).unwrap() {
                let path = entry.unwrap().path();
                if path.is_dir() {
                    sources(&path, out);
                } else if path.extension().is_some_and(|e| e == "rs") {
                    out.push(path);
                }
            }
        }
        let src = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../victory-shader/src");
        let mut files = Vec::new();
        sources(&src, &mut files);
        assert!(files.len() > 10, "the translator's sources, looked for in {}", src.display());
        let mut found = 0;
        let mut unknown = std::collections::BTreeSet::new();
        for file in files {
            let text = std::fs::read_to_string(&file).unwrap();
            let mut rest = text.as_str();
            while let Some(at) = rest.find("\"VICTORY_") {
                let name: String = rest[at + 1..]
                    .chars()
                    .take_while(|c| c.is_ascii_uppercase() || c.is_ascii_digit() || *c == '_')
                    .collect();
                rest = &rest[at + 1 + name.len()..];
                // (a whole string only: `"VICTORY_X=1 does ..."` is a message)
                if !rest.starts_with('"') {
                    continue;
                }
                found += 1;
                let name = name.as_str();
                if !(SWITCHES.contains(&name)
                    || BY_VALUE.contains(&name)
                    || PER_PROGRAM.contains(&name)
                    || NOT_OUTPUT.contains(&name))
                {
                    unknown.insert(format!(
                        "{name} in {}",
                        file.file_name().unwrap_or_default().to_string_lossy()
                    ));
                }
            }
        }
        assert!(found >= 40, "only {found} switch names found: has the way they are read changed?");
        assert!(
            unknown.is_empty(),
            "the translator reads switches the shader cache's key does not account for: {unknown:?}. \
             Add each to SWITCHES in shader_cache.rs if it can change translated code, or to this \
             test's lists with the reason if it cannot."
        );
    }
}
