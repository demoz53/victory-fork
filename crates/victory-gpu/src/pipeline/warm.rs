//! Host pipelines made ahead of a run, into the pipeline library the run
//! reads (`victory warm-pipelines`).
//!
//! A translation switch that changes the code of most programs (say
//! `VICTORY_HEAP_HANDLE_LOCAL`, 2,944 of 3,605) gives each of their
//! pipelines a new name in the library ([`super::library::name`] hashes the
//! translated code): a run with it finds none of them and makes them all
//! while the title plays, and draws whose pipeline is still being made are
//! dropped (black frames for minutes).
//!
//! 1. **Records.** Each pipeline a run makes is written down, before the
//!    driver sees it, in the library's folder: one line of
//!    [`RECORD_FILE`] (its early-list key, its shaders as `stage hash`, and
//!    its graphics state, [`GraphicsState::to_record`]), and each shader
//!    container as the title gave it under [`RECORD_SHADERS`]. Nothing of
//!    it depends on the translation switches. `VICTORY_PIPELINE_RECORD=0`:
//!    none written.
//! 2. **Warm.** [`batch`] makes each recorded pipeline again through the
//!    run's own code ([`make::now`], [`super::host::HostPipeline::new`],
//!    [`super::library::pipeline`]) with this process's switches: the
//!    translation, root signature, description and library name are a
//!    run's, and the made pipelines are stored and saved into the library
//!    as a run saves it. A record whose remade key differs from the one it
//!    was written with is skipped. Compute pipelines need no state, so a
//!    folder of saved containers (`VICTORY_SHADER_SAVE`) gives them too.
//!
//! What a host compiler that dies or never returns does to a warm: from
//! PastiiJay/klbf-lab's pipeline pre-build (GPL-3.0, `apps/victory/src/
//! prebuild.rs`, itself from upstream VIctory 9c58a1c), the ideas, not its
//! code: the pipelines being made are listed in a file before the driver
//! sees them (`-inflight`); after a crash those are made first, one at a
//! time, and one that was made alone when the process died is listed as
//! crashed (`-crashed`) and skipped from then on; the work is done in
//! batches, each in a fresh process (`victory warm-pipelines` restarts its
//! child), and what is saved is listed (`-done`) so a next batch skips it.

use std::collections::{BTreeMap, HashSet};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use super::Shader;
use super::host::GraphicsState;
use super::make::{self, Pending};

/// The pipeline records, in the library's folder.
pub const RECORD_FILE: &str = "recorded-pipelines.txt";
/// The recorded pipelines' shader containers (`<hash>.<stage>.cdxil`), in
/// the library's folder.
pub const RECORD_SHADERS: &str = "recorded-shaders";

/// Set while this process warms the library: it writes no records.
static WARMING: AtomicBool = AtomicBool::new(false);

fn recording() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    !WARMING.load(Ordering::Relaxed)
        && *ON.get_or_init(|| std::env::var_os("VICTORY_PIPELINE_RECORD").is_none_or(|v| v != "0"))
}

/// The keys recorded (this run and earlier ones), loaded on first use.
static RECORDED: Mutex<Option<HashSet<String>>> = Mutex::new(None);
/// Serialises the appends.
static APPEND: Mutex<()> = Mutex::new(());

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

/// The stage names the pipeline code gives shaders.
fn stage_name(stage: &str) -> Option<&'static str> {
    ["vertex", "pixel", "domain", "hull", "geometry", "compute", "mesh", "amplification"]
        .into_iter()
        .find(|s| *s == stage)
}

fn hex(bytes: &[u8]) -> String {
    use std::fmt::Write;
    let mut out = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        let _ = write!(out, "{b:02x}");
    }
    out
}

fn unhex(text: &str) -> Option<Vec<u8>> {
    if !text.len().is_multiple_of(2) {
        return None;
    }
    (0..text.len()).step_by(2).map(|i| u8::from_str_radix(text.get(i..i + 2)?, 16).ok()).collect()
}

/// Writes `bytes` to `path` through a temporary file beside it.
fn write_new(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    let mut tmp = path.as_os_str().to_owned();
    tmp.push(format!(".tmp-{}", std::process::id()));
    let tmp = PathBuf::from(tmp);
    let res = std::fs::write(&tmp, bytes).and_then(|()| std::fs::rename(&tmp, path));
    if res.is_err() {
        let _ = std::fs::remove_file(&tmp);
    }
    res
}

/// One record's line.
fn line(key: &str, shaders: &[Arc<Shader>], graphics: Option<&GraphicsState>) -> String {
    let names: Vec<String> = shaders.iter().map(|s| format!("{} {}", s.stage, s.hash)).collect();
    let state = graphics.map_or_else(|| "-".to_owned(), |g| hex(&g.to_record()));
    format!("{key}\t{}\t{state}\n", names.join(","))
}

/// Writes the record of a pipeline about to be made (see the module's
/// notes), once per key across runs.
pub(crate) fn record(key: &str, shaders: &[Arc<Shader>], graphics: Option<&GraphicsState>) {
    if shaders.is_empty() || !recording() {
        return;
    }
    let Some(dir) = super::library::dir() else { return };
    {
        let mut recorded = lock(&RECORDED);
        let keys = recorded.get_or_insert_with(|| {
            let text = std::fs::read_to_string(dir.join(RECORD_FILE)).unwrap_or_default();
            text.lines().filter_map(|l| l.split('\t').next()).filter(|k| !k.is_empty()).map(str::to_owned).collect()
        });
        if !keys.insert(key.to_owned()) {
            return;
        }
    }
    // The containers first: a record is of use only with its shaders.
    let shaders_dir = dir.join(RECORD_SHADERS);
    if std::fs::create_dir_all(&shaders_dir).is_err() {
        return;
    }
    for s in shaders {
        let path = shaders_dir.join(format!("{}.{}.cdxil", s.hash, s.stage));
        if !path.exists()
            && let Err(e) = write_new(&path, s.container())
        {
            tracing::warn!(error = %e, path = %path.display(), "pipeline record: shader container not written");
            return;
        }
    }
    let text = line(key, shaders, graphics);
    let _one = lock(&APPEND);
    let written = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(dir.join(RECORD_FILE))
        .and_then(|mut f| f.write_all(text.as_bytes()));
    if let Err(e) = written {
        tracing::warn!(error = %e, "pipeline record not written");
    }
}

/// A pipeline to make: its key, shaders and state (`None`: compute, or a
/// pipeline made from the plain graphics description).
#[derive(Clone, Debug, PartialEq)]
pub struct Record {
    pub key: String,
    pub shaders: Vec<(&'static str, String)>,
    pub state: Option<Vec<u8>>,
    /// Whether its key is checked against the remade pipeline's (records
    /// made from a folder of containers have none of their own).
    pub keyed: bool,
}

impl Record {
    /// A line of [`RECORD_FILE`].
    pub fn parse(line: &str) -> Option<Record> {
        let mut parts = line.trim_end().split('\t');
        let key = parts.next()?.to_owned();
        let shaders = parts
            .next()?
            .split(',')
            .map(|s| {
                let (stage, hash) = s.split_once(' ')?;
                Some((stage_name(stage)?, hash.to_owned()))
            })
            .collect::<Option<Vec<_>>>()?;
        let state = match parts.next()? {
            "-" => None,
            text => Some(unhex(text)?),
        };
        (key.len() == 32 && !shaders.is_empty() && parts.next().is_none()).then_some(Record { key, shaders, state, keyed: true })
    }

    /// A compute pipeline of the container `hash` (no record of its own):
    /// the key a compute pipeline made from the compute description has.
    pub fn compute(hash: &str) -> Record {
        let mut key = blake3::Hasher::new();
        key.update(b"compute");
        key.update(hash.as_bytes());
        Record {
            key: key.finalize().to_hex()[..32].to_owned(),
            shaders: vec![("compute", hash.to_owned())],
            state: None,
            keyed: false,
        }
    }

    /// Its shaders as `stage hash`, the hashes `VICTORY_SKIP_PIPELINES` takes.
    pub fn names(&self) -> String {
        let names: Vec<String> = self.shaders.iter().map(|(stage, hash)| format!("{stage} {hash}")).collect();
        names.join(", ")
    }
}

/// The records of `file` (a [`RECORD_FILE`]), each key once; the lines it
/// could not read are counted.
pub fn read_records(file: &Path) -> std::io::Result<(Vec<Record>, usize)> {
    let text = std::fs::read_to_string(file)?;
    let (mut out, mut bad, mut seen) = (Vec::new(), 0, HashSet::new());
    for l in text.lines().filter(|l| !l.trim().is_empty()) {
        match Record::parse(l) {
            Some(r) if seen.insert(r.key.clone()) => out.push(r),
            Some(_) => {}
            None => bad += 1,
        }
    }
    Ok((out, bad))
}

/// A compute pipeline record for each `<hash>.compute.cdxil` in `dir` (a
/// `VICTORY_SHADER_SAVE` folder).
pub fn compute_records(dir: &Path) -> std::io::Result<Vec<Record>> {
    let mut out = Vec::new();
    for entry in std::fs::read_dir(dir)? {
        let name = entry?.file_name();
        let name = name.to_string_lossy();
        if let Some(hash) = name.strip_suffix(".compute.cdxil")
            && hash.len() == 16
            && hash.bytes().all(|b| b.is_ascii_hexdigit())
        {
            out.push(Record::compute(hash));
        }
    }
    out.sort_by(|a, b| a.key.cmp(&b.key));
    Ok(out)
}

/// What a warm is asked to do.
pub struct Options {
    /// The pipelines to make.
    pub records: Vec<Record>,
    /// Where shader containers are looked for (`<hash>.<stage>.cdxil`, or
    /// `.bin` as `VICTORY_DUMP_REFUSED` writes them).
    pub shader_dirs: Vec<PathBuf>,
    /// Where the warm's own lists go (done, in flight, crashed), and the
    /// tag their names carry (the switch set: a list is of use only with
    /// the translations it was made with).
    pub state_dir: PathBuf,
    pub tag: String,
    /// Worker threads.
    pub threads: usize,
    /// Pipelines made (not loaded) in this batch, at most; 0: no limit.
    pub batch: usize,
    /// Go on when the library's file is there but the host refuses it (the
    /// warm then starts an empty library, and its save replaces the file).
    pub allow_fresh: bool,
}

/// What a batch did.
#[derive(Debug, Default)]
pub struct Report {
    pub records: usize,
    /// Listed as done by an earlier batch, or crashed in one.
    pub done_before: usize,
    pub crashed_before: usize,
    /// Made alone after a crash (suspects), and newly listed as crashed.
    pub suspects: usize,
    pub newly_crashed: Vec<String>,
    /// In the library already, made and stored, no host pipeline.
    pub loaded: usize,
    pub made: usize,
    pub refused: usize,
    /// Skipped, by reason.
    pub skipped: BTreeMap<String, usize>,
    /// Not reached in this batch (its limit).
    pub left: usize,
    /// The bytes of the library's file when it was opened.
    pub library_bytes: u64,
    /// The library and the lists were saved.
    pub saved: bool,
}

/// The warm's list files.
pub fn list_path(state_dir: &Path, tag: &str, what: &str) -> PathBuf {
    state_dir.join(format!("warm-{tag}-{what}.txt"))
}

/// The keys a list file holds (the first word of each line).
pub fn read_keys(path: &Path) -> BTreeMap<String, String> {
    std::fs::read_to_string(path)
        .unwrap_or_default()
        .lines()
        .filter_map(|l| {
            let (key, rest) = l.split_once(['\t', ' ']).unwrap_or((l, ""));
            (!key.is_empty()).then(|| (key.to_owned(), rest.to_owned()))
        })
        .collect()
}

fn append(path: &Path, lines: &[String]) -> std::io::Result<()> {
    if lines.is_empty() {
        return Ok(());
    }
    let mut f = std::fs::OpenOptions::new().create(true).append(true).open(path)?;
    f.write_all(lines.concat().as_bytes())?;
    f.sync_all()
}

/// The container of `stage hash` in `dirs`, checked against its hash.
fn container(dirs: &[PathBuf], stage: &str, hash: &str) -> Option<Vec<u8>> {
    for dir in dirs {
        for ext in ["cdxil", "bin"] {
            let Ok(bytes) = std::fs::read(dir.join(format!("{hash}.{stage}.{ext}"))) else { continue };
            if blake3::hash(&bytes).to_hex().starts_with(hash) {
                return Some(bytes);
            }
        }
    }
    None
}

enum Outcome {
    Loaded,
    Made,
    Refused(&'static str),
    Skipped(String),
}

/// The pipelines being made now (their shaders, and since when in seconds
/// of the system clock), written to the in-flight file at each change:
/// before the driver sees a pipeline, so the file names it if the process
/// dies, and the supervisor ends a batch on one that never finishes.
struct InFlight {
    path: PathBuf,
    keys: Mutex<BTreeMap<String, (String, u64)>>,
}

impl InFlight {
    fn set(&self, key: &str, names: Option<String>) {
        let mut keys = lock(&self.keys);
        match names {
            Some(n) => {
                let since = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map_or(0, |d| d.as_secs());
                keys.insert(key.to_owned(), (n, since));
            }
            None => {
                keys.remove(key);
            }
        }
        let text: String = keys.iter().map(|(k, (n, since))| format!("{k}\t{n}\t{since}\n")).collect();
        let _ = std::fs::write(&self.path, text);
    }
}

fn make_one(r: &Record, opts: &Options, inflight: &InFlight) -> Outcome {
    let mut shaders = Vec::with_capacity(r.shaders.len());
    for (stage, hash) in &r.shaders {
        match container(&opts.shader_dirs, stage, hash) {
            Some(bytes) => shaders.push(super::shader(stage, &bytes)),
            None => return Outcome::Skipped("a shader container is not in the shader folders".into()),
        }
    }
    let graphics = match &r.state {
        Some(bytes) => match GraphicsState::from_record(bytes) {
            Some(g) => Some(Box::new(g)),
            None => return Outcome::Skipped("graphics state unreadable".into()),
        },
        None => None,
    };
    let p = Arc::new(Pending::new(shaders, graphics));
    if r.keyed && p.key() != r.key {
        return Outcome::Skipped("the remade pipeline's key differs from the record's".into());
    }
    inflight.set(&r.key, Some(r.names()));
    super::library::take_outcome();
    let made = make::now(&p);
    let outcome = super::library::take_outcome();
    inflight.set(&r.key, None);
    match (made, outcome) {
        (true, super::library::LOADED_ONE) => Outcome::Loaded,
        (true, super::library::MADE_ONE) => Outcome::Made,
        (true, _) => Outcome::Refused("made outside the library"),
        (false, _) if p_untranslated(&p) => Outcome::Refused("a shader does not translate"),
        (false, _) => Outcome::Refused("the host refused it (or a stage is not translated)"),
    }
}

fn p_untranslated(p: &Pending) -> bool {
    p.shaders().iter().any(|s| s.translated().is_none())
}

/// Makes the records' pipelines into the library (set by
/// [`super::use_pipeline_cache`] before), saves it, and lists them as done.
/// A pipeline listed done or crashed in an earlier batch of the same tag is
/// skipped; the ones in flight when an earlier batch died are made first,
/// one at a time.
pub fn batch(opts: &Options) -> Result<Report, String> {
    WARMING.store(true, Ordering::Relaxed);
    let dir = super::library::dir().ok_or("no library folder set")?;
    let file_len = std::fs::metadata(dir.join("host-pipelines.bin")).map_or(0, |m| m.len());
    // The early list is loaded now, so that the save writes it back whole.
    let _ = super::library::wanted_early("");
    let opened = super::library::open_now().ok_or("no host pipeline library (no host GPU?)")?;
    if file_len > 0 && opened == 0 && !opts.allow_fresh {
        return Err(format!(
            "the host refused the library file ({file_len} bytes: another driver or adapter?); \
             a warm would start an empty one and replace the file (--allow-fresh to go on)"
        ));
    }
    let mut report = Report {
        records: opts.records.len(),
        library_bytes: opened,
        ..Report::default()
    };
    let done_path = list_path(&opts.state_dir, &opts.tag, "done");
    let crashed_path = list_path(&opts.state_dir, &opts.tag, "crashed");
    let inflight_path = list_path(&opts.state_dir, &opts.tag, "inflight");
    let done = read_keys(&done_path);
    let mut crashed = read_keys(&crashed_path);
    let suspects = read_keys(&inflight_path);
    if suspects.len() == 1 {
        // Made alone when the process died: the culprit.
        let lines: Vec<String> = suspects.iter().map(|(k, n)| format!("{k}\t{n}\n")).collect();
        append(&crashed_path, &lines).map_err(|e| format!("crashed list not written: {e}"))?;
        report.newly_crashed = lines.iter().map(|l| l.trim_end().to_owned()).collect();
        crashed.extend(suspects.clone());
    }
    let mut alone = Vec::new();
    let mut todo = Vec::new();
    for r in &opts.records {
        if crashed.contains_key(&r.key) {
            report.crashed_before += 1;
        } else if done.contains_key(&r.key) {
            report.done_before += 1;
        } else if suspects.len() > 1 && suspects.contains_key(&r.key) {
            alone.push(r);
        } else {
            todo.push(r);
        }
    }
    report.suspects = alone.len();
    let inflight = InFlight {
        path: inflight_path.clone(),
        keys: Mutex::new(BTreeMap::new()),
    };
    inflight.set("", None);
    let finished: Mutex<Vec<String>> = Mutex::new(Vec::new());
    let counts: Mutex<Report> = Mutex::new(Report::default());
    let made = AtomicUsize::new(0);
    let taken = AtomicUsize::new(0);
    let started = std::time::Instant::now();
    let tally = |r: &Record, outcome: Outcome| {
        let mut c = lock(&counts);
        let word = match outcome {
            Outcome::Loaded => {
                c.loaded += 1;
                "loaded"
            }
            Outcome::Made => {
                c.made += 1;
                made.fetch_add(1, Ordering::Relaxed);
                "made"
            }
            Outcome::Refused(why) => {
                c.refused += 1;
                *c.skipped.entry(format!("no host pipeline: {why}")).or_default() += 1;
                "refused"
            }
            Outcome::Skipped(why) => {
                *c.skipped.entry(why).or_default() += 1;
                return;
            }
        };
        let n = c.loaded + c.made + c.refused;
        drop(c);
        lock(&finished).push(format!("{} {word}\n", r.key));
        if n.is_multiple_of(100) {
            println!(
                "warm         {n} done ({} made) in {:.0} s",
                made.load(Ordering::Relaxed),
                started.elapsed().as_secs_f64()
            );
        }
    };
    for r in &alone {
        println!("warm         suspect of a crash, made alone: {} [{}]", r.key, r.names());
        tally(r, make_one(r, opts, &inflight));
    }
    let limit = if opts.batch == 0 { usize::MAX } else { opts.batch };
    std::thread::scope(|s| {
        for _ in 0..opts.threads.max(1).min(todo.len()) {
            s.spawn(|| {
                while made.load(Ordering::Relaxed) < limit {
                    let Some(r) = todo.get(taken.fetch_add(1, Ordering::Relaxed)) else { break };
                    tally(r, make_one(r, opts, &inflight));
                }
            });
        }
    });
    let c = counts.into_inner().unwrap_or_else(|e| e.into_inner());
    report.loaded = c.loaded;
    report.made = c.made;
    report.refused = c.refused;
    report.skipped = c.skipped;
    report.left = todo.len().saturating_sub(taken.load(Ordering::Relaxed).min(todo.len()));
    println!(
        "warm         {} made, {} loaded, {} without a host pipeline in {:.1} s; saving the library",
        report.made,
        report.loaded,
        report.refused,
        started.elapsed().as_secs_f64()
    );
    let saving = std::time::Instant::now();
    report.saved = super::library::save(true);
    if report.saved {
        let lines = finished.into_inner().unwrap_or_else(|e| e.into_inner());
        append(&done_path, &lines).map_err(|e| format!("done list not written: {e}"))?;
        println!("warm         saved in {:.1} s", saving.elapsed().as_secs_f64());
    }
    // Nothing is in flight any more (a crash from here on is not a pipeline's).
    let _ = std::fs::remove_file(&inflight_path);
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn records_read_back() {
        let line = format!("{}\tvertex 0011223344556677,pixel 8899aabbccddeeff\t47533100\n", "a".repeat(32));
        let r = Record::parse(&line).expect("a record");
        assert_eq!(r.shaders, vec![("vertex", "0011223344556677".to_owned()), ("pixel", "8899aabbccddeeff".to_owned())]);
        assert_eq!(r.state.as_deref(), Some(&b"GS1\0"[..]));
        let compute = format!("{}\tcompute 0011223344556677\t-", "b".repeat(32));
        assert_eq!(Record::parse(&compute).unwrap().state, None);
        assert!(Record::parse("short\tcompute 00\t-").is_none());
        assert!(Record::parse(&format!("{}\tnot-a-stage 00\t-", "c".repeat(32))).is_none());
        assert!(Record::parse(&format!("{}\tcompute 00\t4", "c".repeat(32))).is_none(), "odd hex");
    }

    /// A compute record's key is the one a run gives a pipeline made from
    /// the compute description (shaders, no state).
    #[test]
    fn a_compute_record_has_the_run_key() {
        let bytes = b"not a container, only bytes to hash".to_vec();
        let s = super::super::shader("compute", &bytes);
        let p = Pending::new(vec![s.clone()], None);
        assert_eq!(Record::compute(&s.hash).key, p.key());
    }

    #[test]
    fn lists_read_their_keys() {
        let dir = tempfile::tempdir().unwrap();
        let path = list_path(dir.path(), "t", "done");
        append(&path, &["k1 made\n".into(), "k2\tcompute 00\n".into()]).unwrap();
        let keys = read_keys(&path);
        assert_eq!(keys.keys().map(String::as_str).collect::<Vec<_>>(), ["k1", "k2"]);
        assert_eq!(keys["k2"], "compute 00");
        assert!(read_keys(&dir.path().join("none")).is_empty());
    }

    #[test]
    fn containers_are_found_by_their_hash() {
        let dir = tempfile::tempdir().unwrap();
        let bytes = b"container bytes".to_vec();
        let hash = blake3::hash(&bytes).to_hex()[..16].to_owned();
        std::fs::write(dir.path().join(format!("{hash}.pixel.cdxil")), &bytes).unwrap();
        // Another file under a wrong name is not taken.
        std::fs::write(dir.path().join("0000000000000000.pixel.cdxil"), &bytes).unwrap();
        let dirs = vec![dir.path().to_owned()];
        assert_eq!(container(&dirs, "pixel", &hash), Some(bytes));
        assert_eq!(container(&dirs, "pixel", "0000000000000000"), None);
        assert_eq!(container(&dirs, "vertex", &hash), None);
    }
}
