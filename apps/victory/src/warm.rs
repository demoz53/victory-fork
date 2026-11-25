//! `victory warm-pipelines`: the host pipelines of recorded title pipelines
//! made ahead of a run, with this process's translation switches, into the
//! pipeline library a run reads (`victory_gpu::pipeline::warm`).
//!
//! Run it with the executable the game runs with (the diagnostics build of
//! the same tree) and the same `VICTORY_*` switches as the run it is for:
//! the library names a pipeline by its translated code.
//!
//! - Refuses while a `victory.exe` other than itself runs, or while another
//!   process holds the library's file open (a game has it mapped).
//! - Backs the library's file and the early list up first (`--no-backup`:
//!   not), into a folder beside the library's.
//! - The work is done by child processes (itself with `--child`), a batch
//!   of pipelines each: a child the host driver kills, or that stays on one
//!   pipeline for `--stall` seconds (then ended), is followed by another,
//!   which makes the pipelines in flight at that moment first, one at a
//!   time; one made alone when its process died is listed as crashed and
//!   skipped (see `victory_gpu::pipeline::warm` for where the idea is from).

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, SystemTime};

use anyhow::{Context, Result, bail};
use victory_gpu::pipeline::warm::{self, Record};

const LIBRARY_FILE: &str = "host-pipelines.bin";
const EARLY_FILE: &str = "early-pipelines.txt";
/// A child's exit code when its batch is saved and records are left.
const MORE: i32 = 3;
/// A child's exit code when it could not go on (nothing is gained by another).
const STOP: i32 = 4;

pub struct Args {
    pub library_dir: PathBuf,
    pub records: Vec<PathBuf>,
    pub compute_from: Vec<PathBuf>,
    pub shaders: Vec<PathBuf>,
    pub jobs: Option<usize>,
    pub batch: usize,
    pub stall: u64,
    pub no_backup: bool,
    pub backup_dir: Option<PathBuf>,
    pub shader_cache: Option<PathBuf>,
    pub allow_fresh: bool,
    pub rounds: usize,
    /// Only for a copy of the library: go on while a game runs (the check
    /// that no other process holds the library's file open stays).
    pub allow_running_game: bool,
}

/// The switch set's tag: our executable and the `VICTORY_*` variables
/// (those that only name or steer a run left out), hashed.
fn tag() -> Result<(String, Vec<String>)> {
    let exe = std::env::current_exe().and_then(std::fs::read).context("reading our executable")?;
    let mut switches: Vec<String> = std::env::vars_os()
        .filter_map(|(k, v)| Some((k.into_string().ok()?, v.to_string_lossy().into_owned())))
        .filter(|(k, _)| k.starts_with("VICTORY_"))
        .filter(|(k, _)| !matches!(k.as_str(), "VICTORY_LOG" | "VICTORY_RUN_NAME" | "VICTORY_KEYS_FILE"))
        .map(|(k, v)| format!("{k}={v}"))
        .collect();
    switches.sort();
    let mut h = blake3::Hasher::new();
    h.update(&exe);
    for s in &switches {
        h.update(s.as_bytes());
        h.update(&[0]);
    }
    Ok((h.finalize().to_hex()[..12].to_owned(), switches))
}

/// Another `victory.exe` running (a game), by process id.
fn game_running() -> Result<Option<u32>> {
    let out = std::process::Command::new("tasklist")
        .args(["/FI", "IMAGENAME eq victory.exe", "/FO", "CSV", "/NH"])
        .output()
        .context("running tasklist")?;
    let me = std::process::id();
    Ok(String::from_utf8_lossy(&out.stdout)
        .lines()
        .filter_map(|l| {
            let mut f = l.split(',').map(|f| f.trim_matches('"'));
            let name = f.next()?;
            let pid: u32 = f.next()?.parse().ok()?;
            (name.eq_ignore_ascii_case("victory.exe") && pid != me).then_some(pid)
        })
        .next())
}

/// Whether another process has `path` open (a game maps the library).
fn in_use(path: &Path) -> bool {
    use std::os::windows::fs::OpenOptionsExt;
    path.exists() && std::fs::OpenOptions::new().read(true).share_mode(0).open(path).is_err()
}

fn ensure_alone(library: &Path, allow_running_game: bool) -> Result<()> {
    if !allow_running_game && let Some(pid) = game_running()? {
        bail!("victory.exe (process {pid}) is running: warm the library only while no game runs");
    }
    if in_use(library) {
        bail!("{} is open in another process (a game running?): not warming it", library.display());
    }
    Ok(())
}

fn unix_now() -> u64 {
    SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).map_or(0, |d| d.as_secs())
}

/// Copies the library's file and the early list into a new folder; the folder.
fn backup(dir: &Path, to: Option<&Path>) -> Result<Option<PathBuf>> {
    let library = dir.join(LIBRARY_FILE);
    if !library.exists() {
        println!("backup       no library file yet: nothing to back up");
        return Ok(None);
    }
    let name = dir.file_name().map_or("pipelines".into(), |n| n.to_string_lossy().into_owned());
    let folder = match to {
        Some(to) => to.to_owned(),
        None => dir.with_file_name(format!("{name}-backup-{}", unix_now())),
    };
    std::fs::create_dir_all(&folder).with_context(|| format!("creating {}", folder.display()))?;
    let started = Instant::now();
    for file in [LIBRARY_FILE, EARLY_FILE] {
        let from = dir.join(file);
        if from.exists() {
            let to = folder.join(file);
            let bytes = std::fs::copy(&from, &to).with_context(|| format!("backing {} up", from.display()))?;
            let len = std::fs::metadata(&from)?.len();
            if bytes != len {
                bail!("backup of {} is {bytes} bytes of {len}", from.display());
            }
        }
    }
    println!("backup       {} ({:.1} s)", folder.display(), started.elapsed().as_secs_f64());
    println!(
        "restore      copy /y \"{}\\{LIBRARY_FILE}\" \"{}\\\" and the same for {EARLY_FILE}",
        folder.display(),
        dir.display()
    );
    Ok(Some(folder))
}

/// The records the arguments name, each key once.
fn records(a: &Args, dir: &Path) -> Result<Vec<Record>> {
    let mut files = a.records.clone();
    let default = dir.join(warm::RECORD_FILE);
    if files.is_empty() && default.exists() {
        files.push(default);
    }
    let mut out: Vec<Record> = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for f in &files {
        let (list, bad) = warm::read_records(f).with_context(|| format!("reading {}", f.display()))?;
        if bad > 0 {
            println!("records      {}: {bad} lines not read", f.display());
        }
        out.extend(list.into_iter().filter(|r| seen.insert(r.key.clone())));
    }
    for d in &a.compute_from {
        let list = warm::compute_records(d).with_context(|| format!("reading {}", d.display()))?;
        out.extend(list.into_iter().filter(|r| seen.insert(r.key.clone())));
    }
    Ok(out)
}

fn shader_dirs(a: &Args, dir: &Path) -> Vec<PathBuf> {
    let mut dirs = vec![dir.join(warm::RECORD_SHADERS)];
    dirs.extend(a.shaders.iter().cloned());
    dirs.extend(a.compute_from.iter().cloned());
    dirs
}

/// The supervisor: checks, backup, then child batches until done.
pub fn run(a: Args) -> Result<()> {
    let dir = a.library_dir.clone();
    anyhow::ensure!(dir.is_dir(), "{} is not a folder", dir.display());
    let library = dir.join(LIBRARY_FILE);
    ensure_alone(&library, a.allow_running_game)?;
    let (tag, switches) = tag()?;
    let records = records(&a, &dir)?;
    println!("library      {} ({} bytes)", library.display(), std::fs::metadata(&library).map_or(0, |m| m.len()));
    println!("switches     {} (tag {tag})", if switches.is_empty() { "none".to_owned() } else { switches.join(" ") });
    let (graphics, compute) = records.iter().fold((0, 0), |(g, c), r| if r.state.is_some() { (g + 1, c) } else { (g, c + 1) });
    println!("records      {} pipelines ({graphics} with a graphics state, {compute} without)", records.len());
    if records.is_empty() {
        bail!(
            "no pipelines to make: no {} in the library folder (a run of this build writes it) and no --records or --compute-from",
            warm::RECORD_FILE
        );
    }
    if !a.no_backup {
        backup(&dir, a.backup_dir.as_deref())?;
    }
    let started = Instant::now();
    let exe = std::env::current_exe()?;
    let inflight = warm::list_path(&dir, &tag, "inflight");
    let mut finished = false;
    for round in 1..=a.rounds.max(1) {
        ensure_alone(&library, a.allow_running_game)?;
        println!("round        {round} ({:.0} s in)", started.elapsed().as_secs_f64());
        let mut child = std::process::Command::new(&exe)
            .args(std::env::args_os().skip(1))
            .arg("--child")
            .spawn()
            .context("starting a warm batch")?;
        let status = loop {
            if let Some(status) = child.try_wait()? {
                break Some(status);
            }
            std::thread::sleep(Duration::from_secs(1));
            if let Some((key, secs)) = oldest_in_flight(&inflight)
                && secs > a.stall
            {
                println!("stalled      {key} in flight for {secs} s: ending the batch");
                let _ = child.kill();
                let _ = child.wait();
                break None;
            }
        };
        match status.and_then(|s| s.code()) {
            Some(0) => {
                finished = true;
                break;
            }
            Some(MORE) => {}
            Some(STOP) => bail!("the warm could not go on (see above)"),
            Some(code) => println!("crashed      the batch ended with {code:#x}; the next one makes its pipelines in flight first, alone"),
            None => {}
        }
    }
    let done = warm::read_keys(&warm::list_path(&dir, &tag, "done"));
    let crashed = warm::read_keys(&warm::list_path(&dir, &tag, "crashed"));
    let mut by: BTreeMap<&str, usize> = BTreeMap::new();
    for word in done.values() {
        *by.entry(word.as_str()).or_default() += 1;
    }
    println!(
        "warm         {} in {:.1} s: {} pipelines done ({by:?}), {} crashed or hung the host compiler (listed in {})",
        if finished { "finished" } else { "stopped (rounds used up)" },
        started.elapsed().as_secs_f64(),
        done.len(),
        crashed.len(),
        warm::list_path(&dir, &tag, "crashed").display()
    );
    for (key, names) in &crashed {
        println!("  crashed    {key} [{names}]");
    }
    Ok(())
}

/// The pipeline in flight longest in the in-flight file, and for how long (s).
fn oldest_in_flight(path: &Path) -> Option<(String, u64)> {
    let now = unix_now();
    warm::read_keys(path)
        .into_iter()
        .filter_map(|(key, rest)| {
            let since: u64 = rest.rsplit('\t').next()?.parse().ok()?;
            Some((key, now.saturating_sub(since)))
        })
        .max_by_key(|(_, secs)| *secs)
}

/// One batch, in this process: the exit code for the supervisor.
pub fn child(a: Args) -> Result<i32> {
    let dir = a.library_dir.clone();
    victory_gpu::pipeline::use_pipeline_cache(dir.clone());
    if let Some(cache) = &a.shader_cache {
        victory_gpu::shader_cache::use_shader_cache(cache.clone());
    }
    let (tag, _) = tag()?;
    let opts = warm::Options {
        records: records(&a, &dir)?,
        shader_dirs: shader_dirs(&a, &dir),
        state_dir: dir.clone(),
        tag,
        threads: a.jobs.unwrap_or_else(|| std::thread::available_parallelism().map_or(4, |n| n.get())),
        batch: a.batch,
        allow_fresh: a.allow_fresh,
    };
    let report = match warm::batch(&opts) {
        Ok(r) => r,
        Err(e) => {
            println!("warm         {e}");
            return Ok(STOP);
        }
    };
    println!(
        "batch        {} records: {} done before, {} crashed before, {} suspects made alone; {} loaded, {} made, {} without a host pipeline, {} left; library opened from {} bytes{}",
        report.records,
        report.done_before,
        report.crashed_before,
        report.suspects,
        report.loaded,
        report.made,
        report.refused,
        report.left,
        report.library_bytes,
        if report.saved { ", saved" } else { ", NOT SAVED" }
    );
    for line in &report.newly_crashed {
        println!("  crashed    {line} (made alone when its batch died: skipped from now on)");
    }
    for (why, n) in &report.skipped {
        println!("  {n:>7}  {why}");
    }
    Ok(if !report.saved {
        STOP
    } else if report.left > 0 {
        MORE
    } else {
        0
    })
}
