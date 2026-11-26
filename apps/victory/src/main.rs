//! `victory`: the command-line front end.
//!
//! Every command reads the user's own files and writes only to stdout or to
//! the user's cache directory (outside the repository).

mod census;
mod inspect;
mod run;
mod shader;
mod translate;
#[cfg(windows)]
mod warm;

use std::num::NonZeroUsize;
use std::path::PathBuf;

use anyhow::Result;
use clap::{Parser, Subcommand, ValueEnum};

#[derive(Parser)]
#[command(
    name = "victory",
    version,
    about = "Run an Xbox Series title natively on Windows"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Summarise one PE image: headers, sections, imports, TLS, build id.
    Inspect {
        file: PathBuf,
        /// List every imported function.
        #[arg(long)]
        imports: bool,
        /// List every exported function.
        #[arg(long)]
        exports: bool,
    },
    /// Aggregate the imports of every PE image under the given paths and
    /// show which ones our runtime covers. Defaults to the configured
    /// game and test-title directories.
    Census {
        paths: Vec<PathBuf>,
        /// Do not count host pass-through as covered.
        #[arg(long)]
        strict: bool,
        /// List every missing `dll!function`, most used first.
        #[arg(long)]
        missing: bool,
    },
    /// List the `dll!export` pairs an import library describes.
    LibExports {
        lib: PathBuf,
        /// List every export, not just per-DLL counts.
        #[arg(long)]
        list: bool,
    },
    /// Find shader containers in files or directories (also inside engine
    /// archives, if uncompressed) and report their layout.
    ShaderInfo {
        paths: Vec<PathBuf>,
        /// Print one line per container found.
        #[arg(long)]
        verbose: bool,
        /// Also list the decoded native instructions (implies --verbose).
        #[arg(long)]
        disasm: bool,
    },
    /// Run every shader found (containers, SGD5 files) through the
    /// translator and report what it cannot handle yet.
    Translate {
        paths: Vec<PathBuf>,
        /// Check emitted DXIL with dxil.dll's validator (`VICTORY_DXIL`,
        /// else the newest Windows SDK's).
        #[arg(long)]
        validate: bool,
        /// Also create a host D3D12 pipeline for each emitted shader.
        #[arg(long)]
        pipelines: bool,
        /// Print one line per program and outcome.
        #[arg(long)]
        list: bool,
        /// Print one line per emitted shader with its size and a hash of
        /// its bytes: two runs (two settings of the `VICTORY_*`
        /// translation switches, or two builds) differ in exactly the
        /// programs whose host code changed (`tools/shader-sweep.sh`).
        #[arg(long)]
        digest: bool,
        /// Worker threads (default: every logical core).
        #[arg(long)]
        jobs: Option<NonZeroUsize>,
    },
    /// Make the host pipelines of recorded title pipelines with this
    /// process's translation switches (`VICTORY_*`) into a pipeline
    /// library, ahead of a run with those switches. Only while no game
    /// runs; the library's file is backed up first.
    #[cfg(windows)]
    WarmPipelines {
        /// The library's folder (holding host-pipelines.bin): a run's is
        /// `<cache>/pipelines`.
        #[arg(long)]
        library_dir: PathBuf,
        /// Pipeline record files (default: the folder's
        /// recorded-pipelines.txt, which every run writes).
        #[arg(long)]
        records: Vec<PathBuf>,
        /// Folders of saved shaders (`VICTORY_SHADER_SAVE`): each compute
        /// container there is a compute pipeline to make (they need no
        /// recorded state); also searched for containers.
        #[arg(long)]
        compute_from: Vec<PathBuf>,
        /// More folders with shader containers (`<hash>.<stage>.cdxil`);
        /// the folder's recorded-shaders is always searched.
        #[arg(long)]
        shaders: Vec<PathBuf>,
        /// Worker threads (default: every logical core).
        #[arg(long)]
        jobs: Option<NonZeroUsize>,
        /// Pipelines made per child process.
        #[arg(long, default_value_t = 400)]
        batch: usize,
        /// Seconds one pipeline may take before its batch is ended (it is
        /// then made alone, and skipped if it hangs or crashes again).
        #[arg(long, default_value_t = 300)]
        stall: u64,
        /// Child processes started at most.
        #[arg(long, default_value_t = 200)]
        rounds: usize,
        /// Do not back the library up first.
        #[arg(long)]
        no_backup: bool,
        /// Where the backup goes (default: a new folder beside the library's).
        #[arg(long)]
        backup_dir: Option<PathBuf>,
        /// A shader cache folder to translate through (a run's is
        /// `<cache>/shaders`; the same executable then shares its entries).
        #[arg(long)]
        shader_cache: Option<PathBuf>,
        /// Go on when the host refuses the library's file (the warm then
        /// replaces it with a new library).
        #[arg(long)]
        allow_fresh: bool,
        /// Go on while a victory.exe runs: only for a copy of a library
        /// (a file another process holds open is still refused).
        #[arg(long)]
        allow_running_game: bool,
        /// One batch in this process (the supervisor starts these).
        #[arg(long, hide = true)]
        child: bool,
    },
    /// Prepare, map and bind a title executable; with --execute, run it.
    Run {
        exe: PathBuf,
        /// What unimplemented imports do when called.
        #[arg(long, value_enum, default_value_t = Unimplemented::Trap)]
        unimplemented: Unimplemented,
        /// Disable host pass-through (everything must be implemented by us).
        #[arg(long)]
        strict: bool,
        /// Actually start the title. Without it, stop after binding.
        #[arg(long)]
        execute: bool,
        /// The console model the title sees (device type, memory).
        #[arg(long, value_enum, default_value_t = Console::SeriesX)]
        console: Console,
        /// Arguments for the title, after `--` (it reads the whole command
        /// line; a title that relaunches itself passes some).
        #[arg(last = true)]
        title_args: Vec<String>,
    },
    /// Show the resolved configuration and check it.
    Config,
}

#[derive(Clone, Copy, ValueEnum)]
enum Unimplemented {
    Trap,
    Zero,
}

#[derive(Clone, Copy, ValueEnum)]
enum Console {
    SeriesX,
    SeriesS,
}

fn main() -> Result<()> {
    victory_core::logging::init("info");
    // The audio recorder `run` starts for `VICTORY_AUDIO_CAPTURE`.
    #[cfg(all(windows, feature = "diagnostics"))]
    if victory_hle::modules::audio::capture::child() {
        return Ok(());
    }
    let cli = Cli::parse();
    match cli.command {
        Command::Inspect {
            file,
            imports,
            exports,
        } => inspect::run(&file, imports, exports),
        Command::Census {
            paths,
            strict,
            missing,
        } => census::run(paths, strict, missing),
        Command::LibExports { lib, list } => inspect::lib_exports(&lib, list),
        Command::ShaderInfo {
            paths,
            verbose,
            disasm,
        } => shader::run(paths, verbose || disasm, disasm),
        Command::Translate {
            paths,
            validate,
            pipelines,
            list,
            digest,
            jobs,
        } => {
            // `VICTORY_VIEW_LIMIT=<null slot>`: translate as on a host whose view heap is capped
            if let Some(n) = std::env::var("VICTORY_VIEW_LIMIT").ok().and_then(|v| v.parse().ok()) {
                victory_shader::emit::set_view_heap_limit(n);
            }
            // `VICTORY_HOST_WAVE=32`: translate as for a GPU whose waves are at most 32 lanes
            if let Some(n) = std::env::var("VICTORY_HOST_WAVE").ok().and_then(|v| v.parse().ok()) {
                victory_shader::emit::set_host_max_wave(n);
            }
            translate::run(
                paths,
                validate,
                pipelines,
                list,
                digest,
                jobs.map(NonZeroUsize::get),
            )
        }
        #[cfg(windows)]
        Command::WarmPipelines {
            library_dir,
            records,
            compute_from,
            shaders,
            jobs,
            batch,
            stall,
            rounds,
            no_backup,
            backup_dir,
            shader_cache,
            allow_fresh,
            allow_running_game,
            child,
        } => {
            let args = warm::Args {
                library_dir,
                records,
                compute_from,
                shaders,
                jobs: jobs.map(NonZeroUsize::get),
                batch,
                stall,
                no_backup,
                backup_dir,
                shader_cache,
                allow_fresh,
                rounds,
                allow_running_game,
            };
            if child {
                use std::io::Write;
                let code = warm::child(args)?;
                let _ = std::io::stdout().flush();
                std::process::exit(code);
            }
            warm::run(args)
        }
        Command::Run {
            exe,
            unimplemented,
            strict,
            execute,
            console,
            title_args: _,
        } => run::run(&exe, unimplemented, strict, execute, console),
        Command::Config => show_config(),
    }
}

fn show_config() -> Result<()> {
    let cfg = victory_core::Config::load()?;
    let status = |p: &Option<PathBuf>| match p {
        Some(p) if p.exists() => format!("{}", p.display()),
        Some(p) => format!("{}  (does not exist)", p.display()),
        None => "(not set)".to_owned(),
    };
    println!(
        "config file : {}",
        cfg.source
            .as_ref()
            .map_or("(none)".into(), |p| p.display().to_string())
    );
    println!("game_dir    : {}", status(&cfg.game_dir));
    println!("cache_dir   : {}", cfg.cache_dir.display());
    println!("data_dir    : {}", cfg.data_dir.display());
    for t in &cfg.test_titles {
        println!("test_title  : {}", status(&Some(t.clone())));
    }
    for p in cfg
        .game_dir
        .iter()
        .chain(&cfg.test_titles)
        .chain([&cfg.cache_dir, &cfg.data_dir])
    {
        if let Err(e) = victory_core::ensure_outside_repo(p) {
            println!("POLICY VIOLATION: {e}");
        }
    }
    Ok(())
}

/// Paths given on the command line, or the configured game + test titles.
pub(crate) fn default_paths(paths: Vec<PathBuf>) -> Result<Vec<PathBuf>> {
    if !paths.is_empty() {
        return Ok(paths);
    }
    let cfg = victory_core::Config::load()?;
    let all: Vec<PathBuf> = cfg.game_dir.into_iter().chain(cfg.test_titles).collect();
    anyhow::ensure!(
        !all.is_empty(),
        "no paths given and none configured (see victory.example.toml)"
    );
    Ok(all)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn translate_requires_a_positive_worker_count() {
        let Err(error) = Cli::try_parse_from(["victory", "translate", "--jobs", "0"]) else {
            panic!("zero workers should be rejected");
        };
        assert_eq!(error.kind(), clap::error::ErrorKind::ValueValidation);

        let cli = Cli::try_parse_from(["victory", "translate", "--jobs", "1"]).unwrap();
        assert!(
            matches!(cli.command, Command::Translate { jobs, .. } if jobs == NonZeroUsize::new(1))
        );
    }

    #[test]
    fn translate_takes_the_digest_flag_and_leaves_it_off_by_default() {
        let cli = Cli::try_parse_from(["victory", "translate", "--digest", "a"]).unwrap();
        assert!(matches!(
            cli.command,
            Command::Translate {
                digest: true,
                list: false,
                ..
            }
        ));
        let cli = Cli::try_parse_from(["victory", "translate", "a"]).unwrap();
        assert!(matches!(
            cli.command,
            Command::Translate { digest: false, .. }
        ));
    }
}
