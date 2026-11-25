//! Which console the title runs on. Titles pick settings (resolution,
//! memory budgets) by it, so the user chooses; Series X unless told.

use std::sync::atomic::{AtomicBool, Ordering};

const GIB: usize = 1 << 30;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum Console {
    #[default]
    SeriesX,
    SeriesS,
}

impl Console {
    /// Title memory in bytes: (GPU-optimal, standard). Series X in the
    /// advanced configuration; Series S has one range, counted as standard
    /// (7,776 MiB, the published console-mode figure).
    // shortcut: later system updates gave Series S titles more (amount not
    // published here); the title sizes nothing from this figure
    // (it goes by the device type), so update it if a title does.
    pub const fn title_memory(self) -> (usize, usize) {
        match self {
            Console::SeriesX => (10 * GIB, 7 * GIB / 2),
            Console::SeriesS => (0, 7_776 << 20),
        }
    }

    /// The GPU's pipe configuration, which the addresses of the XOR swizzle
    /// modes follow (textures the title's CPU reads or writes in one).
    /// `VICTORY_XOR_SWIZZLE=0`: one no XOR mode has a pattern for, so those
    /// modes are not addressed, as before (the other modes' patterns hold
    /// for every configuration).
    pub fn pipes(self) -> victory_formats::texture::Pipes {
        if !xor_swizzle() {
            return victory_formats::texture::Pipes {
                pipes_log2: 0,
                packers_log2: 0,
            };
        }
        match self {
            Console::SeriesX => victory_formats::texture::Pipes::SERIES_X,
            Console::SeriesS => victory_formats::texture::Pipes::SERIES_S,
        }
    }

    pub const fn name(self) -> &'static str {
        match self {
            Console::SeriesX => "Xbox Series X",
            Console::SeriesS => "Xbox Series S",
        }
    }
}

/// Whether textures in the pipe and bank XOR swizzle modes are addressed
/// (`VICTORY_XOR_SWIZZLE=0`: not, as before): the ocean height map the
/// title's CPU reads is in one, and a copy back of what the wave passes wrote
/// into it failed to tile ("a texture copied back could not be tiled
/// e=Unsupported"). From PastiiJay/klbf-lab ee7081b7, after MiamiGame/XTP
/// 00c60fb and 188d895 (patterns from Mesa's addrlib, MIT).
pub fn xor_swizzle() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_XOR_SWIZZLE").is_none_or(|v| v != "0"))
}

static SERIES_S: AtomicBool = AtomicBool::new(false);

/// Set before the title starts.
pub fn set(console: Console) {
    SERIES_S.store(console == Console::SeriesS, Ordering::Relaxed);
}

pub fn current() -> Console {
    if SERIES_S.load(Ordering::Relaxed) {
        Console::SeriesS
    } else {
        Console::SeriesX
    }
}
