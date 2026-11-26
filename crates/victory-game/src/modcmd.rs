//! `VICTORY_MOD`: the mod commands, the part that touches nothing of the
//! title (so it is tested anywhere): the one parser both inputs use (the
//! mod file `VICTORY_MOD_FILE` and, with `VICTORY_TITLE_COMMANDS=1`, the
//! keys file's lines), the reader that tails the mod file, the status
//! file's text and atomic write, and the far teleport's steps (over a
//! [`World`] the title side gives, so they run here with a fake).
//!
//! The title side is `modgame` (Windows only): it runs each command on the
//! frame hook's own thread.
//!
//! Recipes after viper-community/viper e44573c6e (`crates/viper/src/game.rs`
//! `teleport_to_waypoint` / `finish_teleport`, `crates/viper-layout/src/
//! game.rs` WEATHER_NAMES, `crates/viper-rt/src/command.rs` argument slots;
//! LGPL-2.1-or-later), written again here.

use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

/// The words that begin a mod command (the keys file hands lines that begin
/// with one of these to the mod when `VICTORY_MOD=1`).
pub const WORDS: &[&str] =
    &["teleport", "tp", "time", "weather", "heal", "invincible", "wanted", "spawn", "vehicle", "native", "status"];

/// Longest command line taken (longer ones are refused).
pub const MOST_LINE: usize = 400;
/// Most `native` arguments (slots are counted separately, see [`Arg::slots`]).
pub const MOST_ARGS: usize = 16;
/// Most argument slots of one `native` call (VIper `command::MAX_ARGS`).
pub const MOST_SLOTS: usize = 32;
/// The game's highest wanted level (VIper `WANTED_MAX_LEVEL`).
pub const WANTED_MAX: u32 = 6;
/// Coordinates beyond this (metres) are refused as typing errors.
pub const FAR_LIMIT: f32 = 100_000.0;

/// One mod command.
#[derive(Clone, Debug, PartialEq)]
pub enum Command {
    /// `teleport x y [z [heading]]` (also `tp ...`): without z the ground is found.
    Teleport { x: f32, y: f32, z: Option<f32>, heading: Option<f32> },
    /// `tp rel <forward m> <up m>`: along the heading, and up.
    Relative { forward: f32, up: f32 },
    /// `tp waypoint`: the map waypoint, on the ground.
    Waypoint,
    /// `time <hh> <mm>`.
    Time { hour: u32, minute: u32 },
    /// `time freeze on|off`.
    Freeze(bool),
    /// `weather <name|0xhash>`: held until `weather cycle`.
    Weather { hash: u32, name: String },
    /// `weather cycle` (or `clear-hold`): back to the game's own weather.
    WeatherCycle,
    /// `heal`.
    Heal,
    /// `invincible on|off`.
    Invincible(bool),
    /// `wanted <0-6>`.
    Wanted(u32),
    /// `wanted never on|off`.
    NeverWanted(bool),
    /// `spawn vehicle <name|0xhash>`.
    SpawnVehicle(Model),
    /// `vehicle repair`.
    Repair,
    /// `vehicle list`: the game's vehicle models, to a file beside the status file.
    VehicleList,
    /// `vehicle enter`: the player into the driver's seat of the vehicle last spawned.
    Enter,
    /// `vehicle exit`: the player gets out (the game's leave-vehicle task).
    Exit,
    /// `native <target> [args...] [r:<n>]`: `results` slots asked for (0 to 8, default 3).
    Native { target: Target, args: Vec<Arg>, results: u32 },
    /// `status`: write the status file now.
    Status,
}

/// A model: a name (searched among the game's vehicle models), its 64-bit
/// hash, or `#n`, the n-th line of `vehicle list`.
#[derive(Clone, Debug, PartialEq)]
pub enum Model {
    Name(String),
    Hash(u64),
    Index(usize),
}

/// Which script-command handler `native` calls.
#[derive(Clone, Debug, PartialEq)]
pub enum Target {
    /// An RVA in the title (a value at or above 0x1_4000_0000 is taken as an
    /// address at the preferred base and turned into an RVA).
    Rva(u32),
    /// `hash:0x...`: the handler whose self-stored command hash is this.
    Hash(u64),
    /// A name from the verified table.
    Name(String),
}

/// One typed `native` argument.
#[derive(Clone, Debug, PartialEq)]
pub enum Arg {
    /// `i:<int>` (low 32 bits of the slot).
    Int(i32),
    /// `u:<u64>` (decimal or 0x hex, the whole slot).
    U64(u64),
    /// `f:<float>`.
    Float(f32),
    /// `b:<0|1|true|false>`.
    Bool(bool),
    /// `s:<text>` or `s:"two words"`: a pointer to a NUL-terminated UTF-8 copy.
    Str(String),
    /// `v:<x>,<y>,<z>`: a vector, two slots (x | y << 32, then z).
    Vec3([f32; 3]),
    /// `h:player`: the player ped's 16-byte script handle, two slots.
    Player,
    /// `h:vehicle`: the player's vehicle's handle (zero when on foot), two slots.
    Vehicle,
    /// `h:none`: the zero handle ("no entity"), two slots.
    NoEntity,
}

impl Arg {
    /// How many 8-byte slots it takes.
    pub fn slots(&self) -> usize {
        match self {
            Arg::Vec3(_) | Arg::Player | Arg::Vehicle | Arg::NoEntity => 2,
            _ => 1,
        }
    }
}

/// One-at-a-time ("joaat") of the lower-cased text: the title's name hash
/// for weather types, weapons and items (VIper `hash::joaat`).
pub fn joaat(text: &str) -> u32 {
    let mut h: u32 = 0;
    for c in text.bytes().map(|c| c.to_ascii_lowercase()) {
        h = h.wrapping_add(u32::from(c));
        h = h.wrapping_add(h << 10);
        h ^= h >> 6;
    }
    h = h.wrapping_add(h << 3);
    h ^= h >> 11;
    h.wrapping_add(h << 15)
}

/// The weather types by name and hash (VIper `WEATHER_NAMES`; "Dark
/// overcast"'s hash is not the joaat of any spelling we tried, so it is
/// listed as VIper has it).
pub const WEATHERS: [(&str, u32); 15] = [
    ("sunny", 0x614A_1F91),
    ("clear", 0x36A8_3D84),
    ("misty", 0x5974_E8E5),
    ("fog", 0xD61B_DE01),
    ("clouds", 0x30FD_AF5C),
    ("overcast", 0xBB89_8D2D),
    ("darkovercast", 0x19D4_F1D9),
    ("drizzle", 0x995C_7F44),
    ("shower", 0xE726_79D5),
    ("rain", 0x54A6_9840),
    ("thunder", 0xB677_829F),
    ("thunderstorm", 0x7C1C_4A13),
    ("hurricane", 0x320D_0951),
    ("hail", 0x75A9_E268),
    ("sandstorm", 0xB17F_6111),
];

/// The table's name of a weather hash.
pub fn weather_name(hash: u32) -> Option<&'static str> {
    WEATHERS.iter().find(|w| w.1 == hash).map(|w| w.0)
}

fn number(w: &str) -> Result<f32, String> {
    w.parse::<f32>()
        .ok()
        .filter(|v| v.is_finite() && v.abs() <= FAR_LIMIT)
        .ok_or_else(|| format!("{w:?} is not a number (finite, at most {FAR_LIMIT} in size)"))
}

fn on_off(w: Option<&str>) -> Result<bool, String> {
    match w {
        Some("on" | "1" | "true") => Ok(true),
        Some("off" | "0" | "false") => Ok(false),
        _ => Err("wants on or off".into()),
    }
}

fn hex_u64(w: &str) -> Option<u64> {
    match w.strip_prefix("0x").or_else(|| w.strip_prefix("0X")) {
        Some(h) => u64::from_str_radix(&h.replace('_', ""), 16).ok(),
        None => w.parse::<u64>().ok(),
    }
}

/// Splits on spaces, keeping `"..."` together (the quotes removed, also
/// after a `s:` prefix).
fn words(line: &str) -> Result<Vec<String>, String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut quoted = false;
    let mut any = false;
    for c in line.chars() {
        match c {
            '"' => {
                quoted = !quoted;
                any = true;
            }
            c if c.is_whitespace() && !quoted => {
                if any {
                    out.push(std::mem::take(&mut cur));
                    any = false;
                }
            }
            c => {
                cur.push(c);
                any = true;
            }
        }
    }
    if quoted {
        return Err("a quote is not closed".into());
    }
    if any {
        out.push(cur);
    }
    Ok(out)
}

fn arg(w: &str) -> Result<Arg, String> {
    let (kind, v) = w.split_once(':').ok_or_else(|| format!("argument {w:?} wants a type: i: u: f: b: s: v: or h:"))?;
    let bad = || format!("argument {w:?} is not a valid {kind}: value");
    Ok(match kind {
        "i" => Arg::Int(match v.strip_prefix("0x") {
            Some(h) => u32::from_str_radix(h, 16).map_err(|_| bad())? as i32,
            None => v.parse::<i32>().map_err(|_| bad())?,
        }),
        "u" => Arg::U64(hex_u64(v).ok_or_else(bad)?),
        "f" => Arg::Float(v.parse::<f32>().ok().filter(|f| f.is_finite()).ok_or_else(bad)?),
        "b" => Arg::Bool(match v {
            "1" | "true" | "on" => true,
            "0" | "false" | "off" => false,
            _ => return Err(bad()),
        }),
        "s" => {
            if v.contains('\0') {
                return Err(bad());
            }
            Arg::Str(v.to_owned())
        }
        "v" => {
            let n: Vec<f32> = v.split(',').map(|x| x.trim().parse::<f32>().ok().filter(|f| f.is_finite())).collect::<Option<_>>().ok_or_else(bad)?;
            match n[..] {
                [x, y, z] => Arg::Vec3([x, y, z]),
                _ => return Err(bad()),
            }
        }
        "h" => match v {
            "player" => Arg::Player,
            "vehicle" => Arg::Vehicle,
            "none" => Arg::NoEntity,
            _ => return Err(format!("argument {w:?}: h: takes player, vehicle or none")),
        },
        _ => return Err(format!("argument {w:?}: unknown type {kind}: (i: u: f: b: s: v: h:)")),
    })
}

/// Parses one command line (`#` comments and blank lines are `Ok(None)`).
pub fn parse(line: &str) -> Result<Option<Command>, String> {
    let line = line.trim();
    if line.is_empty() || line.starts_with('#') {
        return Ok(None);
    }
    if line.len() > MOST_LINE {
        return Err(format!("the line is longer than {MOST_LINE} characters"));
    }
    let w = words(line)?;
    let w: Vec<&str> = w.iter().map(String::as_str).collect();
    let lower = w[0].to_ascii_lowercase();
    let rest = &w[1..];
    let cmd = match (lower.as_str(), rest) {
        ("teleport" | "tp", ["rel", f, u]) => Command::Relative { forward: number(f)?, up: number(u)? },
        ("teleport" | "tp", ["rel", ..]) => return Err("tp rel wants <forward m> <up m>".into()),
        ("teleport" | "tp", ["waypoint"]) => Command::Waypoint,
        ("teleport" | "tp", n) if (2..=4).contains(&n.len()) => {
            let v = n.iter().map(|w| number(w)).collect::<Result<Vec<f32>, _>>()?;
            Command::Teleport { x: v[0], y: v[1], z: v.get(2).copied(), heading: v.get(3).copied() }
        }
        ("teleport" | "tp", _) => return Err("teleport wants x y [z [heading]], rel <forward> <up>, or waypoint".into()),
        ("time", ["freeze", v]) => Command::Freeze(on_off(Some(v))?),
        ("time", [h, m]) => {
            let (hour, minute) = (h.parse::<u32>().ok().filter(|&h| h < 24), m.parse::<u32>().ok().filter(|&m| m < 60));
            match (hour, minute) {
                (Some(hour), Some(minute)) => Command::Time { hour, minute },
                _ => return Err("time wants <hour 0-23> <minute 0-59>".into()),
            }
        }
        ("time", _) => return Err("time wants <hh> <mm> or freeze on|off".into()),
        ("weather", ["cycle" | "clear-hold" | "natural"]) => Command::WeatherCycle,
        ("weather", [name, more @ ..]) => {
            let joined = std::iter::once(*name).chain(more.iter().copied()).collect::<Vec<_>>().join("");
            let key: String = joined.chars().filter(|c| *c != '_' && *c != '-').collect::<String>().to_ascii_lowercase();
            match (WEATHERS.iter().find(|w| w.0 == key), hex_u64(name).filter(|_| name.starts_with("0x"))) {
                (Some(&(n, h)), _) => Command::Weather { hash: h, name: n.into() },
                (None, Some(h)) if more.is_empty() && h <= u64::from(u32::MAX) => {
                    Command::Weather { hash: h as u32, name: weather_name(h as u32).unwrap_or("?").into() }
                }
                (None, _) if more.is_empty() => Command::Weather { hash: joaat(name), name: name.to_ascii_lowercase() },
                _ => return Err(format!("unknown weather {joined:?}")),
            }
        }
        ("weather", _) => return Err("weather wants a name (sunny, clear, rain, ...) or cycle".into()),
        ("heal", []) => Command::Heal,
        ("invincible", [v]) => Command::Invincible(on_off(Some(v))?),
        ("invincible", _) => return Err("invincible wants on or off".into()),
        ("wanted", ["never", v]) => Command::NeverWanted(on_off(Some(v))?),
        ("wanted", [n]) => Command::Wanted(
            n.parse::<u32>().ok().filter(|&n| n <= WANTED_MAX).ok_or_else(|| format!("wanted wants 0 to {WANTED_MAX} or never on|off"))?,
        ),
        ("wanted", _) => return Err(format!("wanted wants 0 to {WANTED_MAX} or never on|off")),
        ("spawn", ["vehicle", m]) => Command::SpawnVehicle(match (hex_u64(m).filter(|_| m.starts_with("0x")), m.strip_prefix('#')) {
            (Some(h), _) => Model::Hash(h),
            (None, Some(n)) => Model::Index(n.parse::<usize>().map_err(|_| format!("{m:?}: # wants a line number of vehicle list"))?),
            (None, None) => Model::Name(m.to_ascii_lowercase()),
        }),
        ("spawn", _) => return Err("spawn wants vehicle <name|0xhash>".into()),
        ("vehicle", ["repair"]) => Command::Repair,
        ("vehicle", ["list"]) => Command::VehicleList,
        ("vehicle", ["enter"]) => Command::Enter,
        ("vehicle", ["exit"]) => Command::Exit,
        ("vehicle", _) => return Err("vehicle wants repair, list, enter or exit".into()),
        ("native", [t, args @ ..]) => {
            if args.len() > MOST_ARGS {
                return Err(format!("native takes at most {MOST_ARGS} arguments"));
            }
            let target = if let Some(h) = t.strip_prefix("hash:") {
                Target::Hash(hex_u64(h).ok_or_else(|| format!("{t:?}: hash: wants a 64-bit number"))?)
            } else if t.starts_with("0x") || t.starts_with("0X") {
                let a = hex_u64(t).ok_or_else(|| format!("{t:?} is not a hex address"))?;
                let rva = if a >= 0x1_4000_0000 { a - 0x1_4000_0000 } else { a };
                Target::Rva(u32::try_from(rva).map_err(|_| format!("{t:?} is outside the title"))?)
            } else {
                Target::Name(t.to_ascii_uppercase())
            };
            let mut results = 3;
            let mut typed = Vec::new();
            for a in args {
                match a.strip_prefix("r:") {
                    Some(n) => results = n.parse::<u32>().ok().filter(|&n| n <= 8).ok_or_else(|| format!("{a:?}: r: wants 0 to 8 result slots"))?,
                    None => typed.push(arg(a)?),
                }
            }
            if typed.iter().map(Arg::slots).sum::<usize>() > MOST_SLOTS {
                return Err(format!("the arguments take more than {MOST_SLOTS} slots"));
            }
            Command::Native { target, args: typed, results }
        }
        ("native", _) => return Err("native wants <0xrva|hash:0x...|NAME> [i: u: f: b: s: v: h: arguments] [r:<result slots>]".into()),
        ("status", []) => Command::Status,
        (other, _) if WORDS.contains(&other) => return Err(format!("{other}: wrong arguments")),
        (other, _) => return Err(format!("unknown command {other:?}")),
    };
    Ok(Some(cmd))
}

/// A target along a heading (radians, 0 north, counter-clockwise; the
/// forward direction is (-sin h, cos h), as the title's forward row).
pub fn relative(from: [f32; 3], heading: f32, forward: f32, up: f32) -> [f32; 3] {
    [from[0] - heading.sin() * forward, from[1] + heading.cos() * forward, from[2] + up]
}

// ---- the mod file -------------------------------------------------------------

/// Reads the lines appended to a file since the last call: whole lines
/// only (a partial last line waits for its newline), from where the file
/// ended when it was first seen (lines written before the game ran are not
/// run), starting again from the top when the file got shorter (a new one).
pub struct Tail {
    path: PathBuf,
    offset: Option<u64>,
    partial: Vec<u8>,
}

impl Tail {
    pub fn new(path: impl Into<PathBuf>) -> Tail {
        Tail { path: path.into(), offset: None, partial: Vec::new() }
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// The new whole lines (at most 64 KiB read a call); nothing when the
    /// file is missing or unchanged.
    pub fn poll(&mut self) -> Vec<String> {
        let Ok(len) = std::fs::metadata(&self.path).map(|m| m.len()) else {
            return Vec::new();
        };
        let Some(at) = self.offset else {
            self.offset = Some(len);
            return Vec::new();
        };
        let at = if len < at {
            self.partial.clear();
            0
        } else {
            at
        };
        if len == at {
            self.offset = Some(at);
            return Vec::new();
        }
        let want = (len - at).min(64 * 1024);
        let mut buf = vec![0u8; want as usize];
        let read = std::fs::File::open(&self.path).and_then(|mut f| {
            f.seek(SeekFrom::Start(at))?;
            let mut n = 0;
            while n < buf.len() {
                match f.read(&mut buf[n..])? {
                    0 => break,
                    k => n += k,
                }
            }
            Ok(n)
        });
        let Ok(n) = read else { return Vec::new() };
        self.offset = Some(at + n as u64);
        self.partial.extend_from_slice(&buf[..n]);
        let mut lines = Vec::new();
        while let Some(i) = self.partial.iter().position(|&b| b == b'\n') {
            let line: Vec<u8> = self.partial.drain(..=i).collect();
            let text = String::from_utf8_lossy(&line);
            let text = text.trim_end_matches(['\n', '\r']).trim();
            if !text.is_empty() {
                lines.push(text.to_owned());
            }
        }
        if self.partial.len() > MOST_LINE * 4 {
            // A "line" this long never ends: drop it.
            self.partial.clear();
        }
        lines
    }
}

// ---- the status file ----------------------------------------------------------

/// `key=value` lines (a value's newlines become spaces).
#[derive(Default)]
pub struct StatusText(String);

impl StatusText {
    pub fn put(&mut self, key: &str, value: impl std::fmt::Display) -> &mut Self {
        let v = value.to_string().replace(['\r', '\n'], " ");
        self.0.push_str(key);
        self.0.push('=');
        self.0.push_str(&v);
        self.0.push('\n');
        self
    }

    pub fn text(&self) -> &str {
        &self.0
    }
}

/// Writes `text` to `path` through a temporary file beside it and a
/// rename, so a reader sees the old file or the new one, never half.
pub fn write_atomic(path: &Path, text: &str) -> std::io::Result<()> {
    let mut tmp = path.as_os_str().to_owned();
    tmp.push(".tmp");
    let tmp = PathBuf::from(tmp);
    std::fs::write(&tmp, text)?;
    std::fs::rename(&tmp, path)
}

// ---- the far teleport -----------------------------------------------------------

/// What the far teleport needs of the title (on the frame hook's thread).
pub trait World {
    /// The ground under (x, y), probing down from `from_z`, against loaded collision.
    fn ground(&mut self, x: f32, y: f32, from_z: f32) -> Option<f32>;
    /// The terrain height map at (x, y) (there also where collision is not loaded).
    fn terrain(&mut self, x: f32, y: f32) -> Option<f32>;
    /// Asks for the collision around a point (a one-shot request).
    fn want_collision(&mut self, at: [f32; 3]);
    /// Moves the player (or the vehicle with it) there, with warp, and turns it.
    fn place(&mut self, at: [f32; 3], heading: Option<f32>) -> Result<(), String>;
}

/// The player is held this long, at most, for the ground to load.
pub const WARP_WAIT_MS: u64 = 12_000;
/// Held this far over the height map when no z is given (VIper: 10 m).
pub const ABOVE_TERRAIN: f32 = 10.0;
/// Held this high without a height map there.
pub const NO_TERRAIN_HIGH: f32 = 1000.0;
/// Set this far over the ground found (VIper: 1 m).
pub const OVER_GROUND: f32 = 1.0;

/// A teleport under way: the player held at the target until its ground
/// is loaded, then set down.
#[derive(Clone, Debug, PartialEq)]
pub struct Warp {
    pub x: f32,
    pub y: f32,
    pub z: Option<f32>,
    pub heading: Option<f32>,
    /// Where it is held meanwhile.
    pub hold_z: f32,
    pub since_ms: u64,
    pub frames: u32,
}

/// One step's outcome.
#[derive(Clone, Debug, PartialEq)]
pub enum WarpStep {
    /// Set down at this point after `frames` frames.
    Done { at: [f32; 3], frames: u32, ground: Option<f32> },
    /// Still held.
    Holding,
    /// No ground in time: set down at this point (`why` says how it was chosen).
    TimedOut { at: [f32; 3], why: &'static str },
    /// The title refused to move the player.
    Failed(String),
}

impl Warp {
    /// Starts a teleport: moves the player at once to the hold point and
    /// takes the first step in the same frame (a near target whose ground
    /// is loaded ends here).
    pub fn start(w: &mut dyn World, x: f32, y: f32, z: Option<f32>, heading: Option<f32>, now_ms: u64) -> (Warp, WarpStep) {
        let hold_z = z.unwrap_or_else(|| w.terrain(x, y).map_or(NO_TERRAIN_HIGH, |h| h + ABOVE_TERRAIN));
        let mut warp = Warp { x, y, z, heading, hold_z, since_ms: now_ms, frames: 0 };
        if let Err(e) = w.place([x, y, hold_z], heading) {
            return (warp, WarpStep::Failed(e));
        }
        let step = warp.step(w, now_ms);
        (warp, step)
    }

    /// One frame: ask for the collision, probe for the ground; set down
    /// when found, else hold (put back at the hold point) until
    /// [`WARP_WAIT_MS`].
    pub fn step(&mut self, w: &mut dyn World, now_ms: u64) -> WarpStep {
        self.frames += 1;
        w.want_collision([self.x, self.y, self.hold_z]);
        let from = self.z.map_or(self.hold_z + 50.0, |z| z + 2.0);
        let found = w.ground(self.x, self.y, from);
        let (at, outcome) = if let Some(g) = found {
            let at = [self.x, self.y, self.z.unwrap_or(g + OVER_GROUND)];
            (at, WarpStep::Done { at, frames: self.frames, ground: Some(g) })
        } else if now_ms.saturating_sub(self.since_ms) >= WARP_WAIT_MS {
            let (z, why) = match (self.z, w.terrain(self.x, self.y)) {
                (Some(z), _) => (z, "no ground loaded in time: set down at the z asked"),
                (None, Some(t)) => (t + OVER_GROUND, "no ground loaded in time: set down on the height map"),
                (None, None) => (self.hold_z, "no ground loaded and no height map: left at the hold height"),
            };
            let at = [self.x, self.y, z];
            (at, WarpStep::TimedOut { at, why })
        } else {
            // Hold: put back where it waits (no fall, no drift).
            if let Err(e) = w.place([self.x, self.y, self.hold_z], None) {
                return WarpStep::Failed(e);
            }
            return WarpStep::Holding;
        };
        match w.place(at, self.heading) {
            Ok(()) => outcome,
            Err(e) => WarpStep::Failed(e),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn p(line: &str) -> Command {
        parse(line).unwrap_or_else(|e| panic!("{line:?}: {e}")).unwrap()
    }

    #[test]
    fn teleport_forms() {
        assert_eq!(p("teleport -2353.31 -5534.35 7.0"), Command::Teleport { x: -2353.31, y: -5534.35, z: Some(7.0), heading: None });
        assert_eq!(p("tp 1 2 3 2.17"), Command::Teleport { x: 1.0, y: 2.0, z: Some(3.0), heading: Some(2.17) });
        assert_eq!(p("  TP 1 2 "), Command::Teleport { x: 1.0, y: 2.0, z: None, heading: None });
        assert_eq!(p("tp rel 5 0"), Command::Relative { forward: 5.0, up: 0.0 });
        assert_eq!(p("teleport waypoint"), Command::Waypoint);
        for bad in ["tp", "tp 1", "tp 1 2 3 4 5", "tp x 2", "tp 1 NaN", "tp 1 inf", "tp 200000 0", "tp rel 5", "tp rel a b"] {
            assert!(parse(bad).is_err(), "{bad:?}");
        }
        assert_eq!(parse("# a note"), Ok(None));
        assert_eq!(parse("   "), Ok(None));
    }

    #[test]
    fn world_and_player_forms() {
        assert_eq!(p("time 6 30"), Command::Time { hour: 6, minute: 30 });
        assert_eq!(p("time freeze on"), Command::Freeze(true));
        assert_eq!(p("time freeze off"), Command::Freeze(false));
        for bad in ["time 24 0", "time 6 60", "time 6", "time freeze", "time freeze maybe"] {
            assert!(parse(bad).is_err(), "{bad:?}");
        }
        assert_eq!(p("weather rain"), Command::Weather { hash: 0x54A6_9840, name: "rain".into() });
        assert_eq!(p("weather Dark Overcast"), Command::Weather { hash: 0x19D4_F1D9, name: "darkovercast".into() });
        assert_eq!(p("weather dark_overcast"), Command::Weather { hash: 0x19D4_F1D9, name: "darkovercast".into() });
        assert_eq!(p("weather 0x36a83d84"), Command::Weather { hash: 0x36A8_3D84, name: "clear".into() });
        assert_eq!(p("weather xmas"), Command::Weather { hash: joaat("xmas"), name: "xmas".into() });
        assert_eq!(p("weather cycle"), Command::WeatherCycle);
        assert_eq!(p("heal"), Command::Heal);
        assert_eq!(p("invincible on"), Command::Invincible(true));
        assert_eq!(p("wanted 0"), Command::Wanted(0));
        assert_eq!(p("wanted 6"), Command::Wanted(6));
        assert!(parse("wanted 7").is_err());
        assert_eq!(p("wanted never on"), Command::NeverWanted(true));
        assert_eq!(p("spawn vehicle Infernus"), Command::SpawnVehicle(Model::Name("infernus".into())));
        assert_eq!(p("spawn vehicle 0x1234abcd00000001"), Command::SpawnVehicle(Model::Hash(0x1234_abcd_0000_0001)));
        assert_eq!(p("vehicle repair"), Command::Repair);
        assert_eq!(p("vehicle list"), Command::VehicleList);
        assert_eq!(p("vehicle enter"), Command::Enter);
        assert_eq!(p("vehicle exit"), Command::Exit);
        assert_eq!(p("spawn vehicle #12"), Command::SpawnVehicle(Model::Index(12)));
        assert!(parse("spawn vehicle #x").is_err());
        assert!(parse("vehicle fly").is_err());
        assert_eq!(p("status"), Command::Status);
        assert!(parse("fly 1").unwrap_err().contains("unknown"));
        assert!(parse("heal now").unwrap_err().contains("wrong arguments"));
    }

    #[test]
    fn native_forms_and_slots() {
        let c = p(r#"native 0x3cd69a0 i:6 i:30 i:0 b:0"#);
        assert_eq!(c, Command::Native { target: Target::Rva(0x3cd_69a0), args: vec![Arg::Int(6), Arg::Int(30), Arg::Int(0), Arg::Bool(false)], results: 3 });
        assert_eq!(p("native 0x143cd69a0").clone(), Command::Native { target: Target::Rva(0x3cd_69a0), args: vec![], results: 3 });
        assert_eq!(p("native hash:0x6471F11FF69D8A77"), Command::Native { target: Target::Hash(0x6471_F11F_F69D_8A77), args: vec![], results: 3 });
        assert_eq!(p("native get_entity_coords h:player b:0"), Command::Native { target: Target::Name("GET_ENTITY_COORDS".into()), args: vec![Arg::Player, Arg::Bool(false)], results: 3 });
        assert_eq!(
            p(r#"native 0x10 s:"two words" f:1.5 v:1,2,3 u:0xff i:-3 h:vehicle"#),
            Command::Native {
                target: Target::Rva(0x10),
                args: vec![Arg::Str("two words".into()), Arg::Float(1.5), Arg::Vec3([1.0, 2.0, 3.0]), Arg::U64(255), Arg::Int(-3), Arg::Vehicle],
                results: 3
            }
        );
        assert_eq!(
            p("native SET_ENTITY_HEALTH h:player i:200 h:none b:0 b:0 r:0"),
            Command::Native {
                target: Target::Name("SET_ENTITY_HEALTH".into()),
                args: vec![Arg::Player, Arg::Int(200), Arg::NoEntity, Arg::Bool(false), Arg::Bool(false)],
                results: 0
            }
        );
        for bad in ["native", "native 0x10 7", "native 0x10 x:1", "native 0x10 f:nan", "native 0x10 v:1,2", "native 0x10 h:cat", r#"native 0x10 s:"open"#, "native 0x10 r:9"] {
            assert!(parse(bad).is_err(), "{bad:?}");
        }
        let many = format!("native 0x10 {}", ["v:1,2,3"; 16].join(" "));
        assert!(parse(&many).is_ok(), "32 slots fit");
        let too = format!("native 0x10 {} i:1", ["v:1,2,3"; 16].join(" "));
        assert!(parse(&too).is_err(), "17 arguments");
    }

    #[test]
    fn joaat_and_relative() {
        assert_eq!(joaat("WEAPON_PISTOL"), 0x1B06_D571);
        assert_eq!(joaat("rain"), 0x54A6_9840);
        for &(n, h) in &WEATHERS {
            if n != "darkovercast" {
                assert_eq!(joaat(n), h, "{n}");
            }
        }
        let t = relative([10.0, 20.0, 3.0], 0.0, 5.0, 1.0);
        assert!((t[0] - 10.0).abs() < 1e-4 && (t[1] - 25.0).abs() < 1e-4 && (t[2] - 4.0).abs() < 1e-4, "north: +y");
        let t = relative([0.0, 0.0, 0.0], std::f32::consts::FRAC_PI_2, 5.0, 0.0);
        assert!((t[0] + 5.0).abs() < 1e-4 && t[1].abs() < 1e-4, "heading pi/2 faces west: -x");
    }

    #[test]
    fn the_tail_reads_only_new_whole_lines() {
        let dir = std::env::temp_dir().join(format!("victory-modtail-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let f = dir.join("cmd.txt");
        let _ = std::fs::remove_file(&f);
        let mut t = Tail::new(&f);
        assert!(t.poll().is_empty(), "no file yet");
        std::fs::write(&f, "heal\n").unwrap();
        // First seen: what was there before is not run.
        assert!(t.poll().is_empty());
        let add = |s: &str| {
            use std::io::Write;
            std::fs::OpenOptions::new().append(true).open(&f).unwrap().write_all(s.as_bytes()).unwrap();
        };
        add("tp 1 2\r\ntime 6");
        assert_eq!(t.poll(), vec!["tp 1 2".to_owned()], "the partial line waits");
        assert!(t.poll().is_empty());
        add(" 30\n\n");
        assert_eq!(t.poll(), vec!["time 6 30".to_owned()]);
        // A new, shorter file: read from its top.
        std::fs::write(&f, "heal\n").unwrap();
        assert_eq!(t.poll(), vec!["heal".to_owned()]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_status_file_is_replaced_whole() {
        let dir = std::env::temp_dir().join(format!("victory-modstatus-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let f = dir.join("status.txt");
        let mut s = StatusText::default();
        s.put("x", 1.5).put("last_result", "error: two\nlines");
        write_atomic(&f, s.text()).unwrap();
        write_atomic(&f, s.text()).unwrap();
        assert_eq!(std::fs::read_to_string(&f).unwrap(), "x=1.5\nlast_result=error: two lines\n");
        assert!(!dir.join("status.txt.tmp").exists());
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A world whose ground appears after `loads_after` probes.
    struct Fake {
        probes: u32,
        loads_after: u32,
        ground: f32,
        terrain: Option<f32>,
        placed: Vec<([f32; 3], Option<f32>)>,
        asked: u32,
    }

    impl World for Fake {
        fn ground(&mut self, _x: f32, _y: f32, from_z: f32) -> Option<f32> {
            self.probes += 1;
            (self.probes > self.loads_after && from_z > self.ground).then_some(self.ground)
        }
        fn terrain(&mut self, _x: f32, _y: f32) -> Option<f32> {
            self.terrain
        }
        fn want_collision(&mut self, _at: [f32; 3]) {
            self.asked += 1;
        }
        fn place(&mut self, at: [f32; 3], heading: Option<f32>) -> Result<(), String> {
            self.placed.push((at, heading));
            Ok(())
        }
    }

    fn fake(loads_after: u32, terrain: Option<f32>) -> Fake {
        Fake { probes: 0, loads_after, ground: 4.0, terrain, placed: Vec::new(), asked: 0 }
    }

    #[test]
    fn a_near_teleport_ends_in_its_first_frame() {
        let mut w = fake(0, Some(3.5));
        let (_, s) = Warp::start(&mut w, 1.0, 2.0, Some(7.0), Some(0.5), 0);
        assert_eq!(s, WarpStep::Done { at: [1.0, 2.0, 7.0], frames: 1, ground: Some(4.0) });
        assert_eq!(w.placed, vec![([1.0, 2.0, 7.0], Some(0.5)), ([1.0, 2.0, 7.0], Some(0.5))]);
    }

    #[test]
    fn a_far_teleport_holds_then_finds_the_ground() {
        let mut w = fake(5, Some(3.5));
        let (mut warp, s) = Warp::start(&mut w, 1.0, 2.0, None, None, 0);
        assert_eq!(warp.hold_z, 13.5, "height map + 10");
        assert_eq!(s, WarpStep::Holding);
        let mut last = s;
        for f in 1..10 {
            last = warp.step(&mut w, f * 100);
            if last != WarpStep::Holding {
                break;
            }
        }
        assert_eq!(last, WarpStep::Done { at: [1.0, 2.0, 5.0], frames: 6, ground: Some(4.0) }, "ground + 1");
        assert!(w.placed[1..w.placed.len() - 1].iter().all(|p| p.0 == [1.0, 2.0, 13.5]), "held at the hold point");
        assert_eq!(w.asked, 6, "collision asked every frame");
    }

    #[test]
    fn a_teleport_without_ground_gives_up_in_time() {
        let mut w = fake(u32::MAX, Some(3.5));
        let (mut warp, _) = Warp::start(&mut w, 1.0, 2.0, None, None, 0);
        assert_eq!(warp.step(&mut w, WARP_WAIT_MS - 1), WarpStep::Holding);
        assert!(matches!(warp.step(&mut w, WARP_WAIT_MS), WarpStep::TimedOut { at: [1.0, 2.0, 4.5], .. }));
        let mut w = fake(u32::MAX, None);
        let (mut warp, _) = Warp::start(&mut w, 1.0, 2.0, None, None, 0);
        assert_eq!(warp.hold_z, NO_TERRAIN_HIGH);
        assert!(matches!(warp.step(&mut w, WARP_WAIT_MS), WarpStep::TimedOut { at: [1.0, 2.0, NO_TERRAIN_HIGH], .. }));
        let mut w = fake(u32::MAX, None);
        let (mut warp, _) = Warp::start(&mut w, 1.0, 2.0, Some(9.0), None, 0);
        assert!(matches!(warp.step(&mut w, WARP_WAIT_MS), WarpStep::TimedOut { at: [1.0, 2.0, 9.0], .. }));
    }
}
