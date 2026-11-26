//! `VICTORY_MOD=1`: the title side of the mod commands ([`crate::modcmd`]).
//! It runs inside the frame hook (the game timer update's detour,
//! [`crate::hooks`]), on the one thread that hook acts on, once a frame:
//! it reads the lines appended to `VICTORY_MOD_FILE` and the keys file's
//! mod lines, runs each command, steps the ones that take several frames
//! (the far teleport), and rewrites `VICTORY_MOD_STATUS` twice a second.
//!
//! Safety rules: every title function or data word is used only after the
//! bytes at its address in the mapped image are the ones recorded here for
//! this executable (1.0.2.1, PE stamp 0x6af783d4), checked once and kept;
//! a data word is reached only through an instruction whose rip-relative
//! target is checked ([`Pin`]); a command that would need an unchecked
//! address is refused, with why, in the status file and the log; nothing
//! but `status` runs while there is no player; a command runs once (held
//! states are re-applied about once a second, and dropped if that fails).
//!
//! Addresses, signatures and recipes: viper-community/viper e44573c6e
//! (`crates/viper-layout/src/game.rs`, `crates/viper-rt/src/engine.rs`,
//! `crates/viper/src/game.rs`; LGPL-2.1-or-later), every signature checked
//! again against our executable's bytes (all 245 of VIper's sites match).

use std::collections::{HashMap, VecDeque};
use std::path::PathBuf;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use crate::hooks::{heading_of, say, unpack_position};
use crate::modcmd::{self, Command, StatusText, Tail, Warp, WarpStep};

/// A title function or instruction: its RVA and the first bytes expected there.
#[derive(Clone, Copy, Debug)]
pub struct Site {
    pub name: &'static str,
    pub rva: u32,
    pub bytes: &'static [u8],
}

/// A data word reached through a rip-relative instruction of a site: the
/// 32-bit displacement at `site.rva + disp`, counted from `site.rva + end`
/// (the instruction's end), must land on `target`.
#[derive(Clone, Copy, Debug)]
pub struct Pin {
    pub site: Site,
    pub disp: u32,
    pub end: u32,
    pub target: u32,
}

/// The sites, from VIper's layout (names kept), checked in our executable.
#[allow(dead_code)]
pub mod sites {
    use super::{Pin, Site};

    /// `FindPlayerPed`: `mov rax, [rip+X]`, X the local `CPed*` word.
    pub const FIND_PLAYER_PED: Site = Site { name: "FIND_PLAYER_PED", rva: 0x04b0330, bytes: &[0x84, 0xc9, 0x75, 0x08, 0x48, 0x8b, 0x05, 0xfd, 0x6a, 0x6a, 0x0a, 0xc3] };
    pub const LOCAL_PLAYER_PED: Pin = Pin { site: FIND_PLAYER_PED, disp: 7, end: 11, target: 0x0ab5_6e38 };
    /// `CEntity::SetPosition(ent, const float4*, bool, bool, bool warp)` (the CPed's vtable slot 0x178).
    pub const ENTITY_SET_POSITION: Site = Site { name: "ENTITY_SET_POSITION", rva: 0x52ae150, bytes: &[0x41, 0x57, 0x41, 0x56, 0x41, 0x55, 0x41, 0x54, 0x56, 0x57, 0x55, 0x53, 0x48, 0x83, 0xec, 0x58, 0x40, 0x8a, 0xac, 0x24, 0xc0, 0x00, 0x00, 0x00] };
    /// `CVehicle::SetPosition(...)`: moves the vehicle with its occupants.
    pub const VEHICLE_SET_POSITION: Site = Site { name: "VEHICLE_SET_POSITION", rva: 0x1a57430, bytes: &[0x56, 0x57, 0x53, 0x48, 0x81, 0xec, 0x90, 0x00, 0x00, 0x00, 0x8a, 0x9c, 0x24, 0xd0, 0x00, 0x00, 0x00, 0x44, 0x88, 0x44, 0x24, 0x30, 0x44, 0x88] };
    /// The CPed's SetHeading (slot 0x180), `(ent, radians)` (read from our executable by patch2).
    pub const CPED_SET_HEADING: Site = Site { name: "CPED_SET_HEADING", rva: 0x1a3a160, bytes: &[0x56, 0x57, 0x53, 0x48, 0x81, 0xec, 0x50, 0x01, 0x00, 0x00, 0xc5, 0x78, 0x29, 0xbc, 0x24, 0x40, 0x01, 0x00, 0x00] };
    /// `CEntity::SetHeading`.
    pub const ENTITY_SET_HEADING: Site = Site { name: "ENTITY_SET_HEADING", rva: 0x1a3a670, bytes: &[0x56, 0x48, 0x83, 0xec, 0x70, 0xc5, 0x78, 0x29, 0x54, 0x24, 0x60, 0xc5, 0x78, 0x29, 0x4c, 0x24, 0x50, 0xc5, 0x78, 0x29, 0x44, 0x24, 0x40, 0xc5] };
    /// `CVehicle::SetHeading(veh, radians)`: an upright matrix around its position.
    pub const CVEHICLE_SET_HEADING: Site = Site { name: "CVEHICLE_SET_HEADING", rva: 0x5cae750, bytes: &[0x56, 0x57, 0x48, 0x83, 0xec, 0x28, 0x48, 0x89, 0xce, 0xe8, 0x12, 0xbf, 0xd8, 0xfb, 0x8b, 0xbe, 0x9c, 0x00, 0x00, 0x00, 0x89, 0xf9, 0xe8, 0xa5] };
    /// `void RequestCollisionAt(const float4*)`: a one-shot collision request.
    pub const COLLISION_REQUEST_AT: Site = Site { name: "COLLISION_REQUEST_AT", rva: 0x446bd00, bytes: &[0x48, 0x89, 0xca, 0x48, 0x8d, 0x0d, 0xe6, 0xbb, 0x98, 0x05, 0xe9, 0xd1, 0xfc, 0x7a, 0xfc, 0xcc] };
    /// `bool GetGroundZAndNormal(const float4*, float* z, float4* normal)` against loaded collision.
    pub const GROUND_Z: Site = Site { name: "GROUND_Z", rva: 0x19afe80, bytes: &[0x56, 0x57, 0x48, 0x83, 0xec, 0x38, 0xc5, 0xf8, 0x28, 0x01, 0xc4, 0xe2, 0x79, 0x18, 0x0d, 0xb9, 0x70, 0x61, 0x04, 0xc4, 0xe2, 0x79, 0x18, 0x15] };
    /// `float HeightMap_GetHeight(float x, float y)` (0 until the map is up).
    pub const HEIGHTMAP_HEIGHT: Site = Site { name: "HEIGHTMAP_HEIGHT", rva: 0x4e78810, bytes: &[0x56, 0x57, 0x53, 0x48, 0x81, 0xec, 0x80, 0x00, 0x00, 0x00, 0xc5, 0x78, 0x29, 0x54, 0x24, 0x70, 0xc5, 0x78, 0x29, 0x4c, 0x24, 0x60, 0xc5, 0x78] };
    /// `void Entity_GetHealthAndMax(CEntity*, float*, float*)`.
    pub const ENTITY_HEALTH_AND_MAX: Site = Site { name: "ENTITY_HEALTH_AND_MAX", rva: 0x02ba4e0, bytes: &[0x56, 0x57, 0x53, 0x48, 0x83, 0xec, 0x20, 0x0f, 0xb6, 0x41, 0x30, 0x4c, 0x89, 0xc6, 0x48, 0x89, 0xd7, 0x48, 0x89, 0xcb, 0x83, 0xf8, 0x02, 0x74] };
    /// `PedComponent51* (u32 entityId)`: +0x1080 the ped's in-vehicle info.
    pub const PED_COMPONENT_51: Site = Site { name: "PED_COMPONENT_51", rva: 0x4da6540, bytes: &[0x89, 0xc8, 0x25, 0x00, 0x00, 0x3e, 0x00, 0x3d, 0x00, 0x00, 0x06, 0x00, 0x75, 0x3a] };
    /// `CVehicle* VehicleFromEntityId(u32)` (generation checked).
    pub const VEHICLE_FROM_ID: Site = Site { name: "VEHICLE_FROM_ID", rva: 0x0bc0170, bytes: &[0x81, 0xf9, 0xff, 0xff, 0x01, 0x00, 0x75, 0x03, 0x31, 0xc0, 0xc3, 0x48, 0x8b, 0x15, 0x46, 0xe7, 0xff, 0x09] };
    /// `CWantedComp* (u32 entityId)`; only players have one.
    pub const WANTED_COMPONENT: Site = Site { name: "WANTED_COMPONENT", rva: 0x0d01e10, bytes: &[0x89, 0xc8, 0x25, 0x00, 0x00, 0x3e, 0x00, 0x3d, 0x00, 0x00, 0x06, 0x00, 0x75, 0x23, 0x81, 0xe1, 0xff, 0xff, 0x01, 0x00, 0x0f, 0xaf, 0x0d, 0xa1] };
    /// `Wanted_RecomputeLevel`: pins the level's offset (+0x5f2c) in the wanted component.
    pub const WANTED_LEVEL_READ: Site = Site { name: "WANTED_LEVEL_READ", rva: 0x09b1440, bytes: &[0x56, 0x57, 0x53, 0x48, 0x83, 0xec, 0x20, 0x48, 0x89, 0xce, 0x8b, 0x89, 0x28, 0x5f, 0x00, 0x00, 0x89, 0xd7, 0x8b, 0x9e, 0x2c, 0x5f, 0x00, 0x00] };
    /// `Waypoint_GetMgr`: `mov rax, [rip+X]; ret`.
    pub const WAYPOINT_MGR_READ: Site = Site { name: "WAYPOINT_MGR_READ", rva: 0x0d8a760, bytes: &[0x48, 0x8b, 0x05, 0x01, 0x1b, 0x77, 0x06, 0xc3] };
    pub const WAYPOINT_MGR: Pin = Pin { site: WAYPOINT_MGR_READ, disp: 3, end: 7, target: 0x074f_c268 };
    /// `void Clock_SetTime(i32 h, i32 m, i32 s)`; `vmovups ymm0, [rip+X]` X the clock struct.
    pub const CLOCK_SET_TIME: Site = Site { name: "CLOCK_SET_TIME", rva: 0x1ab9b20, bytes: &[0x56, 0x57, 0x53, 0x48, 0x83, 0xec, 0x50, 0xc5, 0xfc, 0x10, 0x05, 0x91, 0x32, 0x0a, 0x09, 0x45, 0x89, 0xc1, 0x41, 0x89, 0xd0, 0x89, 0xca, 0x48] };
    pub const GAME_CLOCK: Pin = Pin { site: CLOCK_SET_TIME, disp: 11, end: 15, target: 0x0ab5_cdc0 };
    /// `bool Clock_IsPaused()`: `mov al, [rip+X]`, X the pause-by-script byte.
    pub const CLOCK_IS_PAUSED: Site = Site { name: "CLOCK_IS_PAUSED", rva: 0x3013890, bytes: &[0x8a, 0x05, 0x9b, 0xc7, 0x2c, 0x06, 0x0a, 0x05, 0xc4, 0x0c, 0xff, 0x04, 0x24, 0x01, 0xc3] };
    pub const CLOCK_PAUSED_BY_SCRIPT: Pin = Pin { site: CLOCK_IS_PAUSED, disp: 2, end: 6, target: 0x092e_0031 };
    /// `i32 Weather_GetTypeIndexFromHash(u32)` (-1 none).
    pub const WEATHER_INDEX: Site = Site { name: "WEATHER_INDEX", rva: 0x1cd5210, bytes: &[0x8b, 0x05, 0xbe, 0x0c, 0xe7, 0x08, 0x48, 0x85, 0xc0, 0x74, 0x1d, 0x48, 0x8b, 0x15, 0xb6, 0x0c, 0xe7, 0x08, 0x45, 0x31, 0xc0, 0x42, 0x39, 0x0c] };
    /// In it: `mov edx, [rip+count] ... mov r9, [rip+types]`.
    pub const WEATHER_TYPES_READ: Site = Site { name: "WEATHER_TYPES_READ", rva: 0x1cd5238, bytes: &[0x8b, 0x15, 0x42, 0x0c, 0xe7, 0x08, 0xb8, 0xff, 0xff, 0xff, 0xff, 0x48, 0x85, 0xd2, 0x74, 0x1b, 0x4c, 0x8b, 0x0d, 0x69, 0x0c, 0xe7, 0x08] };
    pub const WEATHER_COUNT: Pin = Pin { site: WEATHER_TYPES_READ, disp: 2, end: 6, target: 0x0ab4_5e80 };
    pub const WEATHER_TYPES: Pin = Pin { site: WEATHER_TYPES_READ, disp: 19, end: 23, target: 0x0ab4_5eb8 };
    /// `void Weather_ResetForcedTransition(u8)`.
    pub const WEATHER_RESET_FORCED: Site = Site { name: "WEATHER_RESET_FORCED", rva: 0x2370e20, bytes: &[0x31, 0xc0, 0x89, 0x05, 0x98, 0x53, 0x7d, 0x08, 0xc6, 0x05, 0x97, 0x53, 0x7d, 0x08, 0x00, 0x84, 0xc9, 0x75, 0x06, 0x8b, 0x05, 0x8f, 0x53, 0x7d] };
    /// `void Weather_SetNowPersist(u32 index, u8 flags)`.
    pub const WEATHER_SET_NOW_PERSIST: Site = Site { name: "WEATHER_SET_NOW_PERSIST", rva: 0x2371e60, bytes: &[0x56, 0x57, 0x48, 0x83, 0xec, 0x28, 0x39, 0x0d, 0x14, 0x40, 0x7d, 0x08, 0x76, 0x4f, 0x89, 0xd6, 0x89, 0xcf, 0xe8, 0xc9, 0xea, 0xff, 0xff, 0x89] };
    /// `void Weather_ClearPersist()`: `mov dword [rip+X], -1`, X the persisting index.
    pub const WEATHER_CLEAR_PERSIST: Site = Site { name: "WEATHER_CLEAR_PERSIST", rva: 0x2370e70, bytes: &[0x48, 0x83, 0xec, 0x28, 0xc5, 0xf0, 0x57, 0xc9, 0xb9, 0xff, 0xff, 0xff, 0xff, 0x41, 0xb0, 0x01, 0xc7, 0x05, 0x42, 0x53, 0x7d, 0x08, 0xff, 0xff] };
    pub const WEATHER_PERSIST: Pin = Pin { site: WEATHER_CLEAR_PERSIST, disp: 18, end: 26, target: 0x0ab4_61cc };
    /// `u8 GetPedDeathState(u32 entityId)`: [`death_state`] values.
    pub const PED_DEATH_STATE: Site = Site { name: "PED_DEATH_STATE", rva: 0x4e23950, bytes: &[0x81, 0xe1, 0xff, 0xff, 0x01, 0x00, 0x81, 0xf9, 0xff, 0xff, 0x01, 0x00, 0x74, 0x0c, 0x48, 0x8b, 0x05, 0x73, 0xf8, 0xe0, 0x02, 0x0f, 0xb6, 0x04] };
    /// `void PedDowned_RestoreHealth(CPed*)`: the first half of getting a downed ped up (health to the revive value, downed bits cleared). Reads the health component unchecked: only on a 
    pub const PED_REVIVE_DOWNED: Site = Site { name: "PED_REVIVE_DOWNED", rva: 0x2b82940, bytes: &[0x56, 0x57, 0x53, 0x48, 0x83, 0xec, 0x20, 0x8b, 0x99, 0x9c, 0x00, 0x00, 0x00, 0x48, 0x89, 0xcf, 0x89, 0xd9, 0xe8, 0xe9, 0x6c, 0x4c, 0x02, 0x48] };
    /// `void Ped_SetDowned(CPed*, u32 downed)`; with health above 0, 0 makes the ped alive again.
    pub const PED_SET_DOWNED: Site = Site { name: "PED_SET_DOWNED", rva: 0x1fc7960, bytes: &[0x41, 0x56, 0x56, 0x57, 0x53, 0x48, 0x83, 0xec, 0x48, 0x8b, 0x99, 0x9c, 0x00, 0x00, 0x00, 0x48, 0x89, 0xce, 0x89, 0xd7, 0x89, 0xd9, 0xe8, 0xc5] };
    /// `float Ped_GetMaxHealth(CPed*)`.
    pub const PED_GET_MAX_HEALTH: Site = Site { name: "PED_GET_MAX_HEALTH", rva: 0x088b860, bytes: &[0x56, 0x48, 0x83, 0xec, 0x20, 0x48, 0x89, 0xce, 0x8b, 0x89, 0x9c, 0x00, 0x00, 0x00, 0xe8, 0x0d, 0xc3, 0x52, 0x04, 0x48, 0x85, 0xc0, 0x74, 0x0e] };
    /// `void Ped_SetHealth(CPed*, float health (xmm1), u8)`; creates the health component if missing.
    pub const PED_SET_HEALTH: Site = Site { name: "PED_SET_HEALTH", rva: 0x191d9b0, bytes: &[0x56, 0x53, 0x48, 0x83, 0xec, 0x38, 0xc5, 0xf8, 0x29, 0x74, 0x24, 0x20, 0x48, 0x89, 0xce, 0x8b, 0x89, 0x9c, 0x00, 0x00, 0x00, 0x44, 0x89, 0xc3] };
    /// `PedComponent32* (u32 entityId)`: +[`damage::MULTIPLIERS`] points at the ped's damage multiplier block (null until one is set).
    pub const PED_COMPONENT_32: Site = Site { name: "PED_COMPONENT_32", rva: 0x4eda820, bytes: &[0x89, 0xc8, 0x25, 0x00, 0x00, 0x3e, 0x00, 0x3d, 0x00, 0x00, 0x06, 0x00, 0x75, 0x3a, 0x81, 0xe1, 0xff, 0xff, 0x01, 0x00, 0x0f, 0xaf, 0x0d, 0x91] };
    /// `void Ped_SetDamageMultiplier(CPed*, int type, float mult)` (`mult` in xmm2): what the game's damage-proofs script command runs. Creates the multiplier block (all 1.0) if missing.
    pub const PED_SET_DAMAGE_MULTIPLIER: Site = Site { name: "PED_SET_DAMAGE_MULTIPLIER", rva: 0x4be3640, bytes: &[0x56, 0x48, 0x83, 0xec, 0x30, 0xc5, 0xf8, 0x29, 0x74, 0x24, 0x20, 0xc5, 0xf8, 0x28, 0xf2, 0x89, 0xd6, 0xe8, 0x5a, 0x9f, 0xff, 0xff, 0x48, 0x85] };
    /// `void Script_ClearPlayerWantedLevel(CPed*)`: what the clear-wanted script command runs (clears the level and the incidents, refreshes the HUD for the local player).
    pub const WANTED_CLEAR_FOR_PLAYER: Site = Site { name: "WANTED_CLEAR_FOR_PLAYER", rva: 0x3eb8530, bytes: &[0x56, 0x57, 0x48, 0x83, 0xec, 0x28, 0x48, 0x85, 0xc9, 0x74, 0x42, 0x8b, 0xb9, 0x9c, 0x00, 0x00, 0x00, 0x48, 0x89, 0xce, 0x89, 0xf9, 0xe8, 0x95] };
    /// `void Wanted_SetLevelForPed(CPed*, int level, bool, bool, bool)`, called as `(ped, n, 1, 1, 1)` like the set-wanted script command. Takes effect in the same call.
    pub const WANTED_SET_LEVEL: Site = Site { name: "WANTED_SET_LEVEL", rva: 0x2367480, bytes: &[0x41, 0x57, 0x41, 0x56, 0x41, 0x55, 0x41, 0x54, 0x56, 0x57, 0x55, 0x53, 0x48, 0x81, 0xec, 0xd8, 0x01, 0x00, 0x00, 0x48, 0x89, 0xce, 0x85, 0xd2] };
    /// `void Wanted_SetMaxLevel(int max)`: writes [`WANTED_MAX_LEVEL_GLOBAL`] (its `mov [rip+X], ecx` is in these bytes) and the matching point cap. 0 = never wanted.
    pub const WANTED_SET_MAX_LEVEL: Site = Site { name: "WANTED_SET_MAX_LEVEL", rva: 0x582f080, bytes: &[0x56, 0x57, 0x48, 0x83, 0xec, 0x28, 0x89, 0x0d, 0x40, 0x16, 0x7f, 0x04, 0x83, 0xf9, 0x06, 0x7c, 0x13, 0x8b, 0x0d, 0x25, 0x5c, 0x61, 0x04, 0x89] };
    /// `Wanted_IgnoresCrimes`: pins [`WANTED_FLAGS`] bit [`WANTED_IGNORE_CRIMES`] (`mov eax, [rcx+0x14]; and eax, 0x20`).
    pub const WANTED_IGNORE_READ: Site = Site { name: "WANTED_IGNORE_READ", rva: 0x00ea540, bytes: &[0x85, 0xd2, 0x75, 0x19, 0x8b, 0x41, 0x14, 0x83, 0xe0, 0x20, 0x0b, 0x81, 0x38, 0x5f, 0x00, 0x00, 0x75, 0x0f, 0x83, 0xb9, 0x40, 0x5f, 0x00, 0x00] };
    pub const WANTED_MAX_LEVEL_GLOBAL: Pin = Pin { site: WANTED_SET_MAX_LEVEL, disp: 8, end: 12, target: 0x0a02_06cc };
}

/// The CPed vtable and the two slots called through an entity's vtable.
const CPED_VTABLE: u32 = 0x62f_7170;
const VT_SET_POSITION: usize = 0x178;
const VT_SET_HEADING: usize = 0x180;
/// Entity fields (VIper `entity`): packed position, forward row, ECS id.
const POSITION: usize = 0x70;
const FORWARD: usize = 0x50;
const ENTITY_ID: usize = 0x9c;
/// In-vehicle info (VIper PED_VEHICLE_INFO, VEHICLE_INFO_FLAGS / _IN_VEHICLE / _VEHICLE_ID).
const PED_VEHICLE_INFO: usize = 0x1080;
const VEHICLE_INFO_FLAGS: usize = 180;
const VEHICLE_INFO_IN_VEHICLE: u8 = 2;
const VEHICLE_INFO_VEHICLE_ID: usize = 248;
/// The wanted component's current level (VIper WANTED_CURRENT_LEVEL).
const WANTED_CURRENT_LEVEL: usize = 0x5f2c;
/// The ped's damage multiplier block in component 32: one f32 per type at
/// FIRST + STRIDE * type (VIper `damage`).
const DAMAGE_MULTIPLIERS: usize = 0x110;
const DAMAGE_FIRST: usize = 0xe4;
const DAMAGE_STRIDE: usize = 12;
const DAMAGE_TYPES: usize = 19;
/// Death states (VIper `death_state`).
const DOWNED: u8 = 4;
const DYING: u8 = 5;
/// The wanted component's flags and its "crimes ignored" bit (VIper WANTED_FLAGS, WANTED_IGNORE_CRIMES).
const WANTED_FLAGS: usize = 0x14;
const WANTED_IGNORE_CRIMES: u32 = 0x20;
/// Clock fields, weather type stride (VIper CLOCK_HOUR / _MINUTE, WEATHER_TYPE_SIZE).
const CLOCK_HOUR: usize = 0x10;
const CLOCK_MINUTE: usize = 0x14;
const WEATHER_TYPE_SIZE: usize = 56;
/// Waypoint slot 0 (VIper `waypoint`).
const WAYPOINT_BLIP: usize = 0x08;
const WAYPOINT_SET: usize = 0x2e;
const WAYPOINT_X: usize = 0x14;
const WAYPOINT_Y: usize = 0x18;

/// How often the status file is rewritten.
const STATUS_EVERY: Duration = Duration::from_millis(500);
/// How often held states (frozen clock, held weather) are re-applied.
const KEEP_EVERY: Duration = Duration::from_secs(1);
/// Most commands run in one frame.
const MOST_A_FRAME: usize = 8;
/// Most commands waiting.
const MOST_QUEUED: usize = 64;

static ON: AtomicBool = AtomicBool::new(false);
/// The keys file's mod lines (the front end's adapter of victory_hle's queue).
static LINE_SOURCE: AtomicUsize = AtomicUsize::new(0);

/// Whether `VICTORY_MOD=1` was taken ([`configure`]).
pub fn enabled() -> bool {
    ON.load(Ordering::Relaxed)
}

/// Where the frame hook takes the keys file's mod lines from.
pub fn set_line_source(take: fn() -> Option<String>) {
    LINE_SOURCE.store(take as usize, Ordering::Relaxed);
}

#[repr(C, align(16))]
struct Float4([f32; 4]);

/// The mod's state, on the frame hook's thread.
struct State {
    tail: Option<Tail>,
    status: Option<PathBuf>,
    status_at: Option<Instant>,
    status_said: bool,
    queue: VecDeque<(&'static str, String)>,
    /// Per site RVA: whether its bytes (and pin) were the expected ones.
    checked: HashMap<u32, bool>,
    warp: Option<WarpJob>,
    last: Option<Last>,
    ok: u64,
    refused: u64,
    started: Instant,
    /// Held states, re-applied about once a second ([`keep`]).
    held: Held,
    kept_at: Option<Instant>,
}

#[derive(Default)]
struct Held {
    /// `time freeze on`.
    freeze: bool,
    /// `weather <name>`: its hash, the game's index for it, its name.
    weather: Option<(u32, u32, String)>,
    /// `invincible on`: the ped it was applied to and its multipliers before.
    invincible: Option<(usize, [f32; DAMAGE_TYPES])>,
    /// `wanted never on`.
    never_wanted: bool,
}

struct WarpJob {
    warp: Warp,
    ped: usize,
    /// The vehicle moved with the player, or 0 on foot.
    vehicle: usize,
    line: String,
}

struct Last {
    source: &'static str,
    line: String,
    result: String,
    frame: u64,
}

static STATE: Mutex<Option<State>> = Mutex::new(None);

/// Reads the switches: `VICTORY_MOD=1` turns it on, `VICTORY_MOD_FILE` and
/// `VICTORY_MOD_STATUS` name its files. Returns whether it is on.
pub fn configure(var: &dyn Fn(&str) -> Option<String>) -> bool {
    let on = var("VICTORY_MOD").is_some_and(|v| v.trim() == "1");
    ON.store(on, Ordering::Relaxed);
    if !on {
        return false;
    }
    let path = |n: &str| var(n).map(|v| v.trim().to_owned()).filter(|v| !v.is_empty()).map(PathBuf::from);
    let (file, status) = (path("VICTORY_MOD_FILE"), path("VICTORY_MOD_STATUS"));
    println!(
        "title mod      VICTORY_MOD=1: commands from {} and the keys file, status to {}",
        file.as_ref().map_or("(no VICTORY_MOD_FILE)".into(), |p| p.display().to_string()),
        status.as_ref().map_or("(no VICTORY_MOD_STATUS)".into(), |p| p.display().to_string())
    );
    *STATE.lock().unwrap_or_else(std::sync::PoisonError::into_inner) = Some(State {
        tail: file.map(Tail::new),
        status,
        status_at: None,
        status_said: false,
        queue: VecDeque::new(),
        checked: HashMap::new(),
        warp: None,
        last: None,
        ok: 0,
        refused: 0,
        started: Instant::now(),
        held: Held::default(),
        kept_at: None,
    });
    true
}

/// Queues a command line for the next frame (the hook's own adapter of the
/// keys file's teleports).
pub fn push(source: &'static str, line: String) {
    if let Some(st) = STATE.lock().unwrap_or_else(std::sync::PoisonError::into_inner).as_mut()
        && st.queue.len() < MOST_QUEUED
    {
        st.queue.push_back((source, line));
    }
}

/// Whether `site`'s bytes in the mapped image are the expected ones.
///
/// # Safety
/// `base` is the mapped title; `site` lies inside its code.
unsafe fn bytes_match(base: usize, site: &Site) -> bool {
    // SAFETY: the title's code, mapped readable for its whole life.
    let have = unsafe { std::slice::from_raw_parts((base + site.rva as usize) as *const u8, site.bytes.len()) };
    have == site.bytes
}

impl State {
    /// Checks the sites (once each, kept), or says which one is not the expected code.
    fn need(&mut self, base: usize, sites: &[Site]) -> Result<(), String> {
        for s in sites {
            // SAFETY: the mapped title; each site lies in its code.
            let ok = *self.checked.entry(s.rva).or_insert_with(|| unsafe { bytes_match(base, s) });
            if !ok {
                return Err(format!("{} (+{:#x}) is not the code recorded for this executable: refused", s.name, s.rva));
            }
        }
        Ok(())
    }

    /// The address of a pinned data word, after checking its site and its rip-relative target.
    fn pin(&mut self, base: usize, p: Pin) -> Result<usize, String> {
        self.need(base, &[p.site])?;
        let key = p.site.rva.wrapping_add(p.disp) | 0x8000_0000;
        let ok = *self.checked.entry(key).or_insert_with(|| {
            // SAFETY: inside the checked site's instruction (its bytes matched).
            let d = unsafe { ((base + (p.site.rva + p.disp) as usize) as *const i32).read_unaligned() };
            (i64::from(p.site.rva + p.end) + i64::from(d)) == i64::from(p.target)
        });
        if !ok {
            return Err(format!("the data word {:#x} read by {} is not where recorded: refused", p.target, p.site.name));
        }
        Ok(base + p.target as usize)
    }
}

// ---- reading and moving entities (the title's own functions) ----------------------

/// # Safety
/// `ent` a live entity, on the thread that updates it.
unsafe fn entity_place(ent: usize) -> ([f32; 3], f32) {
    // SAFETY: the entity's packed position and forward row (VIper's layout).
    unsafe {
        let packed = ((ent + POSITION) as *const [u8; 16]).read_unaligned();
        let fx = ((ent + FORWARD) as *const f32).read_unaligned();
        let fy = ((ent + FORWARD + 4) as *const f32).read_unaligned();
        (unpack_position(&packed), heading_of(fx, fy))
    }
}

/// # Safety
/// `ent` a live entity.
unsafe fn entity_id(ent: usize) -> u32 {
    // SAFETY: a u32 field of the live entity.
    unsafe { ((ent + ENTITY_ID) as *const u32).read_unaligned() }
}

/// The vehicle the player is in (its in-vehicle info and the generation-checked
/// lookup), or 0. Needs PED_COMPONENT_51 and VEHICLE_FROM_ID checked.
///
/// # Safety
/// `ped` the live player, on its thread.
unsafe fn ped_vehicle(base: usize, ped: usize) -> usize {
    // SAFETY: the two title functions (checked by the caller) with an entity id; plain reads of what they return.
    unsafe {
        let comp: extern "C" fn(u32) -> usize = std::mem::transmute(base + sites::PED_COMPONENT_51.rva as usize);
        let veh_of: extern "C" fn(u32) -> usize = std::mem::transmute(base + sites::VEHICLE_FROM_ID.rva as usize);
        let c = comp(entity_id(ped));
        if c == 0 {
            return 0;
        }
        let info = ((c + PED_VEHICLE_INFO) as *const usize).read_unaligned();
        if info == 0 || ((info + VEHICLE_INFO_FLAGS) as *const u8).read() & VEHICLE_INFO_IN_VEHICLE == 0 {
            return 0;
        }
        veh_of(((info + VEHICLE_INFO_VEHICLE_ID) as *const u32).read_unaligned())
    }
}

/// Moves an entity through its own SetPosition with warp (only the two
/// known ones), then turns it through its own SetHeading (only the three
/// known ones).
///
/// # Safety
/// `ent` a live ped or vehicle, on its thread; the five sites checked.
unsafe fn move_entity(base: usize, ent: usize, at: [f32; 3], heading: Option<f32>) -> Result<(), String> {
    // SAFETY: the entity's vtable pointer and two of its slots (read only).
    let (set, turn) = unsafe {
        let vt = (ent as *const usize).read();
        (((vt + VT_SET_POSITION) as *const usize).read(), ((vt + VT_SET_HEADING) as *const usize).read())
    };
    let known_set = [sites::ENTITY_SET_POSITION.rva, sites::VEHICLE_SET_POSITION.rva];
    if !known_set.iter().any(|&r| set == base + r as usize) {
        return Err(format!("its SetPosition slot is {:#x}, not a known one", set.wrapping_sub(base)));
    }
    let v = Float4([at[0], at[1], at[2], 0.0]);
    // SAFETY: a checked SetPosition with VIper's arguments (ent, &float4 16-aligned, 1, 1, warp 1).
    unsafe {
        let f: extern "C" fn(usize, *const Float4, u8, u8, u8) = std::mem::transmute(set);
        f(ent, &v, 1, 1, 1);
    }
    if let Some(h) = heading {
        let known_turn = [sites::CPED_SET_HEADING.rva, sites::ENTITY_SET_HEADING.rva, sites::CVEHICLE_SET_HEADING.rva];
        if !known_turn.iter().any(|&r| turn == base + r as usize) {
            return Err("moved, but its SetHeading slot is not a known one: not turned".into());
        }
        // SAFETY: a checked SetHeading: (ent, radians in xmm1).
        unsafe {
            let g: extern "C" fn(usize, f32) = std::mem::transmute(turn);
            g(ent, h);
        }
    }
    Ok(())
}

const MOVE_SITES: [Site; 7] = [
    sites::ENTITY_SET_POSITION,
    sites::VEHICLE_SET_POSITION,
    sites::CPED_SET_HEADING,
    sites::ENTITY_SET_HEADING,
    sites::CVEHICLE_SET_HEADING,
    sites::PED_COMPONENT_51,
    sites::VEHICLE_FROM_ID,
];
const GROUND_SITES: [Site; 3] = [sites::COLLISION_REQUEST_AT, sites::GROUND_Z, sites::HEIGHTMAP_HEIGHT];

/// The far teleport's [`modcmd::World`] over the title.
struct TitleWorld {
    base: usize,
    ent: usize,
}

impl modcmd::World for TitleWorld {
    fn ground(&mut self, x: f32, y: f32, from_z: f32) -> Option<f32> {
        let pos = Float4([x, y, from_z, 0.0]);
        let mut z = 0.0f32;
        let mut normal = Float4([0.0; 4]);
        // SAFETY: GetGroundZAndNormal (checked before a warp starts), 16-aligned in and out.
        let found = unsafe {
            let f: extern "C" fn(*const Float4, *mut f32, *mut Float4) -> bool = std::mem::transmute(self.base + sites::GROUND_Z.rva as usize);
            f(&pos, &mut z, &mut normal)
        };
        (found && z.is_finite()).then_some(z)
    }

    fn terrain(&mut self, x: f32, y: f32) -> Option<f32> {
        // SAFETY: HeightMap_GetHeight (checked), two floats in xmm0 and xmm1.
        let h = unsafe {
            let f: extern "C" fn(f32, f32) -> f32 = std::mem::transmute(self.base + sites::HEIGHTMAP_HEIGHT.rva as usize);
            f(x, y)
        };
        (h != 0.0 && h.is_finite()).then_some(h)
    }

    fn want_collision(&mut self, at: [f32; 3]) {
        let v = Float4([at[0], at[1], at[2], 0.0]);
        // SAFETY: RequestCollisionAt (checked), a 16-aligned float4.
        unsafe {
            let f: extern "C" fn(*const Float4) = std::mem::transmute(self.base + sites::COLLISION_REQUEST_AT.rva as usize);
            f(&v);
        }
    }

    fn place(&mut self, at: [f32; 3], heading: Option<f32>) -> Result<(), String> {
        // SAFETY: the live player or its vehicle, on the frame hook's thread; the move sites checked.
        unsafe { move_entity(self.base, self.ent, at, heading) }
    }
}

// ---- the frame's work ---------------------------------------------------------------

fn now_ms(st: &State) -> u64 {
    st.started.elapsed().as_millis() as u64
}

/// Once a frame, on the frame hook's thread, with the player's `CPed*` (0
/// when none) and the frame counter of the log lines.
pub fn frame(base: usize, ped: usize, frame_no: u64) {
    if !enabled() {
        return;
    }
    let Ok(mut guard) = STATE.try_lock() else { return };
    let Some(st) = guard.as_mut() else { return };
    if let Some(t) = st.tail.as_mut() {
        for line in t.poll() {
            if st.queue.len() < MOST_QUEUED {
                st.queue.push_back(("modfile", line));
            } else {
                say(true, &format!("VICTORY_MOD: more than {MOST_QUEUED} commands waiting; dropped {line:?}"));
            }
        }
    }
    let source = LINE_SOURCE.load(Ordering::Relaxed);
    if source != 0 {
        // SAFETY: set by `set_line_source` from a function of this type.
        let take: fn() -> Option<String> = unsafe { std::mem::transmute(source) };
        while let Some(line) = take() {
            st.queue.push_back(("keys", line));
        }
    }
    step_warp(st, base, ped, frame_no);
    if ped != 0 && st.kept_at.is_none_or(|t| t.elapsed() >= KEEP_EVERY) {
        st.kept_at = Some(Instant::now());
        keep(st, base, ped, frame_no);
    }
    let mut ran = 0;
    while ran < MOST_A_FRAME {
        let Some((src, line)) = st.queue.pop_front() else { break };
        ran += 1;
        let result = match modcmd::parse(&line) {
            Ok(None) => continue,
            Err(e) => Err(e),
            Ok(Some(cmd)) => run(st, base, ped, frame_no, &line, cmd),
        };
        record(st, src, &line, frame_no, result);
    }
    let due = st.status_at.is_none_or(|t| t.elapsed() >= STATUS_EVERY);
    if due {
        st.status_at = Some(Instant::now());
        write_status(st, base, ped, frame_no);
    }
}

fn record(st: &mut State, source: &'static str, line: &str, frame_no: u64, result: Result<String, String>) {
    let text = match &result {
        Ok(s) => {
            st.ok += 1;
            format!("ok: {s}")
        }
        Err(e) => {
            st.refused += 1;
            format!("refused: {e}")
        }
    };
    say(result.is_err(), &format!("VICTORY_MOD: {source} {line:?} (frame {frame_no}): {text}"));
    st.last = Some(Last { source, line: line.to_owned(), result: text, frame: frame_no });
}

/// Runs one command; `Ok` says what was done.
fn run(st: &mut State, base: usize, ped: usize, frame_no: u64, line: &str, cmd: Command) -> Result<String, String> {
    if cmd == Command::Status {
        st.status_at = None;
        return Ok("the status file is written this frame".into());
    }
    if ped == 0 {
        return Err("no player yet (loading, a menu or a cutscene without one)".into());
    }
    match cmd {
        Command::Teleport { x, y, z, heading } => start_warp(st, base, ped, frame_no, line, x, y, z, heading),
        Command::Relative { forward, up } => {
            // SAFETY: the live player, on its thread.
            let (here, h) = unsafe { entity_place(ped) };
            let t = modcmd::relative(here, h, forward, up);
            start_warp(st, base, ped, frame_no, line, t[0], t[1], Some(t[2]), None)
        }
        Command::Waypoint => {
            let m = st.pin(base, sites::WAYPOINT_MGR)?;
            // SAFETY: the waypoint manager word (pinned) and its slot 0's fields (VIper's layout).
            let wp = unsafe {
                let m = (m as *const usize).read_volatile();
                if m == 0 {
                    None
                } else {
                    let set = ((m + WAYPOINT_BLIP) as *const usize).read_unaligned() != 0 || ((m + WAYPOINT_SET) as *const u8).read() != 0;
                    let (x, y) = (((m + WAYPOINT_X) as *const f32).read_unaligned(), ((m + WAYPOINT_Y) as *const f32).read_unaligned());
                    (set && x.is_finite() && y.is_finite()).then_some((x, y))
                }
            };
            let (x, y) = wp.ok_or("no map waypoint is set")?;
            start_warp(st, base, ped, frame_no, line, x, y, None, None)
        }
        Command::Time { hour, minute } => {
            st.need(base, &[sites::CLOCK_SET_TIME])?;
            // SAFETY: Clock_SetTime (checked): (hour, minute, second), validated by the game.
            unsafe {
                let f: extern "C" fn(i32, i32, i32) = std::mem::transmute(base + sites::CLOCK_SET_TIME.rva as usize);
                f(hour as i32, minute as i32, 0);
            }
            let (hh, mm) = clock(st, base).ok_or("set, but the clock could not be read back")?;
            Ok(format!("the clock is {hh:02}:{mm:02}"))
        }
        Command::Freeze(on) => {
            let b = st.pin(base, sites::CLOCK_PAUSED_BY_SCRIPT)?;
            // SAFETY: the pause-by-script byte (pinned), a flag the game reads each frame.
            unsafe { (b as *mut u8).write_volatile(u8::from(on)) };
            st.held.freeze = on;
            Ok(if on { "the clock is stopped (kept so once a second)" } else { "the clock runs" }.into())
        }
        Command::Weather { hash, name } => {
            let idx = weather_index(st, base, hash)?.ok_or_else(|| format!("the game has no weather {name:?} ({hash:#010x})"))?;
            set_weather(st, base, idx)?;
            st.held.weather = Some((hash, idx, name.clone()));
            Ok(format!("the weather is {name} (type {idx}), held until weather cycle"))
        }
        Command::WeatherCycle => {
            st.need(base, &[sites::WEATHER_CLEAR_PERSIST])?;
            // SAFETY: Weather_ClearPersist (checked), no arguments.
            unsafe {
                let f: extern "C" fn() = std::mem::transmute(base + sites::WEATHER_CLEAR_PERSIST.rva as usize);
                f();
            }
            st.held.weather = None;
            Ok("the weather follows its own cycle again".into())
        }
        Command::Heal => {
            // SAFETY: the live player, on its thread (sites checked inside).
            unsafe { heal(st, base, ped) }
        }
        Command::Invincible(on) => {
            // SAFETY: the live player, on its thread (sites checked inside).
            let now = unsafe { damage_multipliers(st, base, ped) }?;
            if on {
                let saved = match (st.held.invincible, now) {
                    (Some((p, saved)), _) if p == ped => saved,
                    (_, Some(m)) if m != [0.0; DAMAGE_TYPES] => m,
                    _ => [1.0; DAMAGE_TYPES],
                };
                // SAFETY: as above.
                unsafe { set_damage_multipliers(st, base, ped, &[0.0; DAMAGE_TYPES]) }?;
                st.held.invincible = Some((ped, saved));
                Ok("the player takes no damage (all 19 damage multipliers 0, kept so once a second)".into())
            } else {
                let back = match st.held.invincible.take() {
                    Some((p, saved)) if p == ped => saved,
                    _ => [1.0; DAMAGE_TYPES],
                };
                // SAFETY: as above.
                unsafe { set_damage_multipliers(st, base, ped, &back) }?;
                Ok("the player takes damage again (its multipliers as before)".into())
            }
        }
        Command::Wanted(n) => {
            if n > 0 && st.held.never_wanted {
                return Err("never wanted is on (wanted never off first)".into());
            }
            st.need(base, &[sites::WANTED_SET_LEVEL, sites::WANTED_COMPONENT, sites::WANTED_LEVEL_READ])?;
            // SAFETY: Wanted_SetLevelForPed (checked) as the set-wanted script command calls it; then the level read back.
            let now = unsafe {
                let f: extern "C" fn(usize, i32, u8, u8, u8) = std::mem::transmute(base + sites::WANTED_SET_LEVEL.rva as usize);
                f(ped, n as i32, 1, 1, 1);
                wanted_level(base, ped)
            };
            match now {
                Some(l) if l == n as i32 => Ok(format!("wanted level {l}")),
                Some(l) => Ok(format!("asked for {n}; the game holds it at {l} (its highest allowed now may be lower)")),
                None => Err("the player has no wanted state".into()),
            }
        }
        Command::NeverWanted(on) => {
            // SAFETY: the live player, on its thread (sites checked inside).
            unsafe { set_never_wanted(st, base, ped, on) }?;
            st.held.never_wanted = on;
            Ok(if on { "never wanted: cleared, highest level 0, crimes ignored (kept so once a second)" } else { "wanted again possible: highest level 6, crimes count" }.into())
        }
        Command::Status => unreachable!(),
        _ => Err("not in this build yet".into()),
    }
}

/// The player's wanted level (None without a wanted component).
///
/// # Safety
/// `ped` the live player; WANTED_COMPONENT and WANTED_LEVEL_READ checked.
unsafe fn wanted_level(base: usize, ped: usize) -> Option<i32> {
    // SAFETY: GetWantedComponent (checked) with the player's id; the level field WANTED_LEVEL_READ pins.
    unsafe {
        let f: extern "C" fn(u32) -> usize = std::mem::transmute(base + sites::WANTED_COMPONENT.rva as usize);
        let w = f(entity_id(ped));
        (w != 0).then(|| ((w + WANTED_CURRENT_LEVEL) as *const i32).read_unaligned())
    }
}

/// Heals the player (getting it up first if downed), as VIper's `heal`.
///
/// # Safety
/// `ped` the live player, on its thread.
unsafe fn heal(st: &mut State, base: usize, ped: usize) -> Result<String, String> {
    st.need(
        base,
        &[sites::PED_DEATH_STATE, sites::PED_REVIVE_DOWNED, sites::PED_SET_DOWNED, sites::PED_GET_MAX_HEALTH, sites::PED_SET_HEALTH, sites::ENTITY_HEALTH_AND_MAX],
    )?;
    // SAFETY: the checked ped functions, with the player and VIper's arguments.
    unsafe {
        let state: extern "C" fn(u32) -> u8 = std::mem::transmute(base + sites::PED_DEATH_STATE.rva as usize);
        let s = state(entity_id(ped));
        if s >= DYING {
            return Err(format!("the player is dying or dead (state {s}): a heal does not bring it back"));
        }
        if s == DOWNED {
            let revive: extern "C" fn(usize) = std::mem::transmute(base + sites::PED_REVIVE_DOWNED.rva as usize);
            let downed: extern "C" fn(usize, u32) = std::mem::transmute(base + sites::PED_SET_DOWNED.rva as usize);
            revive(ped);
            downed(ped, 0);
        }
        let max: extern "C" fn(usize) -> f32 = std::mem::transmute(base + sites::PED_GET_MAX_HEALTH.rva as usize);
        let set: extern "C" fn(usize, f32, u8) = std::mem::transmute(base + sites::PED_SET_HEALTH.rva as usize);
        let m = max(ped);
        if !(m.is_finite() && m > 0.0) {
            return Err(format!("the player's max health reads {m}: not healed"));
        }
        set(ped, m, 1);
        let read: extern "C" fn(usize, *mut f32, *mut f32) = std::mem::transmute(base + sites::ENTITY_HEALTH_AND_MAX.rva as usize);
        let (mut h, mut mm) = (0.0f32, 0.0f32);
        read(ped, &mut h, &mut mm);
        Ok(format!("health {h:.0} of {mm:.0}{}", if s == DOWNED { " (got up from downed)" } else { "" }))
    }
}

/// The player's 19 damage multipliers (`None` before any was set: all 1).
///
/// # Safety
/// `ped` the live player, on its thread.
unsafe fn damage_multipliers(st: &mut State, base: usize, ped: usize) -> Result<Option<[f32; DAMAGE_TYPES]>, String> {
    st.need(base, &[sites::PED_COMPONENT_32, sites::PED_SET_DAMAGE_MULTIPLIER])?;
    // SAFETY: PedComponent32 (checked) with the player's id; its multiplier block (VIper's layout).
    unsafe {
        let comp: extern "C" fn(u32) -> usize = std::mem::transmute(base + sites::PED_COMPONENT_32.rva as usize);
        let c = comp(entity_id(ped));
        if c == 0 {
            return Ok(None);
        }
        let block = ((c + DAMAGE_MULTIPLIERS) as *const usize).read_unaligned();
        if block == 0 {
            return Ok(None);
        }
        let mut out = [1.0; DAMAGE_TYPES];
        for (t, v) in out.iter_mut().enumerate() {
            *v = ((block + DAMAGE_FIRST + DAMAGE_STRIDE * t) as *const f32).read_unaligned();
        }
        Ok(Some(out))
    }
}

/// Sets each damage multiplier through the game's own setter (what its damage-proofs command runs).
///
/// # Safety
/// `ped` the live player, on its thread.
unsafe fn set_damage_multipliers(st: &mut State, base: usize, ped: usize, values: &[f32; DAMAGE_TYPES]) -> Result<(), String> {
    st.need(base, &[sites::PED_SET_DAMAGE_MULTIPLIER])?;
    // SAFETY: Ped_SetDamageMultiplier (checked): (ped, type 0..18, f32 in xmm2).
    unsafe {
        let set: extern "C" fn(usize, i32, f32) = std::mem::transmute(base + sites::PED_SET_DAMAGE_MULTIPLIER.rva as usize);
        for (t, &v) in values.iter().enumerate() {
            set(ped, t as i32, v);
        }
    }
    Ok(())
}

/// Never wanted on (clear, highest level 0, crimes ignored) or off (6, crimes count), as VIper's `never_wanted`.
///
/// # Safety
/// `ped` the live player, on its thread.
unsafe fn set_never_wanted(st: &mut State, base: usize, ped: usize, on: bool) -> Result<(), String> {
    st.need(base, &[sites::WANTED_CLEAR_FOR_PLAYER, sites::WANTED_SET_MAX_LEVEL, sites::WANTED_COMPONENT, sites::WANTED_IGNORE_READ])?;
    // SAFETY: the checked wanted functions with the player; the flags word WANTED_IGNORE_READ pins.
    unsafe {
        let clear: extern "C" fn(usize) = std::mem::transmute(base + sites::WANTED_CLEAR_FOR_PLAYER.rva as usize);
        let max: extern "C" fn(i32) = std::mem::transmute(base + sites::WANTED_SET_MAX_LEVEL.rva as usize);
        let comp: extern "C" fn(u32) -> usize = std::mem::transmute(base + sites::WANTED_COMPONENT.rva as usize);
        if on {
            clear(ped);
        }
        max(if on { 0 } else { modcmd::WANTED_MAX as i32 });
        let w = comp(entity_id(ped));
        if w == 0 {
            return Err("the player has no wanted state".into());
        }
        let f = (w + WANTED_FLAGS) as *mut u32;
        let v = f.read_unaligned();
        f.write_unaligned(if on { v | WANTED_IGNORE_CRIMES } else { v & !WANTED_IGNORE_CRIMES });
    }
    Ok(())
}

/// Puts never wanted back if the game (a mission) changed the highest level or the flag.
///
/// # Safety
/// `ped` the live player, on its thread.
unsafe fn keep_never_wanted(st: &mut State, base: usize, ped: usize, frame_no: u64) -> Result<(), String> {
    let g = st.pin(base, sites::WANTED_MAX_LEVEL_GLOBAL)?;
    st.need(base, &[sites::WANTED_COMPONENT, sites::WANTED_IGNORE_READ])?;
    // SAFETY: the highest-level word (pinned) and the wanted flags (pinned by WANTED_IGNORE_READ).
    let fine = unsafe {
        let comp: extern "C" fn(u32) -> usize = std::mem::transmute(base + sites::WANTED_COMPONENT.rva as usize);
        let w = comp(entity_id(ped));
        (g as *const i32).read_volatile() == 0 && w != 0 && ((w + WANTED_FLAGS) as *const u32).read_unaligned() & WANTED_IGNORE_CRIMES != 0
    };
    if !fine {
        // SAFETY: the caller's.
        unsafe { set_never_wanted(st, base, ped, true) }?;
        say(false, &format!("VICTORY_MOD: never wanted applied again (the game had changed it; frame {frame_no})"));
    }
    Ok(())
}

/// The game clock (hour, minute), when its struct is where recorded.
fn clock(st: &mut State, base: usize) -> Option<(u32, u32)> {
    let c = st.pin(base, sites::GAME_CLOCK).ok()?;
    // SAFETY: the clock struct (pinned): two u32 fields.
    Some(unsafe { (((c + CLOCK_HOUR) as *const u32).read_volatile(), ((c + CLOCK_MINUTE) as *const u32).read_volatile()) })
}

/// The game's index of a weather hash (`None`: no such type), checked against its type count.
fn weather_index(st: &mut State, base: usize, hash: u32) -> Result<Option<u32>, String> {
    st.need(base, &[sites::WEATHER_INDEX])?;
    let count = st.pin(base, sites::WEATHER_COUNT)?;
    // SAFETY: Weather_GetTypeIndexFromHash (checked); the type count word (pinned).
    let (i, n) = unsafe {
        let f: extern "C" fn(u32) -> i32 = std::mem::transmute(base + sites::WEATHER_INDEX.rva as usize);
        (f(hash), (count as *const u32).read_volatile())
    };
    Ok(u32::try_from(i).ok().filter(|&i| i < n))
}

/// This weather now, held (VIper `set_weather`: reset the forced transition, then set-now-persist).
fn set_weather(st: &mut State, base: usize, idx: u32) -> Result<(), String> {
    st.need(base, &[sites::WEATHER_RESET_FORCED, sites::WEATHER_SET_NOW_PERSIST])?;
    // SAFETY: the two weather functions (checked) with VIper's arguments.
    unsafe {
        let reset: extern "C" fn(u8) = std::mem::transmute(base + sites::WEATHER_RESET_FORCED.rva as usize);
        let set: extern "C" fn(u32, u8) = std::mem::transmute(base + sites::WEATHER_SET_NOW_PERSIST.rva as usize);
        reset(1);
        set(idx, 0);
    }
    Ok(())
}

/// The held weather's index now persisting, if any.
fn persisting_weather(st: &mut State, base: usize) -> Option<i32> {
    let p = st.pin(base, sites::WEATHER_PERSIST).ok()?;
    // SAFETY: the persisting index word (pinned).
    Some(unsafe { (p as *const i32).read_volatile() })
}

/// Re-applies the held states the game (a mission, a script) may have
/// changed; a state that cannot be re-applied is dropped, with a warning.
fn keep(st: &mut State, base: usize, ped: usize, frame_no: u64) {
    if let Some((on, _)) = st.held.invincible {
        // SAFETY: the live player, on its thread (sites checked inside).
        let now = unsafe { damage_multipliers(st, base, ped) };
        let again = match now {
            Err(e) => Err(e),
            Ok(m) if on == ped && m == Some([0.0; DAMAGE_TYPES]) => Ok(false),
            Ok(m) => {
                if on != ped {
                    st.held.invincible = Some((ped, m.unwrap_or([1.0; DAMAGE_TYPES])));
                }
                // SAFETY: as above.
                unsafe { set_damage_multipliers(st, base, ped, &[0.0; DAMAGE_TYPES]) }.map(|()| true)
            }
        };
        match again {
            Ok(true) => say(false, &format!("VICTORY_MOD: invincible applied again (a new player ped, or the game changed it; frame {frame_no})")),
            Ok(false) => {}
            Err(e) => {
                st.held.invincible = None;
                say(true, &format!("VICTORY_MOD: invincible dropped: {e}"));
            }
        }
    }
    if st.held.never_wanted {
        // SAFETY: the live player, on its thread.
        if let Err(e) = unsafe { keep_never_wanted(st, base, ped, frame_no) } {
            st.held.never_wanted = false;
            say(true, &format!("VICTORY_MOD: never wanted dropped: {e}"));
        }
    }
    if st.held.freeze {
        match st.pin(base, sites::CLOCK_PAUSED_BY_SCRIPT) {
            Ok(b) => {
                // SAFETY: the pause-by-script byte (pinned).
                let running = unsafe { (b as *const u8).read_volatile() } & 1 == 0;
                if running {
                    // SAFETY: as above; a flag the game reads each frame.
                    unsafe { (b as *mut u8).write_volatile(1) };
                    say(false, &format!("VICTORY_MOD: the clock was started again by the game; stopped again (frame {frame_no})"));
                }
            }
            Err(e) => {
                st.held.freeze = false;
                say(true, &format!("VICTORY_MOD: time freeze dropped: {e}"));
            }
        }
    }
    if let Some((_, idx, name)) = st.held.weather.clone()
        && persisting_weather(st, base) != Some(idx as i32)
    {
        match set_weather(st, base, idx) {
            Ok(()) => say(false, &format!("VICTORY_MOD: the weather was changed by the game; {name} set again (frame {frame_no})")),
            Err(e) => {
                st.held.weather = None;
                say(true, &format!("VICTORY_MOD: held weather dropped: {e}"));
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn start_warp(
    st: &mut State,
    base: usize,
    ped: usize,
    frame_no: u64,
    line: &str,
    x: f32,
    y: f32,
    z: Option<f32>,
    heading: Option<f32>,
) -> Result<String, String> {
    st.need(base, &MOVE_SITES)?;
    st.need(base, &GROUND_SITES)?;
    // SAFETY: the vtable word of the live player.
    if unsafe { (ped as *const usize).read() } != base + CPED_VTABLE as usize {
        return Err("the player's vtable is not the known CPed one".into());
    }
    if let Some(old) = st.warp.take() {
        say(false, &format!("VICTORY_MOD: the teleport {:?} still waiting for its ground is replaced", old.line));
    }
    // SAFETY: the live player, on its thread (sites checked above).
    let vehicle = unsafe { ped_vehicle(base, ped) };
    let mut world = TitleWorld { base, ent: if vehicle != 0 { vehicle } else { ped } };
    // SAFETY: as above.
    let (from, _) = unsafe { entity_place(ped) };
    let (warp, step) = Warp::start(&mut world, x, y, z, heading, now_ms(st));
    let what = if vehicle != 0 { "the player's vehicle" } else { "the player" };
    let dist = ((from[0] - x).powi(2) + (from[1] - y).powi(2)).sqrt();
    match step {
        WarpStep::Done { at, .. } => Ok(format!("{what} moved {dist:.0} m to x={:.2} y={:.2} z={:.2} (ground loaded)", at[0], at[1], at[2])),
        WarpStep::Holding => {
            let held = warp.hold_z;
            st.warp = Some(WarpJob { warp, ped, vehicle, line: line.to_owned() });
            say(false, &format!("VICTORY_MOD: teleport {dist:.0} m (frame {frame_no}): held at x={x:.2} y={y:.2} z={held:.2} until its ground loads"));
            Ok(format!("{what} moved {dist:.0} m; held at z={held:.2} until the ground there loads"))
        }
        WarpStep::TimedOut { at, why } => Ok(format!("{what} at x={:.2} y={:.2} z={:.2}: {why}", at[0], at[1], at[2])),
        WarpStep::Failed(e) => Err(e),
    }
}

fn step_warp(st: &mut State, base: usize, ped: usize, frame_no: u64) {
    let Some(job) = st.warp.as_mut() else { return };
    // The same player (and vehicle), still there.
    // SAFETY: the live player, on its thread (the move sites were checked when the warp started).
    let same = ped == job.ped && (job.vehicle == 0 || unsafe { ped_vehicle(base, ped) } == job.vehicle);
    if !same {
        let line = job.line.clone();
        st.warp = None;
        say(true, &format!("VICTORY_MOD: teleport {line:?} dropped: the player or its vehicle changed (frame {frame_no})"));
        set_last_result(st, "dropped: the player or its vehicle changed while it waited".into());
        return;
    }
    let mut world = TitleWorld { base, ent: if job.vehicle != 0 { job.vehicle } else { job.ped } };
    let now = st.started.elapsed().as_millis() as u64;
    let step = job.warp.step(&mut world, now);
    let waited = now.saturating_sub(job.warp.since_ms);
    let text = match step {
        WarpStep::Holding => return,
        WarpStep::Done { at, frames, .. } => {
            format!("ok: on the ground at x={:.2} y={:.2} z={:.2} after {frames} frames ({waited} ms)", at[0], at[1], at[2])
        }
        WarpStep::TimedOut { at, why } => format!("ok, but {why}: x={:.2} y={:.2} z={:.2} after {waited} ms", at[0], at[1], at[2]),
        WarpStep::Failed(e) => format!("refused: {e}"),
    };
    let line = job.line.clone();
    st.warp = None;
    say(text.starts_with("refused"), &format!("VICTORY_MOD: teleport {line:?} (frame {frame_no}): {text}"));
    set_last_result(st, text);
}

fn set_last_result(st: &mut State, text: String) {
    if let Some(l) = st.last.as_mut() {
        l.result = text;
    }
}

fn write_status(st: &mut State, base: usize, ped: usize, frame_no: u64) {
    let Some(path) = st.status.clone() else { return };
    let mut s = StatusText::default();
    let unix_ms = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_millis());
    s.put("game", "running").put("updated_unix_ms", unix_ms).put("frame", frame_no);
    if ped == 0 {
        s.put("player", 0);
    } else {
        // SAFETY: the live player, on its thread.
        let (p, h) = unsafe { entity_place(ped) };
        s.put("player", 1)
            .put("x", format!("{:.2}", p[0]))
            .put("y", format!("{:.2}", p[1]))
            .put("z", format!("{:.2}", p[2]))
            .put("heading", format!("{h:.3}"))
            .put("heading_deg", format!("{:.0}", h.to_degrees().rem_euclid(360.0)));
        if st.need(base, &[sites::PED_COMPONENT_51, sites::VEHICLE_FROM_ID]).is_ok() {
            // SAFETY: the live player; both sites checked.
            let v = unsafe { ped_vehicle(base, ped) };
            s.put("in_vehicle", u8::from(v != 0));
        }
        if st.need(base, &[sites::ENTITY_HEALTH_AND_MAX]).is_ok() {
            let (mut hp, mut max) = (0.0f32, 0.0f32);
            // SAFETY: Entity_GetHealthAndMax (checked) on the live player.
            unsafe {
                let f: extern "C" fn(usize, *mut f32, *mut f32) = std::mem::transmute(base + sites::ENTITY_HEALTH_AND_MAX.rva as usize);
                f(ped, &mut hp, &mut max);
            }
            s.put("health", format!("{hp:.0}")).put("max_health", format!("{max:.0}"));
        }
        if st.need(base, &[sites::WANTED_COMPONENT, sites::WANTED_LEVEL_READ]).is_ok() {
            // SAFETY: the live player; both sites checked.
            if let Some(l) = unsafe { wanted_level(base, ped) } {
                s.put("wanted", l.clamp(0, modcmd::WANTED_MAX as i32));
            }
        }
        s.put("invincible", if st.held.invincible.is_some() { "on" } else { "off" });
        s.put("never_wanted", if st.held.never_wanted { "on" } else { "off" });
    }
    if let Some((hh, mm)) = clock(st, base) {
        s.put("clock", format!("{hh:02}:{mm:02}"));
    }
    s.put("freeze_held", if st.held.freeze { "on" } else { "off" });
    s.put("weather_held", st.held.weather.as_ref().map_or("none", |w| w.2.as_str()));
    if let Ok(b) = st.pin(base, sites::CLOCK_PAUSED_BY_SCRIPT) {
        // SAFETY: the pause-by-script byte (pinned).
        s.put("clock_frozen", unsafe { (b as *const u8).read_volatile() } & 1);
    }
    if let (Ok(persist), Ok(types), Ok(count)) =
        (st.pin(base, sites::WEATHER_PERSIST), st.pin(base, sites::WEATHER_TYPES), st.pin(base, sites::WEATHER_COUNT))
    {
        // SAFETY: the three weather words (pinned); an entry read only below the count.
        let held = unsafe {
            let i = (persist as *const i32).read_volatile();
            let n = (count as *const u32).read_volatile() as usize;
            let t = (types as *const usize).read_volatile();
            usize::try_from(i).ok().filter(|&i| i < n.min(512) && t != 0).map(|i| ((t + WEATHER_TYPE_SIZE * i) as *const u32).read_unaligned())
        };
        s.put(
            "weather",
            match held {
                Some(h) => modcmd::weather_name(h).map_or(format!("{h:#010x}"), str::to_owned),
                None => "cycle".into(),
            },
        );
    }
    match &st.warp {
        Some(j) => s.put("teleport", format!("holding {} ms for the ground at x={:.1} y={:.1}", now_ms(st).saturating_sub(j.warp.since_ms), j.warp.x, j.warp.y)),
        None => s.put("teleport", "idle"),
    };
    if let Some(l) = &st.last {
        s.put("last_command", &l.line).put("last_source", l.source).put("last_frame", l.frame).put("last_result", &l.result);
    }
    s.put("commands_ok", st.ok).put("commands_refused", st.refused);
    if let Err(e) = modcmd::write_atomic(&path, s.text())
        && !std::mem::replace(&mut st.status_said, true)
    {
        say(true, &format!("VICTORY_MOD: the status file {} could not be written ({e}); tried again twice a second", path.display()));
    }
}
