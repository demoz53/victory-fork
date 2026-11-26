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
    /// `void Vehicle_RequestFix(CVehicle*)`: the game's own full repair (what its heal cheat uses). The repair systems apply it on the next frame.
    pub const VEHICLE_REQUEST_FIX: Site = Site { name: "VEHICLE_REQUEST_FIX", rva: 0x1d1b1c0, bytes: &[0x56, 0x48, 0x83, 0xec, 0x20, 0x48, 0x89, 0xce, 0x8b, 0x89, 0x9c, 0x00, 0x00, 0x00, 0xe8, 0x6d, 0xe4, 0x32, 0x03, 0x48, 0x85, 0xc0, 0x74, 0x08] };
    /// `PartHealthComp* (u32 entityId)`: `u32` count at [`VEH_PARTS_COUNT`], 20-byte entries at `*(+VEH_PARTS)`.
    pub const VEHICLE_PART_HEALTH: Site = Site { name: "VEHICLE_PART_HEALTH", rva: 0x50572a0, bytes: &[0x89, 0xc8, 0x25, 0x00, 0x00, 0x3e, 0x00, 0x3d, 0x00, 0x00, 0x04, 0x00, 0x75, 0x3a, 0x81, 0xe1, 0xff, 0xff, 0x01, 0x00, 0x0f, 0xaf, 0x0d, 0x29] };
    /// `void PartHealth_RestoreToMax(entry)`.
    pub const VEHICLE_PART_RESTORE: Site = Site { name: "VEHICLE_PART_RESTORE", rva: 0x3d2e580, bytes: &[0xc5, 0xfa, 0x10, 0x09, 0xe9, 0xf7, 0xc3, 0x31, 0xfc] };
    /// `ShaderParams* Entity_GetShaderParams(drawable)`; the drawable is `*(entity + ENTITY_DRAWABLE)`.
    pub const ENTITY_SHADER_PARAMS: Site = Site { name: "ENTITY_SHADER_PARAMS", rva: 0x4f56f90, bytes: &[0x48, 0x83, 0xc1, 0x18, 0xba, 0x15, 0x00, 0x00, 0x00, 0xe9, 0x32, 0x96, 0x11, 0x00] };
    /// `void ShaderParams_SetDirt(params, float dirt (xmm1, 0..1), u8 0)`.
    pub const SHADER_SET_DIRT: Site = Site { name: "SHADER_SET_DIRT", rva: 0x1208100, bytes: &[0x45, 0x84, 0xc0, 0x74, 0x26, 0xc5, 0xfa, 0x11, 0x89, 0x7c, 0x06, 0x00, 0x00, 0xc5, 0xf8, 0x28, 0xc1, 0xc5, 0xfa, 0x10, 0x89, 0x78, 0x06, 0x00] };
    /// `u64 VehicleModels_HashAt(u32 n)`, the model hash of the n-th vehicle model: `mov rdx, [rip+X]` where X ([`VEHICLE_MODEL_POOL`]) holds the vehicle model pool.
    pub const VEHICLE_MODEL_HASH_AT: Site = Site { name: "VEHICLE_MODEL_HASH_AT", rva: 0x4aac670, bytes: &[0x56, 0x57, 0x48, 0x8b, 0x15, 0xc7, 0xd5, 0xa0, 0x05, 0x31, 0xc0, 0x39, 0x4a, 0x10, 0x76, 0x7a, 0x44, 0x8b, 0x42, 0x0c, 0x45, 0x85, 0xc0, 0x74] };
    /// `const char* Text_GetString(u32 label, bool emptyIfMissing)`: the UTF-8 text for a label in the player's language, from the loaded text blocks.
    pub const TEXT_GET_STRING: Site = Site { name: "TEXT_GET_STRING", rva: 0x10fbf80, bytes: &[0x48, 0x83, 0xec, 0x28, 0x41, 0x89, 0xd0, 0x89, 0xca, 0x48, 0x8d, 0x4c, 0x24, 0x20, 0x48, 0xb8, 0x00, 0x00, 0x00, 0x00, 0xff, 0xff, 0xff, 0xff] };
    /// `CBaseModelInfo* GetArchetypeFromHashKey(u64 hash, u32* outModelId, bool)`.
    pub const GET_ARCHETYPE_FROM_HASH: Site = Site { name: "GET_ARCHETYPE_FROM_HASH", rva: 0x00bf810, bytes: &[0x56, 0x57, 0x55, 0x53, 0x48, 0x85, 0xc9, 0x74, 0x4b, 0x44, 0x8b, 0x15, 0xb4, 0x64, 0x61, 0x07] };
    /// `bool HasModelLoaded(u32 modelId)`.
    pub const HAS_MODEL_LOADED: Site = Site { name: "HAS_MODEL_LOADED", rva: 0x4f16110, bytes: &[0x41, 0x57, 0x41, 0x56, 0x41, 0x55, 0x41, 0x54, 0x56, 0x57, 0x55, 0x53, 0x48, 0x83, 0xec, 0x28, 0x89, 0xcd, 0x81, 0xe5, 0xff, 0xff, 0x3f, 0x00] };
    /// `void ModelRequest_Request(u16* request, const u32* modelId, const Defaults*, const u32* zero, u8 flags, u8 1, u8 0, const u8 zero16[16])`: asks the streaming system for a
    pub const MODEL_REQUEST: Site = Site { name: "MODEL_REQUEST", rva: 0x07a0f90, bytes: &[0x41, 0x57, 0x41, 0x56, 0x41, 0x55, 0x41, 0x54, 0x56, 0x57, 0x55, 0x53, 0x48, 0x83, 0xec, 0x38, 0x40, 0x8a, 0xac, 0x24, 0xb0, 0x00, 0x00, 0x00] };
    /// In the vehicle cheat's blocking request: `lea r8, [rip+X]`, X = [`MODEL_REQUEST_DEFAULTS`].
    pub const MODEL_REQUEST_DEFAULTS_READ: Site = Site { name: "MODEL_REQUEST_DEFAULTS_READ", rva: 0x3b25469, bytes: &[0x4c, 0x8d, 0x05, 0x7c, 0x62, 0xfc, 0x06] };
    /// `bool ModelRequest_IsLoaded(const u16* request)`: the requested model is in memory.
    pub const MODEL_REQUEST_LOADED: Site = Site { name: "MODEL_REQUEST_LOADED", rva: 0x51e7ff0, bytes: &[0x56, 0x48, 0x83, 0xec, 0x20, 0x48, 0x89, 0xce, 0xe8, 0xe3, 0x00, 0x12, 0x00, 0x84, 0xc0, 0x74, 0x0d, 0x48, 0x89, 0xf1, 0x48, 0x83, 0xc4, 0x20] };
    /// `void ModelRequest_Release(u16* request)`: drops the request (null-safe on an empty word, which it leaves 0). The model may then be streamed out once nothing else uses it.
    pub const MODEL_REQUEST_RELEASE: Site = Site { name: "MODEL_REQUEST_RELEASE", rva: 0x52d56f0, bytes: &[0x56, 0x48, 0x83, 0xec, 0x20, 0x0f, 0xb7, 0x01, 0xa9, 0xff, 0x3f, 0x00, 0x00, 0x75, 0x06, 0x48, 0x83, 0xc4, 0x20, 0x5e, 0xc3] };
    /// `VehicleCreateParams::VehicleCreateParams(params*)`.
    pub const VEH_PARAMS_CTOR: Site = Site { name: "VEH_PARAMS_CTOR", rva: 0x5285ab0, bytes: &[0x56, 0x57, 0x53, 0x48, 0x83, 0xec, 0x30, 0xc5, 0xf8, 0x29, 0x74, 0x24, 0x20, 0x31, 0xc0] };
    /// `CVehicle* CreateVehicleFromParams(params*)`: the synchronous creation vehicle population uses.
    pub const CREATE_VEHICLE_FROM_PARAMS: Site = Site { name: "CREATE_VEHICLE_FROM_PARAMS", rva: 0x2363cb0, bytes: &[0x56, 0x48, 0x83, 0xec, 0x20, 0x48, 0x89, 0xce, 0xe8, 0xd3, 0xff, 0xff, 0xff, 0x84, 0xc0, 0x74, 0x4c] };
    /// `bool World_RegisterNewEntity(manager, entity, params*, int slot /* -1 = allocate */)`: what vehicle population calls right after creation.
    pub const WORLD_REGISTER_NEW_ENTITY: Site = Site { name: "WORLD_REGISTER_NEW_ENTITY", rva: 0x0a7e7e0, bytes: &[0x56, 0x57, 0x53, 0x48, 0x83, 0xec, 0x20, 0x4c, 0x89, 0xc3, 0x48, 0x89, 0xd7, 0x48, 0x89, 0xce] };
    /// `mov rcx, [rip+X]`: X ([`WORLD_ENTITY_MANAGER`]) is the world entity manager.
    pub const WORLD_MANAGER_READ: Site = Site { name: "WORLD_MANAGER_READ", rva: 0x0c2fa70, bytes: &[0x48, 0x8b, 0x0d, 0xb9, 0xb2, 0xd7, 0x08] };
    /// `u64 World_Add(CEntity*, u32 location, u8 a3, u8 skip_notify)`: inserts an entity into the world scene and (when `skip_notify == 0`) notifies components, including adding
    pub const WORLD_ADD: Site = Site { name: "WORLD_ADD", rva: 0x509c0b0, bytes: &[0x48, 0x83, 0xec, 0x58, 0x31, 0xc0, 0x44, 0x88, 0x4c, 0x24, 0x28, 0x45, 0x31, 0xc9, 0x66, 0xc7, 0x44, 0x24, 0x30, 0xff, 0xff, 0xc7, 0x44, 0x24] };
    /// `mov edx, [rip+X]`: X ([`WORLD_DEFAULT_LOCATION`]) is the default world location passed to [`WORLD_ADD`].
    pub const WORLD_DEFAULT_LOCATION_READ: Site = Site { name: "WORLD_DEFAULT_LOCATION_READ", rva: 0x066e152, bytes: &[0x8b, 0x15, 0x98, 0x7c, 0x4d, 0x0a] };
    /// `void Phys_Wake(CEntity*, void* physics_world)`: wakes a sleeping physics instance (state 2 -> 1 active collider) so gravity, collisions and velocity take effect.
    pub const PHYS_WAKE: Site = Site { name: "PHYS_WAKE", rva: 0x15a7a10, bytes: &[0x48, 0x83, 0xe1, 0xf0, 0x48, 0x8b, 0x49, 0x38, 0x48, 0x85, 0xc9, 0x74, 0x16, 0x80, 0xb9, 0x56, 0x01, 0x00, 0x00, 0x02, 0x0f, 0x84, 0x46, 0x0d] };
    /// `lea rdx, [rip+X]` in `VehicleComp_OnAddedToWorld`: X ([`PHYSICS_WORLD`]) is the physics world passed to [`PHYS_WAKE`].
    pub const PHYSICS_WORLD_READ: Site = Site { name: "PHYSICS_WORLD_READ", rva: 0x080a8eb, bytes: &[0x48, 0x8d, 0x15, 0xd6, 0x97, 0x33, 0x0a] };
    /// `mov eax, [rip+X]` in the script vehicle creator: X ([`VEHICLE_COUNTER`]) is the creation counter stored into each params block.
    pub const VEHICLE_COUNTER_READ: Site = Site { name: "VEHICLE_COUNTER_READ", rva: 0x4ab8bed, bytes: &[0x8b, 0x05, 0x5d, 0x2b, 0x76, 0x05] };
    /// Destructor of the params sub-object at [`vehicle_params::SUB_OBJECT`] (a `jmp` thunk).
    pub const VEH_PARAMS_SUB_DTOR: Site = Site { name: "VEH_PARAMS_SUB_DTOR", rva: 0x52858a0, bytes: &[0xe9, 0x2b, 0x00, 0x00, 0x00, 0xcc] };
    /// Releases the object pointer at [`vehicle_params::RESOURCE`] (null-safe).
    pub const VEH_PARAMS_RELEASE: Site = Site { name: "VEH_PARAMS_RELEASE", rva: 0x5224170, bytes: &[0x48, 0x85, 0xc9, 0x0f, 0x85, 0x07, 0x00, 0x00, 0x00, 0xc3] };
    /// The game heap's free (a `jmp` thunk).
    pub const GAME_FREE: Site = Site { name: "GAME_FREE", rva: 0x4d9c130, bytes: &[0xe9, 0x1b, 0x00, 0x00, 0x00, 0xcc] };
    /// `void SetPedIntoVehicle_Impl(CPed*, CVehicle*, bool anySeat, u32 seat)`: what the put-ped-in-vehicle
    /// script command runs (it clears the ped's tasks first).
    pub const PED_INTO_VEHICLE: Site = Site { name: "PED_INTO_VEHICLE", rva: 0x4c03280, bytes: &[0x41, 0x57, 0x41, 0x56, 0x56, 0x57, 0x55, 0x53, 0x48, 0x83, 0xec, 0x38, 0x44, 0x89, 0xcd, 0x45, 0x89, 0xc6, 0x48, 0x89, 0xd6, 0x49, 0x89, 0xcf] };
    /// The exception table's first entries (function start, end, unwind RVAs).
    pub const EXCEPTION_TABLE: Site = Site { name: "EXCEPTION_TABLE", rva: 0xabcf000, bytes: &[0x00, 0x10, 0x00, 0x00, 0xec, 0x10, 0x00, 0x00, 0x58, 0x0d, 0x7a, 0x06, 0xf0, 0x10, 0x00, 0x00, 0x34, 0x11, 0x00, 0x00, 0xc4, 0x0c, 0x7a, 0x06] };
    /// `void EntityToScriptHandle(CEntity*, Handle16* out, int mode)`; the natives use mode 2.
    pub const ENTITY_TO_SCRIPT_HANDLE: Site = Site { name: "ENTITY_TO_SCRIPT_HANDLE", rva: 0x02a6520, bytes: &[0x41, 0x56, 0x56, 0x57, 0x53, 0x48, 0x83, 0xec, 0x38, 0xc5, 0xf8, 0x57, 0xc0, 0x48, 0x89, 0xd6] };
    pub const VEHICLE_MODEL_POOL: Pin = Pin { site: VEHICLE_MODEL_HASH_AT, disp: 5, end: 9, target: 0x0a4b_9c40 };
    pub const MODEL_REQUEST_DEFAULTS: Pin = Pin { site: MODEL_REQUEST_DEFAULTS_READ, disp: 3, end: 7, target: 0x0aae_b6ec };
    pub const WORLD_ENTITY_MANAGER: Pin = Pin { site: WORLD_MANAGER_READ, disp: 3, end: 7, target: 0x099a_ad30 };
    pub const WORLD_DEFAULT_LOCATION: Pin = Pin { site: WORLD_DEFAULT_LOCATION_READ, disp: 2, end: 6, target: 0x0ab4_5df0 };
    pub const PHYSICS_WORLD: Pin = Pin { site: PHYSICS_WORLD_READ, disp: 3, end: 7, target: 0x0ab4_40c8 };
    pub const VEHICLE_COUNTER: Pin = Pin { site: VEHICLE_COUNTER_READ, disp: 2, end: 6, target: 0x0a21_b750 };
    /// The vehicle pool object, read by VehicleFromEntityId (`mov rdx, [rip+X]`).
    pub const VEHICLE_POOL: Pin = Pin { site: VEHICLE_FROM_ID, disp: 14, end: 18, target: 0x0abb_e8c8 };
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
    spawn: Option<SpawnJob>,
    /// The game's vehicle models, read once.
    models: Option<Vec<VehicleModel>>,
    /// The vehicle last spawned and its pool slot (checked alive before use), or (0, 0).
    spawned: (usize, u32),
    /// `wanted n` waiting for the game to show the level: (n, since, line).
    wanted_check: Option<(u32, Instant, String)>,
    /// Times never wanted was put back since it was set (said once, then counted).
    never_again: u32,
    /// Every script-command handler (hash, rva), indexed on the first `native hash:`.
    handlers: Option<Vec<(u64, u32)>>,
    native_result: Option<String>,
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
        spawn: None,
        models: None,
        spawned: (0, 0),
        wanted_check: None,
        never_again: 0,
        handlers: None,
        native_result: None,
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

// ---- vehicles -------------------------------------------------------------------------

/// The script vehicle creator's params block (VIper `vehicle_params`); the
/// world entity manager keeps a pointer to it for a while after creation.
#[repr(C, align(32))]
struct VehicleParams([u8; VP_SIZE]);

const VP_SIZE: usize = 0x800;
/// How long a registered params block is kept before it is cleaned up.
const VP_KEEP: Duration = Duration::from_secs(3);
const VP_MODEL_ID: usize = 0x630;
const VP_KIND: usize = 0x634;
const VP_KIND_VALUE: u64 = 0x0000_0007_0000_0005;
const VP_ENTITY_TYPE: usize = 0x644;
const VP_ENTITY_TYPE_VALUE: u16 = 6;
const VP_MATRIX: usize = 0x650;
const VP_COUNTER: usize = 0x2d0;
const VP_HAS_VARIATION: usize = 0x129;
/// The script creator's other defaults, after the constructor: (offset, bytes, value).
const VP_DEFAULTS: &[(usize, u8, u32)] = &[
    (0x730, 2, 0x10),
    (0x691, 1, 1),
    (0x693, 1, 0),
    (0x694, 1, 0),
    (0x696, 1, 1),
    (0x6e0, 1, 6),
    (0x2d8, 4, u32::MAX),
    (0x2dc, 4, 0),
    (0x098, 4, 0),
    (0x09d, 1, 0),
    (0x6c4, 1, 0),
    (0x6c6, 1, 1),
    (0x120, 4, 0),
    (0x0d8, 4, 0),
    (0x0dc, 4, 0),
    (0x6d6, 1, 0),
    (0x6d8, 4, 0), // f32 0.0
    (0x6d7, 1, 0),
    (0x695, 1, 1),
    (0x277, 1, 1),
];
/// Cleanup offsets, in the script creator's order.
const VP_SUB_OBJECT: usize = 0x280;
const VP_SUB_OBJECT_TAG: usize = 0xd0;
const VP_SUB_OBJECT_TAG_VALUE: u64 = 0x16;
const VP_BUFFER_A_TAG: usize = 0x158;
const VP_BUFFER_A_TAG_VALUE: u64 = 0x10;
const VP_BUFFER_A_LEN: usize = 0x144;
const VP_BUFFER_A_CAP: usize = 0x140;
const VP_BUFFER_A: usize = 0x148;
const VP_RESOURCE: usize = 0x130;
const VP_BUFFER_B_LEN: usize = 0xcc;
const VP_BUFFER_B: usize = 0xd0;
/// Archetype fields (VIper `archetype`, `vehicle_model`).
const ARCH_HASH: usize = 0x60;
const ARCH_TYPE: usize = 153;
const ARCH_TYPE_VEHICLE: u8 = 5;
const INVALID_MODEL_ID: u32 = 0x3f_ffff;
const NAME_LABEL: usize = 0x118;
const MAKE_LABEL: usize = 0x11c;
/// Pool layout (VIper `pool`).
const POOL_STORAGE: usize = 0x00;
const POOL_STRIDE: usize = 0x08;
const POOL_CAPACITY: usize = 0x0c;
const POOL_FLAGS: usize = 0x28;
const POOL_LIVE_MASK: usize = 0x30;
const POOL_GENERATION_MASK: u32 = 0x07fe_0000;
/// The entity's physics instance, set by the world entity manager.
const PHYSICS_INST: usize = 0x38;
/// Model request flags of the script request-model command; the request index bits.
const REQUEST_FLAGS_SCRIPT: u8 = 6;
const REQUEST_INDEX_MASK: u16 = 0x3fff;
/// Waits: for the model to stream in, for the physics instance.
const MODEL_WAIT: Duration = Duration::from_secs(10);
const PHYSICS_WAIT: Duration = Duration::from_secs(5);
/// Vehicle parts and the drawable (VIper VEH_PARTS_COUNT, VEH_PARTS, VEH_PART_SIZE, ENTITY_DRAWABLE).
const VEH_PARTS_COUNT: usize = 196;
const VEH_PARTS: usize = 200;
const VEH_PART_SIZE: usize = 20;
const ENTITY_DRAWABLE: usize = 0x28;

const SPAWN_SITES: [Site; 18] = [
    sites::GET_ARCHETYPE_FROM_HASH,
    sites::HAS_MODEL_LOADED,
    sites::MODEL_REQUEST,
    sites::MODEL_REQUEST_LOADED,
    sites::MODEL_REQUEST_RELEASE,
    sites::VEH_PARAMS_CTOR,
    sites::CREATE_VEHICLE_FROM_PARAMS,
    sites::WORLD_REGISTER_NEW_ENTITY,
    sites::WORLD_ADD,
    sites::PHYS_WAKE,
    sites::VEH_PARAMS_SUB_DTOR,
    sites::VEH_PARAMS_RELEASE,
    sites::GAME_FREE,
    sites::VEHICLE_FROM_ID,
    sites::GROUND_Z,
    sites::MODEL_REQUEST_DEFAULTS_READ,
    sites::VEHICLE_COUNTER_READ,
    sites::WORLD_MANAGER_READ,
];

/// A vehicle model the game defines.
#[derive(Clone)]
struct VehicleModel {
    hash: u64,
    name: String,
    make: String,
}

/// A spawn under way, over several frames.
struct SpawnJob {
    line: String,
    model: VehicleModel,
    id: u32,
    word: u16,
    since: Instant,
    /// Once made: the vehicle, its params (until [`VP_KEEP`]), whether it is in the world.
    made: Option<(usize, u32, Option<Box<VehicleParams>>, Instant, bool)>,
}

/// Reads a NUL-terminated text of at most 256 bytes.
///
/// # Safety
/// `p` null or a readable NUL-terminated text.
unsafe fn c_text(p: *const u8) -> String {
    if p.is_null() {
        return String::new();
    }
    let mut n = 0;
    // SAFETY: bounded walk of the text up to its NUL.
    unsafe {
        while n < 256 && *p.add(n) != 0 {
            n += 1;
        }
        String::from_utf8_lossy(std::slice::from_raw_parts(p, n)).into_owned()
    }
}

/// Every vehicle model the game defines (the vehicle model pool, as VIper
/// walks it) with its display name and make in the player's language.
fn vehicle_models(st: &mut State, base: usize) -> Result<Vec<VehicleModel>, String> {
    st.need(base, &[sites::TEXT_GET_STRING])?;
    let pool = st.pin(base, sites::VEHICLE_MODEL_POOL)?;
    let plausible = |p: usize| (0x1_0000..0x7fff_ffff_0000).contains(&p) && p & 7 == 0;
    let mut out = Vec::new();
    // SAFETY: the pool word (pinned) and the pool it points at, every pointer checked for plausibility, every slot
    // index below its capacity; Text_GetString (checked) copied at once.
    unsafe {
        let p = (pool as *const usize).read_volatile();
        if !plausible(p) {
            return Err("the vehicle model pool is not there yet".into());
        }
        let storage = ((p + POOL_STORAGE) as *const usize).read_unaligned();
        let stride = ((p + POOL_STRIDE) as *const u32).read_unaligned() as usize;
        let cap = (((p + POOL_CAPACITY) as *const u32).read_unaligned() as usize).min(1 << 17);
        let flags = ((p + POOL_FLAGS) as *const usize).read_unaligned();
        let mask = ((p + POOL_LIVE_MASK) as *const u32).read_unaligned();
        if !plausible(storage) || stride == 0 || !plausible(flags) {
            return Err("the vehicle model pool reads wrong".into());
        }
        let text: extern "C" fn(u32, u8) -> *const u8 = std::mem::transmute(base + sites::TEXT_GET_STRING.rva as usize);
        for i in 0..cap {
            if mask & !((flags + 4 * i) as *const u32).read_unaligned() != 0 {
                continue;
            }
            let mi = storage + stride * i;
            if ((mi + ARCH_TYPE) as *const u8).read() & 7 != ARCH_TYPE_VEHICLE {
                continue;
            }
            let hash = ((mi + ARCH_HASH) as *const u64).read_unaligned();
            let name = c_text(text(((mi + NAME_LABEL) as *const u32).read_unaligned(), 1));
            let make = c_text(text(((mi + MAKE_LABEL) as *const u32).read_unaligned(), 1));
            out.push(VehicleModel { hash, name, make });
        }
    }
    Ok(out)
}

fn squash(s: &str) -> String {
    s.chars().filter(|c| c.is_alphanumeric()).collect::<String>().to_lowercase()
}

/// The model a `spawn vehicle` names: its hash; or a display name (spaces and case
/// ignored, with or without the make), or a name whose joaat is the hash's low half.
fn find_model(models: &[VehicleModel], m: &modcmd::Model) -> Result<VehicleModel, String> {
    match m {
        modcmd::Model::Hash(h) => Ok(models.iter().find(|v| v.hash == *h).cloned().unwrap_or(VehicleModel { hash: *h, name: format!("{h:#018x}"), make: String::new() })),
        modcmd::Model::Index(i) => models.get(*i).cloned().ok_or_else(|| format!("vehicle list has {} lines, not #{i}", models.len())),
        modcmd::Model::Name(n) => {
            let key = squash(n);
            let j = modcmd::joaat(n);
            let hit = models
                .iter()
                .find(|v| squash(&v.name) == key)
                .or_else(|| models.iter().find(|v| squash(&format!("{}{}", v.make, v.name)) == key))
                .or_else(|| models.iter().find(|v| v.hash as u32 == j));
            hit.cloned().ok_or_else(|| format!("no vehicle model named {n:?} (vehicle list writes the game's names to a file)"))
        }
    }
}

/// The archetype of a model hash: (model id, is a vehicle).
fn lookup_model(base: usize, hash: u64) -> Option<(u32, bool)> {
    let mut id = INVALID_MODEL_ID;
    // SAFETY: GetArchetypeFromHashKey (checked by the caller); the archetype's type byte.
    unsafe {
        let f: extern "C" fn(u64, *mut u32, u8) -> usize = std::mem::transmute(base + sites::GET_ARCHETYPE_FROM_HASH.rva as usize);
        let arch = f(hash, &mut id, 0);
        if arch == 0 || id & INVALID_MODEL_ID == INVALID_MODEL_ID {
            return None;
        }
        Some((id, ((arch + ARCH_TYPE) as *const u8).read() & 7 == ARCH_TYPE_VEHICLE))
    }
}

fn start_spawn(st: &mut State, base: usize, line: &str, m: &modcmd::Model) -> Result<String, String> {
    if st.spawn.is_some() {
        return Err("a vehicle spawn is still under way".into());
    }
    st.need(base, &SPAWN_SITES)?;
    for p in [sites::MODEL_REQUEST_DEFAULTS, sites::VEHICLE_COUNTER, sites::WORLD_ENTITY_MANAGER, sites::WORLD_DEFAULT_LOCATION, sites::PHYSICS_WORLD, sites::VEHICLE_POOL] {
        st.pin(base, p)?;
    }
    let models = match st.models.clone() {
        Some(m) => m,
        None => {
            let m = vehicle_models(st, base)?;
            st.models = Some(m.clone());
            m
        }
    };
    let model = find_model(&models, m)?;
    let (id, vehicle) = lookup_model(base, model.hash).ok_or_else(|| format!("the game has no model {:#018x}", model.hash))?;
    if !vehicle {
        return Err(format!("{:#018x} is not a vehicle model", model.hash));
    }
    let defaults = base + sites::MODEL_REQUEST_DEFAULTS.target as usize;
    #[repr(C, align(16))]
    struct Zero16([u8; 16]);
    let (mut word, zero, extra) = (0u16, 0u32, Zero16([0; 16]));
    // SAFETY: ModelRequest_Request (checked) with VIper's arguments: (&word, &id, defaults (pinned), &0, 6, 1, 0, &zero16).
    unsafe {
        let f: extern "C" fn(*mut u16, *const u32, usize, *const u32, u8, u8, u8, *const Zero16) = std::mem::transmute(base + sites::MODEL_REQUEST.rva as usize);
        f(&mut word, &id, defaults, &zero, REQUEST_FLAGS_SCRIPT, 1, 0, &extra);
    }
    let what = format!("{} {} ({:#018x})", model.make, model.name, model.hash).trim().to_owned();
    st.spawn = Some(SpawnJob { line: line.to_owned(), model, id, word, since: Instant::now(), made: None });
    Ok(format!("{what}: model asked for; made in front of the player once it is loaded"))
}

/// Gives back a model request.
fn release_request(base: usize, word: &mut u16) {
    // SAFETY: ModelRequest_Release (checked at the spawn's start), null-safe on an empty word.
    unsafe {
        let f: extern "C" fn(*mut u16) = std::mem::transmute(base + sites::MODEL_REQUEST_RELEASE.rva as usize);
        f(word);
    }
}

/// Releases what the game put in a params block, in the script creator's order (VIper `VehicleParams::cleanup`).
fn cleanup_params(base: usize, mut params: Box<VehicleParams>) {
    let p = params.0.as_mut_ptr() as usize;
    // SAFETY: the three checked functions on our own block, as the game's creator cleans its own.
    unsafe {
        let sub_dtor: extern "C" fn(usize) = std::mem::transmute(base + sites::VEH_PARAMS_SUB_DTOR.rva as usize);
        let release: extern "C" fn(usize) = std::mem::transmute(base + sites::VEH_PARAMS_RELEASE.rva as usize);
        let free: extern "C" fn(usize) = std::mem::transmute(base + sites::GAME_FREE.rva as usize);
        let q = p + VP_SUB_OBJECT;
        ((q + VP_SUB_OBJECT_TAG) as *mut u64).write_unaligned(VP_SUB_OBJECT_TAG_VALUE);
        sub_dtor(q);
        ((p + VP_BUFFER_A_TAG) as *mut u64).write_unaligned(VP_BUFFER_A_TAG_VALUE);
        ((p + VP_BUFFER_A_LEN) as *mut u32).write_unaligned(0);
        let a = ((p + VP_BUFFER_A) as *const usize).read_unaligned();
        if a != 0 {
            free(a);
        }
        ((p + VP_BUFFER_A_CAP) as *mut u32).write_unaligned(0);
        ((p + VP_BUFFER_A) as *mut u64).write_unaligned(0);
        release(((p + VP_RESOURCE) as *const usize).read_unaligned());
        ((p + VP_BUFFER_B_LEN) as *mut u32).write_unaligned(0);
        let b = ((p + VP_BUFFER_B) as *const usize).read_unaligned();
        if b != 0 {
            free(b);
        }
    }
}

/// Creates the vehicle as the game's script creator fills its params (VIper `create_vehicle`).
fn create_vehicle(base: usize, id: u32, at: [f32; 3], heading: f32) -> Option<(usize, Option<Box<VehicleParams>>)> {
    let mut params = Box::new(VehicleParams([0; VP_SIZE]));
    let p = params.0.as_mut_ptr() as usize;
    let (s, c) = heading.sin_cos();
    let rows = [[c, s, 0.0, 0.0], [-s, c, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [at[0], at[1], at[2], 0.0]];
    // SAFETY: our own 0x800-byte block (32-aligned) filled at VIper's offsets; the checked constructor, creator and
    // register calls; the counter and manager words pinned at the spawn's start.
    unsafe {
        let ctor: extern "C" fn(usize) = std::mem::transmute(base + sites::VEH_PARAMS_CTOR.rva as usize);
        let create: extern "C" fn(usize) -> usize = std::mem::transmute(base + sites::CREATE_VEHICLE_FROM_PARAMS.rva as usize);
        let register: extern "C" fn(usize, usize, usize, i32) -> u8 = std::mem::transmute(base + sites::WORLD_REGISTER_NEW_ENTITY.rva as usize);
        ctor(p);
        for &(off, n, v) in VP_DEFAULTS {
            match n {
                1 => ((p + off) as *mut u8).write(v as u8),
                2 => ((p + off) as *mut u16).write_unaligned(v as u16),
                _ => ((p + off) as *mut u32).write_unaligned(v),
            }
        }
        ((p + VP_MODEL_ID) as *mut u32).write_unaligned(id);
        ((p + VP_KIND) as *mut u64).write_unaligned(VP_KIND_VALUE);
        ((p + VP_ENTITY_TYPE) as *mut u16).write_unaligned(VP_ENTITY_TYPE_VALUE);
        for (i, row) in rows.iter().enumerate() {
            ((p + VP_MATRIX + 16 * i) as *mut [f32; 4]).write_unaligned(*row);
        }
        let counter = (base + sites::VEHICLE_COUNTER.target as usize) as *mut u32;
        let n = counter.read_volatile().wrapping_add(1);
        counter.write_volatile(n);
        ((p + VP_COUNTER) as *mut u32).write_unaligned(n);
        ((p + VP_HAS_VARIATION) as *mut u8).write(0);
        let veh = create(p);
        if veh == 0 {
            cleanup_params(base, params);
            return None;
        }
        let world = ((base + sites::WORLD_ENTITY_MANAGER.target as usize) as *const usize).read_volatile();
        if world != 0 && register(world, veh, p, -1) != 0 {
            Some((veh, Some(params)))
        } else {
            cleanup_params(base, params);
            Some((veh, None))
        }
    }
}

/// Whether `veh` is still the live vehicle of its id (the generation-checked lookup).
/// The vehicle pool slot of `veh` (index | generation), when it is a live slot
/// (VIper `Pool::slot_of`; the pool word pinned by the caller). A vehicle is
/// the same live one while its slot stays the same: the entity id at +0x9c is
/// not usable for this right after creation (the first spawn in the game was
/// judged gone a frame after it was made, and never put in the world).
fn vehicle_slot(base: usize, veh: usize) -> Option<u32> {
    let plausible = |p: usize| (0x1_0000..0x7fff_ffff_0000).contains(&p) && p & 7 == 0;
    // SAFETY: the pool word (pinned) and its object's fields, each pointer checked, the index below the capacity.
    unsafe {
        let p = ((base + sites::VEHICLE_POOL.target as usize) as *const usize).read_volatile();
        if !plausible(p) {
            return None;
        }
        let storage = ((p + POOL_STORAGE) as *const usize).read_unaligned();
        let stride = ((p + POOL_STRIDE) as *const u32).read_unaligned() as usize;
        let cap = (((p + POOL_CAPACITY) as *const u32).read_unaligned() as usize).min(1 << 17);
        let flags = ((p + POOL_FLAGS) as *const usize).read_unaligned();
        if !plausible(storage) || stride == 0 || !plausible(flags) || veh < storage || !(veh - storage).is_multiple_of(stride) {
            return None;
        }
        let i = (veh - storage) / stride;
        if i >= cap {
            return None;
        }
        let f = ((flags + 4 * i) as *const u32).read_unaligned();
        let mask = ((p + POOL_LIVE_MASK) as *const u32).read_unaligned();
        (mask & !f == 0).then_some(i as u32 | (f & POOL_GENERATION_MASK))
    }
}

/// Whether `veh` is still the live vehicle of that pool slot.
fn vehicle_alive(base: usize, veh: usize, slot: u32) -> bool {
    veh != 0 && vehicle_slot(base, veh) == Some(slot)
}

/// One frame of a spawn under way.
fn step_spawn(st: &mut State, base: usize, ped: usize, frame_no: u64) {
    let Some(job) = st.spawn.as_mut() else { return };
    let name = format!("{} {}", job.model.make, job.model.name).trim().to_owned();
    let mut finish: Option<(bool, String)> = None;
    let mut spawned_now = (0, 0);
    match job.made.as_mut() {
        None => {
            let w = job.word;
            // SAFETY: ModelRequest_IsLoaded / HasModelLoaded (checked) on our request word and model id.
            let loaded = unsafe {
                let req: extern "C" fn(*const u16) -> u8 = std::mem::transmute(base + sites::MODEL_REQUEST_LOADED.rva as usize);
                let has: extern "C" fn(u32) -> u8 = std::mem::transmute(base + sites::HAS_MODEL_LOADED.rva as usize);
                (w & REQUEST_INDEX_MASK != 0 && req(&w) != 0) || has(job.id) != 0
            };
            if loaded && ped != 0 {
                // SAFETY: the live player, on its thread.
                let (here, h) = unsafe { entity_place(ped) };
                let mut at = modcmd::relative(here, h, 5.0, 0.0);
                let mut world = TitleWorld { base, ent: ped };
                at[2] = modcmd::World::ground(&mut world, at[0], at[1], here[2] + 5.0).map_or(here[2] + 0.5, |g| g + 0.5);
                let made = create_vehicle(base, job.id, at, h);
                release_request(base, &mut job.word);
                match made {
                    Some((veh, params)) => match vehicle_slot(base, veh) {
                        Some(slot) => {
                            say(false, &format!("VICTORY_MOD: {name} made at x={:.2} y={:.2} z={:.2} (frame {frame_no}, {} ms after asking; pool slot {slot:#x})", at[0], at[1], at[2], job.since.elapsed().as_millis()));
                            job.made = Some((veh, slot, params, Instant::now(), false));
                        }
                        None => {
                            // Not in the vehicle pool: leave it alone (never touched again), only clean our block.
                            if let Some(p) = params {
                                cleanup_params(base, p);
                            }
                            finish = Some((false, format!("refused: {name} was made at {veh:#x}, which is not a live vehicle pool slot; left alone")));
                        }
                    },
                    None => finish = Some((false, format!("refused: the game did not make {name} (vehicle budget or pool full?)"))),
                }
            } else if job.since.elapsed() >= MODEL_WAIT {
                release_request(base, &mut job.word);
                finish = Some((false, format!("refused: {name}'s model did not load in {} s", MODEL_WAIT.as_secs())));
            }
        }
        Some((veh, slot, params, at, in_world)) => {
            let (veh, slot) = (*veh, *slot);
            if !vehicle_alive(base, veh, slot) {
                if let Some(p) = params.take() {
                    cleanup_params(base, p);
                }
                finish = Some((false, format!("{name} was made, but its vehicle pool slot is no longer live: the game removed it")));
            } else {
                // SAFETY: the live vehicle we made (checked above).
                let phys = unsafe { ((veh + PHYSICS_INST) as *const usize).read_unaligned() };
                if !*in_world && phys != 0 {
                    // SAFETY: World_Add and Phys_Wake (checked) on the live vehicle, with the pinned location and world.
                    unsafe {
                        let loc = ((base + sites::WORLD_DEFAULT_LOCATION.target as usize) as *const u32).read_volatile();
                        let add: extern "C" fn(usize, u32, u8, u8) -> u64 = std::mem::transmute(base + sites::WORLD_ADD.rva as usize);
                        let wake: extern "C" fn(usize, usize) = std::mem::transmute(base + sites::PHYS_WAKE.rva as usize);
                        add(veh, loc, 0, 0);
                        wake(veh, base + sites::PHYSICS_WORLD.target as usize);
                    }
                    *in_world = true;
                } else if !*in_world && at.elapsed() >= PHYSICS_WAIT {
                    *in_world = true;
                    say(true, &format!("VICTORY_MOD: {name} has no physics after {} s: not woken", PHYSICS_WAIT.as_secs()));
                }
                if at.elapsed() >= VP_KEEP
                    && let Some(p) = params.take()
                {
                    cleanup_params(base, p);
                }
                if *in_world && params.is_none() {
                    finish = Some((true, format!("ok: {name} is in front of the player")));
                    spawned_now = (veh, slot);
                }
            }
        }
    }
    if spawned_now.0 != 0 {
        st.spawned = spawned_now;
    }
    if let Some((good, text)) = finish {
        let line = job.line.clone();
        st.spawn = None;
        say(!good, &format!("VICTORY_MOD: spawn {line:?} (frame {frame_no}): {text}"));
        set_last_result(st, text);
    }
}

/// Repairs the player's vehicle (VIper `repair_vehicle`): the game's own fix, every part to full, clean.
fn repair(st: &mut State, base: usize, ped: usize) -> Result<String, String> {
    st.need(base, &[sites::PED_COMPONENT_51, sites::VEHICLE_FROM_ID, sites::VEHICLE_REQUEST_FIX, sites::VEHICLE_PART_HEALTH, sites::VEHICLE_PART_RESTORE, sites::ENTITY_SHADER_PARAMS, sites::SHADER_SET_DIRT])?;
    // SAFETY: the live player, on its thread; the sites checked above.
    let veh = unsafe { ped_vehicle(base, ped) };
    if veh == 0 {
        return Err("the player is not in a vehicle".into());
    }
    // SAFETY: the player's live vehicle; the checked functions with VIper's arguments; part entries below their count.
    unsafe {
        let fix: extern "C" fn(usize) = std::mem::transmute(base + sites::VEHICLE_REQUEST_FIX.rva as usize);
        let parts_of: extern "C" fn(u32) -> usize = std::mem::transmute(base + sites::VEHICLE_PART_HEALTH.rva as usize);
        let restore: extern "C" fn(usize) = std::mem::transmute(base + sites::VEHICLE_PART_RESTORE.rva as usize);
        let shader: extern "C" fn(usize) -> usize = std::mem::transmute(base + sites::ENTITY_SHADER_PARAMS.rva as usize);
        let dirt: extern "C" fn(usize, f32, u8) = std::mem::transmute(base + sites::SHADER_SET_DIRT.rva as usize);
        fix(veh);
        let mut n = 0;
        let parts = parts_of(entity_id(veh));
        if parts != 0 {
            let count = ((parts + VEH_PARTS_COUNT) as *const u32).read_unaligned() as usize;
            let list = ((parts + VEH_PARTS) as *const usize).read_unaligned();
            if list != 0 {
                for i in 0..count.min(256) {
                    restore(list + VEH_PART_SIZE * i);
                    n += 1;
                }
            }
        }
        let drawable = ((veh + ENTITY_DRAWABLE) as *const usize).read_unaligned();
        if drawable != 0 {
            let p = shader(drawable);
            if p != 0 {
                dirt(p, 0.0, 0);
            }
        }
        Ok(format!("the game's repair asked for (applied next frame), {n} parts restored, cleaned"))
    }
}

/// Writes the game's vehicle models (hash, make, name) beside the status file.
fn vehicle_list(st: &mut State, base: usize) -> Result<String, String> {
    let models = vehicle_models(st, base)?;
    st.models = Some(models.clone());
    let path = st.status.as_ref().map(|p| {
        let mut s = p.as_os_str().to_owned();
        s.push(".vehicles.txt");
        PathBuf::from(s)
    });
    let Some(path) = path else { return Err("no VICTORY_MOD_STATUS: nowhere to write the list".into()) };
    let mut text = String::from("# index hash name make (spawn vehicle <name without spaces>, <hash> or #<index>)\n");
    for (i, m) in models.iter().enumerate() {
        text.push_str(&format!("#{i}\t{:#018x}\t{}\t{}\n", m.hash, m.name, m.make));
    }
    modcmd::write_atomic(&path, &text).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(format!("{} vehicle models written to {}", models.len(), path.display()))
}

// ---- script commands (`native`) -------------------------------------------------------

/// Script commands checked in our executable: (name, command hash, handler RVA,
/// signature), from VIper's verified command table (viper-layout game.rs
/// `command!`). A call by name also decodes the handler and requires its
/// self-stored hash to be this one.
pub const NATIVES: &[(&str, u64, u32, &str)] = &[
    ("PLAYER_PED_ID", 0x6471f11ff69d8a77, 0x4504480, "() -> Handle"),
    ("GET_ENTITY_COORDS", 0x6c660bb25792a0f9, 0x3c09d00, "(Handle entity, bool) -> Vec3"),
    ("GET_ENTITY_HEALTH", 0x158c3f087dc97878, 0x3c09f50, "(Handle entity) -> i32"),
    ("SET_ENTITY_HEALTH", 0x742054ea53d89812, 0x3c13be0, "(Handle entity, i32 health, Handle source, bool, bool as_damage)"),
    ("GET_PED_MAX_HEALTH", 0x17cdb10cb8ff5be9, 0x4bcd2c0, "(Handle ped) -> i32"),
    ("SET_PED_MAX_HEALTH", 0xb0cbe62c0a09c873, 0x4bd9830, "(Handle ped, i32 max)"),
    ("RESURRECT_PED", 0x74639fca8e43717c, 0x4bd5550, "(Handle ped)"),
    ("SET_PED_CAN_RAGDOLL", 0xec4b37be93c4b292, 0x4bd7ef0, "(Handle ped, bool can)"),
    ("GET_CLOCK_HOURS", 0x3d4e6e6637d30df7, 0x3cd0e10, "() -> i32"),
    ("GET_CLOCK_MINUTES", 0x47dc2440b1e9ddce, 0x3cd0e60, "() -> i32"),
    ("SET_CLOCK_TIME", 0x4d21127c5ad3a60, 0x3cd69a0, "(i32 hour, i32 minute, i32 second, bool)"),
    ("SET_CLOCK_DATE", 0xa9d6be1dfd52cf6d, 0x3cd68f0, "(i32 day, i32 month, i32 year)"),
    ("ADD_TO_CLOCK_TIME", 0x88fe2cc95483dae3, 0x3ccd6d0, "(i32 hours, i32 minutes, i32 seconds)"),
    ("SET_MILLISECONDS_PER_GAME_MINUTE", 0xb7253097fb1124c7, 0x3cd7780, "(i32 ms)"),
    ("RESET_MILLISECONDS_PER_GAME_MINUTE", 0xd20e6baf34f4cf18, 0x3cd62d0, "()"),
    ("SET_WEATHER_TYPE", 0x1f6a4a5a6e8b0e5, 0x3e920d0, "(u32 weather, bool now, bool hold, bool blend, f32 blend_seconds, bool, bool)"),
    ("SET_WEATHER_TYPE_OVERTIME", 0xb271e5bc174bba1f, 0x3e92070, "(u32 weather, f32 seconds, f32)"),
    ("SET_RAIN_LEVEL", 0xa0a82dd5d52bbf6e, 0x3e92150, "(f32)"),
    ("CLEAR_RAIN_LEVEL", 0xa117af47cf2c2e11, 0x3e834b0, "()"),
    ("FORCE_LIGHTNING_FLASH", 0xaef05f7fc9624a4d, 0x3e84720, "()"),
    ("ADD_EXPLOSION", 0xc5985661ac45da60, 0x3c048a0, "(Vec3 at, i32 kind, i32, f32 damage_scale, bool audible, bool invisible, f32 camera_shake)"),
    ("SET_RANDOM_VEHICLE_DENSITY", 0xfa3ff61b2cbda50c, 0x4a8e170, "(f32)"),
    ("SET_PARKED_VEHICLE_DENSITY", 0xe0650b81a99ac517, 0x4a90870, "(f32)"),
    ("RESET_RANDOM_VEHICLE_DENSITY", 0x3210eacb106acab4, 0x4a7f920, "()"),
    ("RESET_PARKED_VEHICLE_DENSITY", 0xeddcdddd92a5c845, 0x4a7fb90, "()"),
    ("GET_PLAYER_VEHICLE", 0x4696a1e656b0c1f5, 0x4a83e40, "() -> Handle"),
    ("SET_VEHICLE_ENGINE_ON", 0x3465d18749b305eb, 0x4a93160, "(Handle vehicle, bool on, bool instantly, bool stay_off)"),
    ("SET_VEHICLE_DIRT_LEVEL", 0xab0eae71bc5183fc, 0x4a95500, "(Handle vehicle, f32 dirt)"),
    ("MODIFY_VEHICLE_TOP_SPEED", 0x8b353295ca526c73, 0x4a8be20, "(Handle vehicle, f32 percent)"),
    ("SET_TIMECYCLE_MODIFIER", 0xdf6b189febc2f881, 0x3c16530, "(u32 nameHash)"),
    ("CLEAR_TIMECYCLE_MODIFIER", 0x29cb6e04f1c0b50d, 0x3c066f0, "()"),
    ("GET_GAME_TIMER", 0x582c0998b0981713, 0x3e85810, "() -> i32"),
    ("GET_FRAME_TIME", 0xeb31c3c7a18dfdd6, 0x3e85740, "() -> f32"),
    ("GET_GAMEPLAY_CAM_COORD", 0xf467a691a6acdfdf, 0x3cd1e40, "() -> Vec3"),
    ("GET_WATER_HEIGHT", 0x5d9d3fc0230c96f5, 0x4a87930, "(Vec3 at, f32* height) -> bool"),
    ("IS_MODEL_VALID", 0xf848f103c6d37271, 0x445f9b0, "(u64 model) -> bool"),
    ("IS_MODEL_A_VEHICLE", 0xd21824f5cab09d98, 0x4460540, "(u64 model) -> bool"),
    ("ACTIVATE_PHYSICS", 0xf1df8cf8efcee52, 0x44fbb70, "(Handle e)"),
    ("SET_ENTITY_ALPHA", 0xe2618458f20896d4, 0x3c12e30, "(Handle e, i32 alpha 0..=255, bool skin)"),
    ("RESET_ENTITY_ALPHA", 0xb88626548bb4610b, 0x3c11d20, "(Handle e)"),
];

/// The title's code section and its exception table (VIper CODE_START / CODE_END, EXCEPTION_TABLE).
const CODE_START: u32 = 0x1000;
const CODE_END: u32 = 0x05f4_0966;
const PDATA_ENTRIES: u32 = 0x0036_cbac / 12;
/// Most argument and result slots (VIper `command::MAX_ARGS`, `MAX_RESULT`).
const MOST_RESULTS: usize = 8;

/// The call context every handler takes in rcx (VIper `command::Ctx`, = modpapi's ViNativeContext).
#[repr(C)]
struct Ctx {
    args: *mut u64,
    result: *mut u64,
    arg_count: u32,
    result_count: u32,
}

#[repr(C, align(16))]
struct Slots<const N: usize>([u64; N]);

/// Code bytes inside the title's code section, or `None`.
fn code(base: usize, rva: u32, len: usize) -> Option<&'static [u8]> {
    let end = (rva as usize).checked_add(len)?;
    // SAFETY: inside the mapped title's code section, readable for its whole life.
    (rva >= CODE_START && end <= CODE_END as usize).then(|| unsafe { std::slice::from_raw_parts((base + rva as usize) as *const u8, len) })
}

/// If the function at `rva` is a script-command handler, its command hash:
/// it starts by testing its context's argument count (`cmp dword [rcx+0x10], 0`
/// then `js`, possibly after one 3-byte instruction), and the branch target
/// stores the hash with a `movabs` (VIper `command::decode_handler`). Reads
/// code only; never calls it.
fn decode_handler(base: usize, rva: u32) -> Option<u64> {
    let b = code(base, rva, 64 + 4 + 3 + 6)?;
    let i = b[..64 + 4].windows(4).position(|w| w == [0x83, 0x79, 0x10, 0x00])?;
    let j = i + 4;
    let mut target = None;
    for k in [j, j + 3] {
        if b[k] == 0x78 {
            target = Some(i64::from(rva) + k as i64 + 2 + i64::from(b[k + 1] as i8));
            break;
        }
        if b[k] == 0x0f && b[k + 1] == 0x88 {
            let d = i32::from_le_bytes([b[k + 2], b[k + 3], b[k + 4], b[k + 5]]);
            target = Some(i64::from(rva) + k as i64 + 6 + i64::from(d));
            break;
        }
    }
    let t = u32::try_from(target?).ok()?;
    let c = code(base, t, 40 + 10)?;
    (0..40).find_map(|m| (matches!(c[m], 0x48 | 0x49) && (0xb8..=0xbf).contains(&c[m + 1])).then(|| u64::from_le_bytes(c[m + 2..m + 10].try_into().unwrap())))
}

/// Every handler in the exception table's functions: (hash, rva) sorted by
/// hash, a hash found at two addresses dropped (VIper `index_handlers`).
fn index_handlers(st: &mut State, base: usize) -> Result<&Vec<(u64, u32)>, String> {
    if st.handlers.is_none() {
        st.need(base, &[sites::EXCEPTION_TABLE])?;
        let t0 = Instant::now();
        let mut out = Vec::new();
        for n in 0..PDATA_ENTRIES {
            // SAFETY: the exception table (its first bytes checked), mapped readable; 12-byte entries below its count.
            let begin = unsafe { ((base + (sites::EXCEPTION_TABLE.rva + 12 * n) as usize) as *const u32).read_unaligned() };
            if (CODE_START..CODE_END).contains(&begin)
                && let Some(h) = decode_handler(base, begin)
            {
                out.push((h, begin));
            }
        }
        out.sort_unstable();
        let mut dup = std::collections::HashSet::new();
        for w in out.windows(2) {
            if w[0].0 == w[1].0 && w[0].1 != w[1].1 {
                dup.insert(w[0].0);
            }
        }
        out.dedup();
        out.retain(|e| !dup.contains(&e.0));
        say(false, &format!("VICTORY_MOD: {} script-command handlers indexed in {} ms", out.len(), t0.elapsed().as_millis()));
        st.handlers = Some(out);
    }
    Ok(st.handlers.as_ref().unwrap())
}

/// The handler a `native` names, with its decoded hash.
fn resolve(st: &mut State, base: usize, target: &modcmd::Target) -> Result<(u32, u64, String), String> {
    match target {
        modcmd::Target::Rva(rva) => {
            let h = decode_handler(base, *rva).ok_or_else(|| format!("+{rva:#x} is not a script-command handler (its start does not decode as one): refused"))?;
            let name = NATIVES.iter().find(|n| n.1 == h).map_or(String::new(), |n| n.0.to_owned());
            Ok((*rva, h, name))
        }
        modcmd::Target::Hash(h) => {
            let idx = index_handlers(st, base)?;
            let i = idx.binary_search_by_key(h, |e| e.0).map_err(|_| format!("no handler stores the hash {h:#018x}"))?;
            let rva = idx[i].1;
            let name = NATIVES.iter().find(|n| n.1 == *h).map_or(String::new(), |n| n.0.to_owned());
            Ok((rva, *h, name))
        }
        modcmd::Target::Name(n) => {
            let &(name, hash, rva, _) = NATIVES.iter().find(|e| e.0 == n).ok_or_else(|| format!("{n} is not in the table of checked commands (use its address or hash:)"))?;
            match decode_handler(base, rva) {
                Some(h) if h == hash => Ok((rva, hash, name.to_owned())),
                _ => Err(format!("{name} (+{rva:#x}) does not decode to its hash {hash:#018x} in this executable: refused")),
            }
        }
    }
}

/// The player's (or its vehicle's) 16-byte script handle.
///
/// # Safety
/// `ent` a live entity, on its thread.
unsafe fn script_handle(base: usize, ent: usize) -> [u64; 2] {
    let mut h = [0u64; 2];
    // SAFETY: EntityToScriptHandle (checked by the caller), mode 2 as the natives use.
    unsafe {
        let f: extern "C" fn(usize, *mut [u64; 2], i32) = std::mem::transmute(base + sites::ENTITY_TO_SCRIPT_HANDLE.rva as usize);
        f(ent, &mut h, 2);
    }
    h
}

/// Calls a script-command handler with typed arguments in the context it
/// takes: 8-byte argument slots (ints and floats in the low bytes, vectors
/// and handles two slots), a zeroed 8-slot result buffer, the argument and
/// result counts. No script is made current: commands that act for a running
/// script (script-owned entities, script events) do nothing or are unsafe.
fn native(st: &mut State, base: usize, ped: usize, target: &modcmd::Target, args: &[modcmd::Arg], results: u32) -> Result<String, String> {
    use modcmd::Arg;
    let (rva, hash, name) = resolve(st, base, target)?;
    let handles = args.iter().any(|a| matches!(a, Arg::Player | Arg::Vehicle));
    if handles {
        st.need(base, &[sites::ENTITY_TO_SCRIPT_HANDLE, sites::PED_COMPONENT_51, sites::VEHICLE_FROM_ID])?;
    }
    let mut slots = Slots([0u64; modcmd::MOST_SLOTS]);
    let mut texts = Vec::new();
    let mut n = 0;
    for a in args {
        let put: Vec<u64> = match a {
            Arg::Int(v) => vec![u64::from(*v as u32)],
            Arg::U64(v) => vec![*v],
            Arg::Float(v) => vec![u64::from(v.to_bits())],
            Arg::Bool(v) => vec![u64::from(*v)],
            Arg::Str(s) => {
                let c = std::ffi::CString::new(s.as_bytes()).map_err(|_| "a string argument holds a NUL")?;
                let p = c.as_ptr() as u64;
                texts.push(c);
                vec![p]
            }
            Arg::Vec3([x, y, z]) => vec![u64::from(x.to_bits()) | u64::from(y.to_bits()) << 32, u64::from(z.to_bits())],
            // SAFETY: the live player, on its thread; the sites checked above.
            Arg::Player => unsafe { script_handle(base, ped) }.to_vec(),
            Arg::Vehicle => {
                // SAFETY: as above.
                let v = unsafe { ped_vehicle(base, ped) };
                // SAFETY: the player's live vehicle.
                if v == 0 { vec![0, 0] } else { unsafe { script_handle(base, v) }.to_vec() }
            }
            Arg::NoEntity => vec![0, 0],
        };
        slots.0[n..n + put.len()].copy_from_slice(&put);
        n += put.len();
    }
    let mut out = Slots([0u64; MOST_RESULTS]);
    let mut ctx = Ctx { args: slots.0.as_mut_ptr(), result: out.0.as_mut_ptr(), arg_count: n as u32, result_count: results.min(MOST_RESULTS as u32) };
    // SAFETY: a script-command handler of this executable (its start decoded as one, with this hash), called as the
    // game's script VM calls it, on the frame hook's thread; the arguments are what the user typed (their meaning is
    // theirs to get right; strings stay alive until it returns).
    unsafe {
        let f: extern "C" fn(*mut Ctx) = std::mem::transmute(base + rva as usize);
        f(&mut ctx);
    }
    drop(texts);
    let r = out.0;
    let lo = r[0] as u32;
    let text = format!(
        "{} +{rva:#x} hash {hash:#018x}: slots {:#x} {:#x} {:#x}; slot 0 as i32 {}, as f32 {}; as a vec3 (x and y in slot 0, z in slot 1) ({}, {}, {})",
        if name.is_empty() { "handler" } else { &name },
        r[0],
        r[1],
        r[2],
        lo as i32,
        f32::from_bits(lo),
        f32::from_bits(lo),
        f32::from_bits((r[0] >> 32) as u32),
        f32::from_bits(r[1] as u32)
    );
    st.native_result = Some(text.clone());
    Ok(text)
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
    step_spawn(st, base, ped, frame_no);
    step_wanted(st, base, ped, frame_no);
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
                Some(_) => {
                    st.wanted_check = Some((n, Instant::now(), line.to_owned()));
                    Ok(format!("wanted level {n} asked; the game shows it a moment later (watch wanted)"))
                }
                None => Err("the player has no wanted state".into()),
            }
        }
        Command::NeverWanted(on) => {
            // SAFETY: the live player, on its thread (sites checked inside).
            unsafe { set_never_wanted(st, base, ped, on) }?;
            st.held.never_wanted = on;
            st.never_again = 0;
            Ok(if on { "never wanted: cleared, highest level 0, crimes ignored (kept so once a second)" } else { "wanted again possible: highest level 6, crimes count" }.into())
        }
        Command::SpawnVehicle(m) => start_spawn(st, base, line, &m),
        Command::Repair => repair(st, base, ped),
        Command::VehicleList => vehicle_list(st, base),
        Command::Enter => {
            st.need(base, &[sites::PED_INTO_VEHICLE, sites::VEHICLE_FROM_ID, sites::PED_COMPONENT_51])?;
            st.pin(base, sites::VEHICLE_POOL)?;
            let (veh, slot) = st.spawned;
            if !vehicle_alive(base, veh, slot) {
                return Err("no vehicle spawned by the mod is there (spawn vehicle first)".into());
            }
            // SAFETY: the live player and the live vehicle we made (checked above).
            let (me, car) = unsafe { (entity_place(ped).0, entity_place(veh).0) };
            let d = ((me[0] - car[0]).powi(2) + (me[1] - car[1]).powi(2) + (me[2] - car[2]).powi(2)).sqrt();
            if d > 50.0 {
                return Err(format!("the vehicle spawned is {d:.0} m away (at most 50 m)"));
            }
            // SAFETY: SetPedIntoVehicle_Impl (checked) as the put-ped-in-vehicle command calls it: driver's seat 0.
            unsafe {
                let f: extern "C" fn(usize, usize, u8, u32) = std::mem::transmute(base + sites::PED_INTO_VEHICLE.rva as usize);
                f(ped, veh, 0, 0);
            }
            // SAFETY: as above.
            let now = unsafe { ped_vehicle(base, ped) };
            Ok(if now == veh { "the player is in the driver's seat".into() } else { "asked; the in-vehicle state shows it a frame or two later (see in_vehicle)".into() })
        }
        Command::Native { target, args, results } => native(st, base, ped, &target, &args, results),
        Command::Status => unreachable!(),
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
    let (max, flags, level) = unsafe {
        let comp: extern "C" fn(u32) -> usize = std::mem::transmute(base + sites::WANTED_COMPONENT.rva as usize);
        let w = comp(entity_id(ped));
        if w == 0 {
            return Err("the player has no wanted state".into());
        }
        ((g as *const i32).read_volatile(), ((w + WANTED_FLAGS) as *const u32).read_unaligned(), ((w + WANTED_CURRENT_LEVEL) as *const i32).read_unaligned())
    };
    if max != 0 || flags & WANTED_IGNORE_CRIMES == 0 || level > 0 {
        // SAFETY: the caller's.
        unsafe { set_never_wanted(st, base, ped, true) }?;
        st.never_again += 1;
        if st.never_again == 1 || st.never_again.is_power_of_two() {
            say(false, &format!(
                "VICTORY_MOD: never wanted put back ({} times; the game had: highest level {max}, crimes-ignored flag {}, level {level}; frame {frame_no})",
                st.never_again,
                u8::from(flags & WANTED_IGNORE_CRIMES != 0)
            ));
        }
    }
    Ok(())
}

/// After `wanted n`: the level the game shows over the next 3 s (it applies a set level a moment later).
fn step_wanted(st: &mut State, base: usize, ped: usize, frame_no: u64) {
    let Some((n, since, line)) = st.wanted_check.clone() else { return };
    if ped == 0 || st.need(base, &[sites::WANTED_COMPONENT, sites::WANTED_LEVEL_READ]).is_err() {
        st.wanted_check = None;
        return;
    }
    // SAFETY: the live player; both sites checked.
    let l = unsafe { wanted_level(base, ped) }.unwrap_or(-1);
    let ms = since.elapsed().as_millis();
    let text = if l == n as i32 {
        format!("ok: wanted level {l} (shown {ms} ms after asking)")
    } else if since.elapsed() >= Duration::from_secs(3) {
        format!("ok, but the game holds the wanted level at {l} 3 s after {n} was asked (a mission or its highest allowed level may hold it)")
    } else {
        return;
    };
    st.wanted_check = None;
    say(false, &format!("VICTORY_MOD: {line:?} (frame {frame_no}): {text}"));
    if st.last.as_ref().is_some_and(|x| x.line == line) {
        set_last_result(st, text);
    }
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
    match &st.spawn {
        Some(j) if j.made.is_none() => s.put("spawn", format!("loading {} {} for {} ms", j.model.make, j.model.name, j.since.elapsed().as_millis())),
        Some(j) => s.put("spawn", format!("made {} {}, settling", j.model.make, j.model.name)),
        None => s.put("spawn", "idle"),
    };
    if let Some(l) = &st.last {
        s.put("last_command", &l.line).put("last_source", l.source).put("last_frame", l.frame).put("last_result", &l.result);
    }
    if let Some(r) = &st.native_result {
        s.put("native_result", r);
    }
    s.put("commands_ok", st.ok).put("commands_refused", st.refused);
    if let Err(e) = modcmd::write_atomic(&path, s.text())
        && !std::mem::replace(&mut st.status_said, true)
    {
        say(true, &format!("VICTORY_MOD: the status file {} could not be written ({e}); tried again twice a second", path.display()));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_pin_lands_on_its_target_in_the_recorded_bytes() {
        let pins = [
            sites::LOCAL_PLAYER_PED,
            sites::WAYPOINT_MGR,
            sites::GAME_CLOCK,
            sites::CLOCK_PAUSED_BY_SCRIPT,
            sites::WEATHER_COUNT,
            sites::WEATHER_TYPES,
            sites::WEATHER_PERSIST,
            sites::WANTED_MAX_LEVEL_GLOBAL,
            sites::VEHICLE_MODEL_POOL,
            sites::MODEL_REQUEST_DEFAULTS,
            sites::WORLD_ENTITY_MANAGER,
            sites::WORLD_DEFAULT_LOCATION,
            sites::PHYSICS_WORLD,
            sites::VEHICLE_COUNTER,
            sites::VEHICLE_POOL,
        ];
        for p in pins {
            let d = p.disp as usize;
            let b: [u8; 4] = p.site.bytes[d..d + 4].try_into().unwrap();
            let target = i64::from(p.site.rva + p.end) + i64::from(i32::from_le_bytes(b));
            assert_eq!(target, i64::from(p.target), "{}", p.site.name);
            assert!(p.end as usize >= d + 4);
        }
    }

    #[test]
    fn checks_refuse_other_code() {
        // A fake image: one site's bytes at its RVA, the rest zero.
        let s = sites::CLOCK_IS_PAUSED;
        let mut image = vec![0u8; s.rva as usize + 64];
        image[s.rva as usize..s.rva as usize + s.bytes.len()].copy_from_slice(s.bytes);
        let base = image.as_ptr() as usize;
        let mut st = State {
            tail: None,
            status: None,
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
            spawn: None,
            models: None,
            spawned: (0, 0),
            wanted_check: None,
            never_again: 0,
            handlers: None,
            native_result: None,
        };
        assert!(st.need(base, &[s]).is_ok());
        assert!(st.pin(base, sites::CLOCK_PAUSED_BY_SCRIPT).is_ok_and(|a| a == base + 0x092e_0031));
        image[s.rva as usize + 3] ^= 1;
        let mut fresh = HashMap::new();
        std::mem::swap(&mut st.checked, &mut fresh);
        let e = st.need(base, &[s]).unwrap_err();
        assert!(e.contains("CLOCK_IS_PAUSED") && e.contains("refused"), "{e}");
    }

    #[test]
    fn handlers_decode_from_their_first_bytes() {
        // A handler at 0x2000: mov eax, 1 (3 bytes, skipped), cmp dword [rcx+0x10], 0; js +0x10; at the target
        // a movabs rax, <hash>.
        let mut image = vec![0xccu8; 0x3000];
        let f = 0x2000;
        image[f..f + 3].copy_from_slice(&[0x31, 0xc0, 0x90]);
        image[f + 3..f + 7].copy_from_slice(&[0x83, 0x79, 0x10, 0x00]);
        image[f + 7..f + 9].copy_from_slice(&[0x78, 0x10]);
        let t = f + 9 + 0x10;
        image[t] = 0x48;
        image[t + 1] = 0xb8;
        image[t + 2..t + 10].copy_from_slice(&0x6471_f11f_f69d_8a77u64.to_le_bytes());
        let base = image.as_ptr() as usize;
        assert_eq!(decode_handler(base, f as u32), Some(0x6471_f11f_f69d_8a77));
        assert_eq!(decode_handler(base, 0x1000), None, "int3 filler is no handler");
        assert_eq!(decode_handler(base, 0x800), None, "below the code section");
        // The table's names are unique and its hashes all different.
        let mut names: Vec<_> = NATIVES.iter().map(|n| n.0).collect();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), NATIVES.len());
    }

    #[test]
    fn vehicle_names_are_found_loosely() {
        let m = |h: u64, n: &str, k: &str| VehicleModel { hash: h, name: n.into(), make: k.into() };
        let models = [m(1, "Infernus", "Pegassi"), m(2, "Banshee 900R", "Bravado"), m(0xdead_0000_0000 | u64::from(modcmd::joaat("adder")), "Adder", "Truffade")];
        let find = |n: &str| find_model(&models, &modcmd::Model::Name(n.into())).map(|v| v.hash);
        assert_eq!(find("infernus"), Ok(1));
        assert_eq!(find("banshee900r"), Ok(2));
        assert_eq!(find("bravadobanshee900r"), Ok(2));
        assert_eq!(find("adder"), Ok(models[2].hash));
        assert!(find("unicorn").is_err());
        assert_eq!(find_model(&models, &modcmd::Model::Hash(2)).map(|v| v.name), Ok("Banshee 900R".into()));
        assert_eq!(find_model(&models, &modcmd::Model::Hash(9)).map(|v| v.hash), Ok(9), "an unlisted hash is tried as is");
    }
}
