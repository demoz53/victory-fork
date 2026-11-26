# The mod panel and the mod commands (`VICTORY_MOD=1`)

Single player only. The game side runs inside VIctory's frame hook (a detour of the game timer update) and calls the
game's own functions; no game file is changed. Everything is off unless `VICTORY_MOD=1`, and off with
`VICTORY_TITLE_PATCHES=0`.

## Starting it

1. Start the game with these switches set (beside your usual ones):

   ```
   VICTORY_MOD=1
   VICTORY_MOD_FILE=C:\Users\<you>\Documents\victory-data\mod\commands.txt
   VICTORY_MOD_STATUS=C:\Users\<you>\Documents\victory-data\mod\status.txt
   ```

   The folder must exist (the panel makes it when it first sends). For the keys-file route (tests) add
   `VICTORY_TITLE_COMMANDS=1` (a diagnostics build).

2. Start the panel (Python 3 with tkinter; nothing to install):

   ```
   python tools\modpanel\modpanel.py --mod-file <the same commands.txt> --status-file <the same status.txt> --top
   ```

   Without `--mod-file` / `--status-file` it uses `VICTORY_MOD_FILE` / `VICTORY_MOD_STATUS` from its environment, else
   `Documents\victory-data\mod\commands.txt` and `status.txt`. `--top` keeps it above the game's window. `--places
   <file>` picks another places file (default `places.json` beside the panel, made on its first start).

The panel only appends lines to the mod file and reads the status file. It says "The game is not running" when the
status file is missing or older than 3 s, and then sends nothing (the game only runs lines added while it runs).

## Commands

One per line, in the mod file or (with `VICTORY_TITLE_COMMANDS=1`) as lines of a keys-file route. Numbers are metres
and radians; x east, y north, z up; heading 0 = north, counter-clockwise (pi/2 faces west). `#` starts a comment.
Every command answers in the status file (`last_command`, `last_result`) and in the log
(`VICTORY_MOD: <modfile|keys> "<line>" (frame N): ok: ...` or `refused: <why>`).

| Command | Example | What |
|---|---|---|
| `teleport x y [z [heading]]` (or `tp`) | `teleport -2355.6 -5526.3 7.2 2.17` | Any distance. The player (or its vehicle, with whoever is in it) moves at once; if the ground there is not loaded yet it is held at the target (at z, or 10 m over the game's height map without z) until the game has loaded it, then set down: at z, or 1 m over the ground found. After 12 s without ground: at z, or 1 m over the height map, with a warning. |
| `teleport x y` | `teleport -2155 -5330` | As above; the ground is found. |
| `tp rel <forward> <up>` | `tp rel 5 0`, `tp rel 0 5` | Along the way the player faces, and up (z = now + up). |
| `tp waypoint` | `tp waypoint` | To the map waypoint, on the ground (refused without one). |
| `time <hh> <mm>` | `time 21 0` | Sets the game clock (the game's own clock setter). |
| `time freeze on\|off` | `time freeze on` | Stops or starts the clock (the flag the game's pause-clock command sets); re-applied once a second while on, since missions may start it. |
| `weather <name>` | `weather rain` | Sets the weather now and holds it (re-applied once a second if the game changes it). Names: sunny clear misty fog clouds overcast darkovercast drizzle shower rain thunder thunderstorm hurricane hail sandstorm, or any other name or `0x` hash the game knows (refused if it has none). |
| `weather cycle` | `weather cycle` | Back to the game's own weather. |
| `heal` | `heal` | Full health (gets the player up first if downed; refused if dying or dead). |
| `invincible on\|off` | `invincible on` | All 19 damage multipliers 0 (the game's damage-proofs setter); off puts back what the player had. Re-applied once a second and on a new player ped. |
| `wanted <0-6>` | `wanted 2` | Sets the wanted level (the game's maximum is 6); says if the game holds it lower. |
| `wanted never on\|off` | `wanted never on` | Clears the level, sets the highest level to 0 and makes crimes ignored; off: 6 and crimes count. Re-applied once a second (missions set their own). |
| `vehicle list` | `vehicle list` | Writes every vehicle model the game defines to `<status file>.vehicles.txt`: `#index hash name make` (names in the game's language). The panel's "List the game's vehicles" button fills its vehicle box from it. |
| `spawn vehicle <name>` | `spawn vehicle infernus` | A vehicle 5 m in front of the player, facing the same way, on the ground there. The name is the game's display name (case and spaces ignored, with or without the make), else a name whose one-at-a-time hash is the model hash's low half. The model is asked for and the vehicle made when it has loaded (over several frames, never blocking; 10 s at most), then put in the world and its physics woken. One spawn at a time. |
| `spawn vehicle 0x<hash>` / `#<n>` | `spawn vehicle #0` | By 64-bit model hash, or by line number of `vehicle list`. |
| `vehicle enter` | `vehicle enter` | The player into the driver's seat of the vehicle the mod spawned last (at most 50 m away). |
| `vehicle repair` | `vehicle repair` | The player's vehicle: the game's own full repair (applied the next frame), every part to full health, cleaned. |
| `status` | `status` | Rewrite the status file now. |

Coming in the next piece: `native`. It is already parsed and answers "not in this build yet".

## The status file

`key=value` lines, rewritten twice a second (through a temporary file and a rename, so it is never half written):
`game updated_unix_ms frame player x y z heading heading_deg in_vehicle health max_health wanted clock clock_frozen
weather freeze_held weather_held invincible never_wanted teleport spawn last_command last_source last_frame last_result commands_ok commands_refused`.

## Safety

- Every game address is used only after the bytes there are the ones recorded for this executable (1.0.2.1, PE stamp
  0x6af783d4), and every data word only through a checked instruction that reads it; otherwise the command is
  refused with the reason.
- Calls happen only on the frame hook's own thread, once a frame; nothing blocks a frame.
- Nothing but `status` runs while there is no player.
- A command runs once. A teleport still waiting for its ground is replaced by the next teleport, and dropped if the
  player or its vehicle changes.

## What is proven

| Piece | Offline (tests) | In the game |
|---|---|---|
| Mod file, status file, parser, teleports | parser, tail (partial lines, new file), atomic status write, the far teleport's steps with a fake world | not yet |
| Time, weather, heal, invincible, wanted | parser; every address's bytes and every data word's pin checked against the executable | not yet |
| Vehicles | parser, name matching, pins | not yet |

Sources: viper-community/viper e44573c6e (LGPL-2.1-or-later): addresses, signatures and recipes, all checked again
against the executable's bytes.
