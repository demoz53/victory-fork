# SWITCHES.md - every switch added since test/all14n, in test/all16

`test/all16` = `test/all15b` (3b9488230) + merges of test/lx, test/cxall, test/stream1 (holds test/fairq),
test/shirt1, test/speed1, test/compose. One build, every feature behind its switch: set several in one run line.

How this list was made: `git diff test/all14n..HEAD | grep -oE '"VICTORY_[A-Z0-9_]+"'` (added lines), then each
switch read at its definition. "Shader" = the switch changes translated shader code: the shader cache keys on it, and
a run with a value not used before on this executable needs a pipeline warm before its picture or speed means
anything. Every new executable (this one too) translates and warms once anyway.

## With nothing set, all16 differs from all15b in three ways
| Change | Came with | Back to all15b behaviour |
|---|---|---|
| `VICTORY_LOOP_EXITS` is on | test/lx | `VICTORY_LOOP_EXITS=0` (shader code changes: warm) |
| `VICTORY_FAIR_DUE` is on | test/stream1 (fairq) | `VICTORY_FAIR_DUE=0` |
| `VICTORY_RELOOK_MS` is 20000 (all15b: a fixed 4000, cap 256 frames) | test/stream1 (fairq) | `VICTORY_RELOOK_MS=4000` |

Exactly all15b: `VICTORY_LOOP_EXITS=0 VICTORY_FAIR_DUE=0 VICTORY_RELOOK_MS=4000`.

## New in all16 (merged on top of all15b)
| Switch | Values | Default | What it does | From | Shader |
|---|---|---|---|---|---|
| `VICTORY_LOOP_EXITS` | `0` off; unset or anything else on | ON (all15b: on only with `=1`) | wave64 compute programs test the loop iteration limit at the loop's own exits, not in the latch (removes the dark tiles on water) | test/lx (rule from all15b ec51e9045, darkworldhm 08fdf0b47) | YES |
| `VICTORY_REFRESH_RANGES` | `1` | off | a submission's refresh ranges of a host copy are joined when within 64 KiB of each other (coarser than our default exact union; reviewer: unlikely to gain) | test/cxall | no |
| `VICTORY_BACK_BLOCKS` | `1` | off | copy-back writes only the 1 KiB blocks the merge changed, not the whole span (fewer write-watch echoes) | test/cxall | no |
| `VICTORY_DECLARE_ONCE` | `1` | off | a list records the GPU-write declaration of one host copy and range once between refreshes, not at every dispatch | test/cxall | no |
| `VICTORY_NAMED_WRITES` | `1` | off | raw-address stores copy back only the buffers the dispatch names; RISK: a store through a pointer loaded from memory into an unnamed copy is not copied back (test alone first) | test/cxall | no |
| `VICTORY_GUARD_REST` | `0`/unset off; `1` = 3000 us; `n` > 1 = n microseconds | off | a write-faulted block stays open (told as written) for that long before it is made read-only again | test/cxall | no |
| `VICTORY_OPEN_ONE_VIEW` | `1` | off | a write fault opens only the faulting view; its aliases stay protected and are marked written (use with GUARD_REST as a pair) | test/cxall | no |
| `VICTORY_FAIR_DUE` | `0` off; unset or anything else on | ON | due texture slots are served in turns, one looked at before and one never looked at | test/stream1 (fairq; fxceless PR #1246) | no |
| `VICTORY_RELOOK_MS` | milliseconds | 20000 (all15b: 4000) | longest wait before a slot is looked at again, clamped to 16-1024 frames (`4000`: 16-256 as before) | test/stream1 (fairq) | no |
| `VICTORY_TEXTURE_KEEP` | `1` | off | host copies of title-filled textures are kept while the card has room (not dropped after 5 s idle); cs-keep1: 5x fewer textures made again | test/stream1 | no |
| `VICTORY_TEXTURE_KEEP_SECS` | 5-3600 | 600 | how long TEXTURE_KEEP keeps an idle host copy; only read with `VICTORY_TEXTURE_KEEP=1` | test/stream1 | no |
| `VICTORY_DSTORAGE_WORKERS` | 1-64 | logical processors clamped to 2-16 (16 here) | number of DirectStorage worker threads | test/stream1 | no |
| `VICTORY_DSTORAGE_PRIORITY` | `normal` | below normal | DirectStorage workers run at normal priority | test/stream1 | no |
| `VICTORY_DSTORAGE_HANDLES` | `1` | off | each DirectStorage worker reads a file through a handle of its own (side by side, not in turns) | test/stream1 | no |
| `VICTORY_STREAM_PRIORITY` | `1` | off | the title thread that maps pages and submits requests is lifted from below normal to above normal | test/stream1 | no |
| `VICTORY_COPY_KEEP` | `1` | off | the first host copy into a title-filled texture asked to be uploaded in the last 4 frames is kept in a second texture and copied back 6 and 30 frames later (floral shirt confetti) | test/shirt1 | no |
| `VICTORY_TEXTURE_TRACE` | `n` = most lines of each kind | 0 | log only: names such textures, late uploads and restores | test/shirt1 | no |
| `VICTORY_FAST_CORES` | `1`-`4` (stages: 1 our landing/settling threads, 2 + wide helpers, 3 + title threads that submit, 4 + texture makers) | 0 | asks the scheduler to keep those threads on the fast cores | test/speed1 | no |
| `VICTORY_PROCESS_PRIORITY` | `above` or `1` | as started | the process runs at above-normal priority | test/speed1 | no |
| `VICTORY_CARD_AWAKE` | `1`; `turns` (20 s on, 20 s off) | off | a small compute pass keeps the card busy between submissions; MEASURED A LOSS on this RTX 4060 Ti | test/speed1 | no |
| `VICTORY_CARD_AWAKE_STEPS` | 1000-4000000 | 400000 | length of that pass; only read with CARD_AWAKE | test/speed1 | no |
| `VICTORY_FRAME_PARTS` | `1` | off | log only, diagnostics builds: list-call time by the line of list.rs that made the call | test/speed1 | no |
| `VICTORY_COMPOSE_TRACE` | any value but `0` or empty | off | log only: plane sizes, rectangles and swap chain of each present when they change | test/compose | no |

## Came with all15b (new since all14n, already in the base)
| Switch | Values | Default | What it does | From | Shader |
|---|---|---|---|---|---|
| `VICTORY_NORM_EXACT` | any value but `0` on | OFF (all14n: on) | exact UNORM/SNORM conversion in buffer format loads (suspected of breaking a shirt; re-test once COPY_KEEP settles the shirt) | all15 198f964f8 | YES |
| `VICTORY_HEAP_BELOW_ZERO` | any value but `0` on | OFF (all14c: on) | a descriptor read from below the heap start is taken as slot 0 (changes about 3,461 programs) | test/all14c, off in all15 | YES |
| `VICTORY_LOCAL_LDS_SYNC` | `0` off | ON | programs with wave-synchronous LDS access are kept out of the per-half-wave form | test/all14w | YES |
| `VICTORY_NAN_GUARD` | `1` pixel colour exports; `stores` typed stores too | off | clamps NaN/infinity in those outputs | test/all13v | YES |
| `VICTORY_INT_VIEWS_AS_FLOAT` | `0` off | ON | typed buffer views of 32-bit integer formats are made in the float format of the same width | test/all13v | no |
| `VICTORY_STALE_HEAD_SIZE` | `0` off | ON | a placed texture's own first bytes are the smaller of HEAD and its size (small tint palettes no longer make neighbours stale) | test/all13v | no |
| `VICTORY_TEXTURE_BACK_STATE` | `0` off | ON | after a texture copy-back the texture returns to the state the title's commands left it in | test/all14t | no |
| `VICTORY_TEXTURE_BACK_VIEWS` | `1` | off | only the mips and slices the list's unordered access views expose are copied back | test/all14t | no |
| `VICTORY_PARTIAL_PROMPT` | `1` | off | a partly mapped texture is read again as soon as a larger mip is all there, not every 30 frames (was in the harmful cs-all1 set: not yet separated) | test/all14t | no |
| `VICTORY_RESIDENT_MIPS` | `1` | off | a partly mapped texture has only its fully mapped mips detiled; the larger keep what they held | test/all14t | no |
| `VICTORY_HOST_LOAD_UNLOCKED` | `0` off | ON | a host DLL is loaded with the module cache unlocked (loader-lock deadlock fix) | test/all14c | no |
| `VICTORY_PIPELINE_STACK_MB` | MiB; `0` = 2 MiB as before | 64 | stack of each pipeline worker (the driver's compiler overflowed 2 MiB) | test/all14c | no |
| `VICTORY_PIPELINE_STUCK_S` | seconds; `0` never | 15 | how long a pipeline may be with a worker before it counts as stuck | test/all14c | no |
| `VICTORY_SCREEN_PASS_WAIT_MS` | milliseconds | 8000 (before: 20 s) | waiting allowed per title frame for passes over the screen | test/all14c | no |
| `VICTORY_UNMAP_NOTIFY_AFTER` | `1` | off | an unmapped view is reported to the GPU side after the unmap, not before | test/all14c | no |
| `VICTORY_WAIT_UNMAPPED_UNMET` | `1` | off | a wait on title memory that cannot be read does not hold (default: it holds) | test/all14c | no |
| `VICTORY_XMA_FRAME_END` | `0` off | ON | an XMA frame's decode ends at its data, not at the caller's bit count (endless-loop fix) | test/all14c | no |
| `VICTORY_XMA_RING_WRAP` | `0` off | ON | XMA read and write blocks are taken modulo the output ring | test/all14c | no |

`VICTORY_HOST_LOCK_CHILD` also appears in the diff: it is a marker a unit test sets for its own child process, not a
switch.

## Combining them
- No pair of the new switches has to stay apart: each merged cleanly and they act on different paths. Nothing refuses
  at startup.
- Pairs meant to go together: `VICTORY_GUARD_REST=1 VICTORY_OPEN_ONE_VIEW=1`; `VICTORY_TEXTURE_KEEP_SECS` and
  `VICTORY_CARD_AWAKE_STEPS` do nothing without their main switch.
- Speed group judged safe together: `VICTORY_DECLARE_ONCE=1 VICTORY_BACK_BLOCKS=1 VICTORY_GUARD_REST=1
  VICTORY_OPEN_ONE_VIEW=1 VICTORY_FAST_CORES=3 VICTORY_PROCESS_PRIORITY=above`. Left out on purpose: NAMED_WRITES
  (picture risk: alone first), REFRESH_RANGES (coarser than the default), CARD_AWAKE (measured loss), FRAME_PARTS
  (measurement; add it to any run for the per-call-site table).
- Streaming and texture group judged safe together: `VICTORY_TEXTURE_KEEP=1 VICTORY_COPY_KEEP=1
  VICTORY_TEXTURE_TRACE=40`. Left out on purpose: STREAM_PRIORITY, DSTORAGE_WORKERS/PRIORITY/HANDLES and
  PARTIAL_PROMPT (run cs-all1: one of them took the boat to 2,474 ms a frame; separate them one per run first).
- Shader-changing switches (LOOP_EXITS, NORM_EXACT, HEAP_BELOW_ZERO, LOCAL_LDS_SYNC, NAN_GUARD): give each value
  its own warm; do not mix a first use of one of them into a speed comparison.
