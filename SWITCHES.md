# SWITCHES.md - every switch added since test/all14n, in test/all20

`test/all20` = `test/all19` (0045f4def) + merges, in this order, of test/small3 (ed9ad620b; based on all17) and
test/locks (27c580714; based on all18). Two new default-on changes (rows 21 and 22 below): `VICTORY_HEAP_FORGET`
and `VICTORY_SDWA_SEXT` (the second changes translated shader code).

`test/all19` = `test/all18` (ddb959c0e) + merges, in this order, of test/meshclamp (9e5d01b28), test/scrub
(b9ebab139), test/room1 (f424f55f8, then again to its tip 849247e67) and test/heapw (b3a6e21c2, then again to its tip
694eb0672). Left out: test/small3 (its worktree had uncommitted changes and no pre-build stamp: its agent was still
at work). One new default-on change, `VICTORY_AMBIENT_SETTLE` (row 20 below); it acts only with `VICTORY_AMBIENT`
set, so a run with nothing set behaves as all18.

`test/all18` = `test/all17` (8cc48702e) + merges, in this order, of test/bake2 (53231fab3), test/variants
(69b26a019), test/ambient (8d5dd30df), test/speed2 (588ca3b52), test/light1 (987d937cb) and test/tables
(322159026). Left out (their agents were still at work when the merges were made): test/scrub, test/meshclamp.
The new switches are in "New in all18" below.

`test/all17` = `test/all16` (fb98a3eda) + merges of test/avfix, test/slots, test/stream2 (to 5fe884eb9), test/res1,
test/split, test/loops2 (to its tip 864343dff), five commits of test/testkit (0600708fa, 5867481c7,
d59bd65f6, c3ec9f984, 2bf01ab20), the two new commits of test/speed1 (to 4e2e1f6e1), test/clamps (d4b2d6324), one
commit of test/all13y (95d04e060) and test/pairs (97ac88160). `test/all16` = `test/all15b` (3b9488230) +
test/lx, test/cxall, test/stream1 (holds test/fairq), test/shirt1, test/speed1 (to 625bc32b6), test/compose.
One build, every feature behind its switch: set several in one run line.

How this list was made: `git diff test/all14n..HEAD | grep -oE '"VICTORY_[A-Z0-9_]+"'` (added lines), then each
switch read at its definition. "Shader" = the switch changes translated shader code: the shader cache keys on it, and
a run with a value not used before on this executable needs a pipeline warm before its picture or speed means
anything. Every new executable (this one too) translates and warms once anyway.

## DEFAULT-ON changes since test/all15b (what a run with nothing set does differently)
| # | Change | Came with | Back to the old behaviour |
|---|---|---|---|
| 1 | `VICTORY_LOOP_EXITS` is on (wave64 loop limit tested at the loop's exits) | test/lx (all16) | `VICTORY_LOOP_EXITS=0` (shader code changes: warm) |
| 2 | `VICTORY_FAIR_DUE` is on (looked-at and fresh texture slots served in turns) | test/stream1 (all16) | `VICTORY_FAIR_DUE=0` |
| 3 | `VICTORY_RELOOK_MS` is 20000 (all15b: a fixed 4000, cap 256 frames) | test/stream1 (all16) | `VICTORY_RELOOK_MS=4000` |
| 4 | `VICTORY_REMAP_REOPEN` is on (a remap notice gives write access back to the blocks it forgets) | test/avfix | `VICTORY_REMAP_REOPEN=0` |
| 5 | `VICTORY_DYNAMIC_SLOTS` unset = 606,784 dynamic descriptor slots on a capped view heap with a heap map (before: 262,144) | test/slots | `VICTORY_DYNAMIC_SLOTS=262144` |
| 6 | `VICTORY_TEXTURE_BACK_CHECK` is on (a texture copy-back lands only where its texture still lies, and only over the bytes it covers) | test/slots | `VICTORY_TEXTURE_BACK_CHECK=0` |
| 7 | `VICTORY_DYNAMIC_ACROSS` is on (the search for free dynamic slots finds a run across its starting place) | test/slots | `VICTORY_DYNAMIC_ACROSS=0` |
| 8 | `VICTORY_TABLES_EARLY` is on (with VICTORY_RELAY: a flight's dynamic slots are given back when the card has run it) | test/slots | `VICTORY_TABLES_EARLY=0` |
| 9 | `VICTORY_ALIAS_ALWAYS` is on (a view of a texture the card wrote is copied from it even if the list already uploaded the view) | test/slots | `VICTORY_ALIAS_ALWAYS=0` |
| 10 | `VICTORY_QUEUE_AGE` is on (log only: histogram of frames between a slot's write and its texture on the card) | test/stream2 | `VICTORY_QUEUE_AGE=0` |
| 11 | `VICTORY_MAPPED_EXACT` is on (a texture with only pages in its middle mapped counts as there; looked at in turns) | test/stream2 5fe884eb9 | `VICTORY_MAPPED_EXACT=0` |
| 12 | `VICTORY_PERMUTE_HALVES` is on (ds_bpermute reads across the two host waves of a console wave; before: refused) | test/loops2 664ee75e0 | `VICTORY_PERMUTE_HALVES=0` (shader code) |
| 13 | `VICTORY_SHORT_WAVE_LAST` is on (the short host wave of a group takes the last console lanes; before: arrival order) | test/loops2 a7afedf05 | `VICTORY_SHORT_WAVE_LAST=0` (shader code) |
| 14 | `VICTORY_LANE_MODULO` is on (v_readlane and v_writelane take a computed lane modulo the wave) | test/loops2 533f14c8a | `VICTORY_LANE_MODULO=0` (shader code) |
| 15 | `VICTORY_THREAD_TICKET` is on (console threads of groups of several host waves come from a counter, not from rounding: four sky ambient programs had duplicate thread numbers) | test/loops2 864343dff | `VICTORY_THREAD_TICKET=0` (shader code) |
| 16 | the translated-shader cache is keyed by the translator's fingerprint, not by the executable's hash (builds with the same translator share entries; no picture change) | test/testkit d59bd65f6 | `VICTORY_SHADER_CACHE_KEY=exe` |
| 17 | `VICTORY_BAKE_WIDE` is full (the bake's care also goes to three more ped-bake pixel shaders, the ped baker's geometry stages and the bake's shape) | test/bake2 087bca876 (all18) | `VICTORY_BAKE_WIDE=0` |
| 18 | `VICTORY_CLEAR_CHECKED` is on (a slot found unmapped is made null only while the write looked at is still the queued one) | test/bake2 53231fab3 (all18) | `VICTORY_CLEAR_CHECKED=0` |
| 19 | `VICTORY_TARGET_VARIANTS` is on (a draw into render targets of other formats than its pipeline declares gets a host pipeline made for the bound formats) | test/variants 7c52fed2b (all18) | `VICTORY_TARGET_VARIANTS=0` |
| 20 | `VICTORY_AMBIENT_SETTLE` is 250,3 (with `VICTORY_AMBIENT` only: the stand-in's mean of the light around follows by the clock, a jump taken at once; before: 0.04 a frame) | test/room1 4ef159e3f (all19) | `VICTORY_AMBIENT_SETTLE=0` |
| 21 | `VICTORY_HEAP_FORGET` is on (a dropped host view heap takes its entries out of the heap passes' caches keyed by its address) | test/small3 a939d5a72 (all20) | `VICTORY_HEAP_FORGET=0` |
| 22 | `VICTORY_SDWA_SEXT` is on (an SDWA WORD_0 destination with UNUSED_SEXT sign-extends a 16-bit result) | test/small3 c5dd98428 (all20) | `VICTORY_SDWA_SEXT=0` (shader code changes: warm) |

Everything restored (all15b behaviour, one line; all18 added BAKE_WIDE, CLEAR_CHECKED, TARGET_VARIANTS, all19 AMBIENT_SETTLE, all20 HEAP_FORGET and SDWA_SEXT):
`VICTORY_LOOP_EXITS=0 VICTORY_FAIR_DUE=0 VICTORY_RELOOK_MS=4000 VICTORY_REMAP_REOPEN=0 VICTORY_DYNAMIC_SLOTS=262144 VICTORY_TEXTURE_BACK_CHECK=0 VICTORY_DYNAMIC_ACROSS=0 VICTORY_TABLES_EARLY=0 VICTORY_ALIAS_ALWAYS=0 VICTORY_QUEUE_AGE=0 VICTORY_MAPPED_EXACT=0 VICTORY_PERMUTE_HALVES=0 VICTORY_SHORT_WAVE_LAST=0 VICTORY_LANE_MODULO=0 VICTORY_THREAD_TICKET=0 VICTORY_BAKE_WIDE=0 VICTORY_CLEAR_CHECKED=0 VICTORY_TARGET_VARIANTS=0 VICTORY_AMBIENT_SETTLE=0 VICTORY_HEAP_FORGET=0 VICTORY_SDWA_SEXT=0 VICTORY_SHADER_CACHE_KEY=exe`
(leave `VICTORY_SHADER_CACHE_KEY=exe` out to keep the shared cache: it changes no picture, only which cache
entries a build finds. Five of these change translated code - LOOP_EXITS and the four lane fixes of test/loops2,
343 compute programs by its agent's count - so a run with them restored needs a warm of its own.)

Always on, with no switch (no behaviour change, some cost): the "title memory remapped" work line every 5 s and the
look it needs at every mapping call, whether the pages are there already (test/speed1 ec85139be); the "dstorage:
workers" line's two extra fields; a trace string built at each composed present. Since all18, log and count only:
the bake line (a line per pipeline recognised as a bake, a count at most every 5 s), the clears CLEAR_CHECKED left
out (texture queue line), the first 16 of the title's pipeline syncs logged and all counted, the first 16 structured
buffer views that cannot be made raw (they read zeros on NVIDIA) warned and all counted (frame line), the formats of
the bound render targets kept at each set of targets (for TARGET_VARIANTS), and the skipped-pipeline test made once
per pipeline at its creation (a bool per pipeline, now also in release builds; `VICTORY_SKIP_PIPELINES` is still a
diagnostics-build switch).

## New in all20 (merged on top of all19)
| Switch | Values | Default | What it does | From | Shader |
|---|---|---|---|---|---|
| `VICTORY_CARD_FETCH_INFLIGHT` | `1` (+ `VICTORY_CARD_FETCH_INFLIGHT_MB`, MiB, 64) | off (only with the relay) | with the relay, the pieces a flight's compare flags are copied to PC memory inside the flight's own compare list (no fetch on the copy queue beside later flights: no torn fetches); a landing that cannot use them fetches the old way on the main queue. Works with `VICTORY_INDIRECT_WINDOW=1` (see the commit) | test/small3 ee81b4916 (fxceless 9376fd60c, PR #648 Jonirulah) | no |
| `VICTORY_CARD_HASH_LEN` | `1` | off | the card's hash pass and compare of a title buffer's copy cover the title buffer's own bytes (16-byte steps) instead of the whole 64 KiB-rounded buffer; for small copies, pair with `VICTORY_CARD_BUFFERS_MIN_KB=64`. Our form of CARD_HASH_WIDE is the 64-lane CARD_COMPARE, always on (no switch) | test/small3 2d156d0de (fxceless c36971b21) | no |
| `VICTORY_RT_SKIP_RESOLVE` | `1` | off | while the structure builder does not run, pipelines whose compute container names `CS_RaytracingResolve` (about 1,000) are not made ahead and their dispatches (direct or indirect, before the clamp) are left out. By name, no list, traced shadows kept, follows the builder (test/speed2's `VICTORY_SKIP_RT` is a 1,034-hash list, always). Frame line: "dispatches of the ray-tracing resolve programs ... left out per frame" | test/small3 355ef67fd (lordon1a c7dc354e3, after cool_fox 8cc31ab3) | no |
| `VICTORY_HEAP_FORGET` | `0` off | ON | a dropped host view heap takes its entries out of the heap passes' caches keyed by its address (slot ranges, unordered spans, "same as sent" notes): a new heap at the same address no longer inherits stale buffer copies, and the old copies are let go. | test/small3 a939d5a72 (after fxceless PR #1276, UwUCammie) | no |
| `VICTORY_SDWA_SEXT` | `0` off | ON | a 16-bit operation's SDWA destination WORD_0 with UNUSED_SEXT sign-extends bit 15 into bits 31:16 (was zero-filled); reserved dst_unused 3 refused. | test/small3 c5dd98428 (Marvisak PR #3124, KainanP) | YES (in the cache key) |
| `VICTORY_LOCK_STATS` | `1` | off | waits on the recording path's shared locks (slots, kept, ring, textures, heaps, views, buffers, placed, retired, submit), per lock, on the 5 s frame line; without it a lock is taken as before after one load of a once-set flag | test/locks adf68f33c | no |
| `VICTORY_HEAP_LOCKS` | `1`; `alt` 20 s each way in turn (logged) | off | `HostHeap::made` read under the shared side of a reader-writer lock for slots already made, `descriptor::VIEWS` and `SHADER_HEAPS` read shared, `HostHeap::kept` cut into 64 locks; without it every path takes the writer side alone, as the one lock before | test/locks 27c580714 (fxceless 5d659e0c8, PR #1146 part 1) | no |

## New in all19 (merged on top of all18)
| Switch | Values | Default | What it does | From | Shader |
|---|---|---|---|---|---|
| `VICTORY_MESH_CLAMP_IN_DRAW` | `1`; `2` also one wait for every shader write before a launched record when a dispatch or a storing draw came since the last such wait | off | every mesh pipeline gets an amplification stage of ours (`launch.hlsl`) that reads the title's indirect mesh record where its passes wrote it, bounds it as the clamp does (65,535 groups, the list's budget) and launches the mesh groups: a single indirect mesh record needs no clamp pass and none of its two waits; calls with a count buffer or several records keep the clamp on the same pipeline made without the stage | test/meshclamp 9e5d01b28 (fablinoo/kelvaroq patch 0041) | YES (every mesh shader declares a 4-byte payload; in the cache key) |
| `VICTORY_NAN_GUARD` (new values) | `scrub`: every NaN and both infinities become 0 in pixel colour exports and image stores of every stage; `scrub-exports`: colour exports alone (`1` and `stores` as before) | off | NaN/infinity scrubbing that also catches NVIDIA's NaN 0x7FFFFFFF and does not leave +-65504 for the bloom; under `scrub` an image store through an integer view also turns words 0x7F800000-0x7FFFFFFF and 0xFF800000-0xFFFFFFFF (-1 among them) into 0 | test/scrub b9ebab139 (trustno1-coder 77e13a974) | YES (the raw value is in the key) |
| `VICTORY_AMBIENT_SETTLE` | `<ms>[,<jump>[,<fresh s>]]`; `0` as before (0.04 a frame) | ON: `250,3` (read only with `VICTORY_AMBIENT`) | the stand-in's slow mean of the light around follows with a time constant of <ms>; a change by more than <jump> either way (a new room) is taken at once, and for <fresh> seconds after it the title's leftover light is bounded by the estimate itself | test/room1 4ef159e3f (avatarmanz b4ba4bbb3, 9576c45f1) | no (the stand-in's own HLSL: 16 root constants since test/pic1, 17 since test/pic2) |
| `VICTORY_AMBIENT_REFLECT` | a multiple of the light around | off (read only with `VICTORY_AMBIENT`) | after CS_UpdateHistory (found by name), every reflection brighter than that multiple of the light around is scaled down to it (the sky's unoccluded glaze on glossy floors indoors) | test/room1 f87637a03, 849247e67 (Belobug 44dcd4371) | no (a pass of ours) |
| `VICTORY_AMBIENT_REFLECT_AT` | `48`; anything else 40 | 40 (read only with `VICTORY_AMBIENT_REFLECT`) | MOOT in the game (runs pic1-cap/cap2: dwords 40 and 48 name ONE resource, the resolve's (1,152)/(1,200) image): which image of `CS_UpdateHistory` the hold acts on: table dword 40 (u31, where the program stores the colour UNPACKED) or 48 (u36, where it stores it PACKED as R9G9B9E5 when two of its constants are set; read from its code). With REFLECT on, the log says once what both are (`the images CS_UpdateHistory writes`: format, size, title resource) and the resolve's diffuse light (`the resolve's diffuse light image`) | test/pic1 | no (a pass of ours) |
| `VICTORY_AMBIENT_REFLECT_RESOLVES` | `1` | off (read only with `VICTORY_AMBIENT` and `VICTORY_AMBIENT_REFLECT`) | the reflections are also held right after the two programs that resolve them (`CS_TemporalResolve2_Opaque_*`, `CS_TemporalResolve1_Alpha_*`: the first 2D R32_UINT image at table (1,128) or (1,136)), each texel to REFLECT's multiple of the stand-in's estimate at the four points about it (a fifth of the room's slow mean), hue kept: the transparent surfaces' resolve (window panes, glasses) is one the CS_UpdateHistory hold does not reach | test/pic5 (avatarmanz PR #314, Serph91P 6c7e3a87c) | no (the stand-in's own HLSL) |
| `VICTORY_AMBIENT_SHADOW_MASKS` | `1` | off (read only with `VICTORY_AMBIENT`) | after each `CS_RTDistantShadowTrace*` program (the sun's distant traced shadows: trace, leak plug, temporal upsamples), every 2D float/unorm image it writes is filled with 1 (lit); integer images are left (logged once). Without the structure builder its rays walk structures nothing built | test/pic5 (ScarlettPort stand-in, via JMAN730/project-flamingo 83af519c8, ideas and program list; our own pass) | no (the stand-in's own HLSL) |
| `VICTORY_WRITE_ALIAS_TEXTURES` | `0` restores | ON | a write of ours into title memory (shader results put back, the title's copies, the title's kernel reads: `watch::writing`) already made the textures picked by index placed in the written range be looked at again; now also those placed at the other views of the written 64 KiB pages (at most `VICTORY_GUARD_ALIASES` views a page). The card writes skin and tattoo textures' memory through a buffer view; their uploads stayed stale (noise on skin, a new pattern each launch). Counter: the "textures picked by index" line's "pages at other views of memory we wrote looked at again". Test: the other view's page is listed, the written view's own pages not | test/pic4 (Belobug PR #139, Zetsukae a208d61db, faac6de43) | no |
| `VICTORY_PLANE_NOW` | `1` | off | a depth plane whose twin a list wrote and that the same list then uses (a draw's depth test, a shader read, a copy) gets its pass right there, before the use, not when the list closes; the title's pipeline, root signatures, heaps, targets, viewports, scissors and topology are set again (`State::replay`) and the draw's binding is made again. The memory line's "times used before the pass" then stays where it was (about +1 a frame without it) and the new "drawn at once (VICTORY_PLANE_NOW)" counts these passes. GPU test: "equal" draws in the same list see the list's classes with it, the old plane without it | test/pic3 (Op-lite/victory-scarlett silent-defects audit C4, ideas) | no |
| `VICTORY_E5_STAND_IN` | `32` | off (16-bit float, as before) | R9G9B9E5 render targets get R32G32B32A32_FLOAT on the host instead of R16G16B16A16_FLOAT: 16-bit float turns values past 65,504 into infinity (the console's format saturates at 65,408), later passes make NaN of it (black or purple where the light is brightest). Twice the memory for those targets; their pipelines are made anew (other target format), so the first run with it compiles pipelines | test/pic3 (Ranxanat0r/cool_fox-dlss-ranxanat0r 3eb180e12, Humb8l) | no (pipelines differ, shader code does not) |
| `VICTORY_AMBIENT_AIR_FINITE` | `0` restores | ON (read only with `VICTORY_AMBIENT` and `VICTORY_AMBIENT_AIR`) | the air pass counts a NaN in the scattered-light volume as nothing and an infinity as 64512 (the 10-bit float's most) before bounding it: an infinite lamp value was scaled by 0 into NaN, drawn black or purple/pink where the light is brightest. A clear bug fix with a GPU test | test/pic2 (Ranxanat0r/cool_fox-dlss-ranxanat0r dfc88ffc7, Humb8l) | no (the stand-in's own HLSL) |
| `VICTORY_AMBIENT_MAGENTA` | `1` | off (read only with `VICTORY_AMBIENT`) | texels whose green is below half their red and blue (and red or blue above 0.02) are memory the probes held, not light: in the resolve's diffuse light the title's leftover there counts as nothing; in the reflections (only with `VICTORY_AMBIENT_REFLECT`) the held light around (its colour times REFLECT's multiple) replaces them: the purple curtain panel, lavender white cloth | test/pic2 (Belobug 81ebb7bdc; avatarmanz issue #71, idea) | no (the stand-in's own HLSL) |
| `VICTORY_AMBIENT_SKIP_DARK` | a share of the 64x36 cells, `0.1` | off (read only with `VICTORY_AMBIENT`) | a frame whose light of visible surfaces is dark in all but that share of the cells (a cut, a fade, a cleared image on the first frame of a shot) is not followed: the stand-in's cells and slow mean stay as the last lit frame left them (without it such a frame takes the light around to nothing at once by SETTLE's jump) | test/pic1 | no (the stand-in's own HLSL) |
| `VICTORY_AMBIENT_MOST` | a multiple; `0` the estimate replaces the title's output everywhere | unset: 4 (as before) | with `VICTORY_AMBIENT`: how much of what the title left in the resolve's diffuse light image the stand-in keeps, as a multiple of its estimate (the house bedroom's foliage picture and dark blocks are such leftovers) | test/room1 f424f55f8 (darkworldhm 08fdf0b47, idea only) | no |
| `VICTORY_HEAP_WORKER` | `1` | off | the pass over textures picked by index runs on a worker thread of its own into upload lists that the next submission with flights sent ahead runs before the title's lists (slots made under one lock with that submission; a slot rewritten meanwhile is refused and due again); bake draws still upload on the recording thread. Refused with a warning without `VICTORY_RELAY` or `VICTORY_LATE_WAIT` | test/heapw 6f257f5f7, b3a6e21c2 (fxceless d4a528ba0, PR #1007) | no |

## New in all18 (merged on top of all17)
| Switch | Values | Default | What it does | From | Shader |
|---|---|---|---|---|---|
| `VICTORY_BAKE_WIDE` | `0` the four pixel shaders known before; `lists` the pixel and geometry shader lists | ON (both lists and the shape: no depth target, 4+ render targets, the first square and at least 512) | which draws count as the title's texture bakes and get the bake's care (`VICTORY_BAKE_TEXTURES_MS`, `VICTORY_BAKE_SOURCES_FIRST`): three ped-bake pixel shaders the old list missed, the ped baker's vertex and mesh stages, the bake's shape | test/bake2 087bca876 (codec_d cool_fox f4b48ad766) | no |
| `VICTORY_CLEAR_CHECKED` | `0` off | ON | the texture pass makes a slot found unmapped null only while the write it looked at is still the queued one (a texture mirrored in meanwhile was left black) | test/bake2 53231fab3 (fxceless PR #632, ryotdev) | no |
| `VICTORY_TARGET_VARIANTS` | `0` off | ON | a draw whose bound render target view has another format than its pipeline declares for that slot gets a host pipeline of the same shaders and state for the bound formats (made at its first draw, logged once, counted in the frame line) | test/variants 7c52fed2b (fxceless a35c55739; Marvisak PR #3109) | no (host pipelines only, made at first use) |
| `VICTORY_SYNC_BARRIER` | `1` | off | each of the title's pipeline syncs is one barrier on every shader write (copy lists skipped) | test/variants 70861a9cf (realblastex/flamingo a1178ce85) | no |
| `VICTORY_AMBIENT` | a share of the estimate (`0.8` most faithful, `1` all); `constant:<level>`; `0`/`off` | off | a stand-in for the traced indirect light while the structure builder is off: after CS_ScreenProbes_TemporalResolve its diffuse light image is raised to an estimate from the frame's screen radiance (and what the title left there capped at 4x it) | test/ambient 8d5dd30df (avatarmanz 82ca13473, 7da523a9d; Belobug 06031237, 4f45ab88, cd9f78ce) | no (the title's shaders are untouched; it records compute passes of its own) |
| `VICTORY_AMBIENT_SHADE` | `<strength>,<radius m>` | `1,0.6` (read only with `VICTORY_AMBIENT`) | the creases' shade of the stand-in | test/ambient | no |
| `VICTORY_AMBIENT_AIR` | a multiple (`1.5` most faithful) | off | with `VICTORY_AMBIENT`: the scattered-light volume CS_AccumulateLocalLightInscatteringVolume writes is bounded by that multiple of the light around (the brown veil indoors) | test/ambient | no |
| `VICTORY_SKIP_RT` | `1` | off | draws and dispatches of pipelines holding one of the 1,034 ray-tracing resolve and traced-shadow programs (`pipeline/rt_off.txt`) do not run, and such pipelines are not made ahead; the picture loses traced indirect light and traced shadows | test/speed2 588ca3b52 (trustno1-coder 77e13a974 list; fxceless PR #465 rule) | no |
| `VICTORY_PRESENT_TRACE` | `1` | off | log only: a line a present (new, repeat or BACK, each display plane's last write) and a count every 32 presents | test/light1 987d937cb | no |
| `VICTORY_PRESENT_NEWEST` | `1` | off | a present of a plane not written since it was last shown, while another plane holds a newer picture, shows the newer one (single-plane presents only: a composed present of several planes, `VICTORY_COMPOSE_PLANES`, is not changed) | test/light1 987d937cb | no |
| `VICTORY_TABLE_CONSTANTS_IN_PLACE` | `1` | off | translated shaders read a constant buffer that comes through a descriptor table as a raw buffer, and the title's heap slot of a constant buffer is a raw view of its host copy: a table holding a constant takes no dynamic slots (a constant no host copy holds on 16 bytes is copied to list scratch) | test/tables 322159026 (Marvisak 683ad7a99; CuteBooty b472495fb, 7a547593b) | YES (in the cache key: the first run with it translates again) |

Not in all18, for the next pass: `VICTORY_NAN_GUARD=scrub`/`scrub-exports` (test/scrub b9ebab139, shader),
`VICTORY_MESH_CLAMP_IN_DRAW` (test/meshclamp 9e5d01b28, shader). Both merge onto all18 with one small conflict
each at most (the cache key's switch list in shader_cache.rs for meshclamp; none for scrub).

## New in all17 (merged on top of all16)
| Switch | Values | Default | What it does | From | Shader |
|---|---|---|---|---|---|
| `VICTORY_REMAP_REOPEN` | `0` off | ON | the runtime's notice of a new mapping makes blocks a guard had made read-only writable again, not only forgotten (a write there was an access violation nobody took) | test/avfix | no |
| `VICTORY_DYNAMIC_SLOTS` | a number of slots, at least 262144 | unset: 606,784 with a heap map on a capped heap; 262,144 without a map | dynamic descriptor slots a capped host view heap takes from the title's pool (262,144 ran dry in the open world: tables refused, draws dropped) | test/slots (fxceless PR #1258) | no |
| `VICTORY_TEXTURE_BACK_CHECK` | `0` off | ON | a texture copy-back takes its place in title memory when its first writer is recorded, lands only if the texture still lies there, and overwrites only the bytes it covers | test/slots (DexNom PR #63) | no |
| `VICTORY_DYNAMIC_ACROSS` | `0` off | ON | the search for free dynamic slots also finds a run lying across the place it starts from | test/slots | no |
| `VICTORY_TABLES_EARLY` | `0` off | ON | with the relay, a flight's dynamic descriptor slots are given back once the card has run it, before its landing | test/slots (Korinocho eade35662) | no |
| `VICTORY_ALIAS_ALWAYS` | `0` off | ON | a placed view owes its copy from the card-written source until the list has made that copy, even after an upload of the view earlier in the list | test/slots (fxceless PR #1241) | no |
| `VICTORY_QUEUE_AGE` | `0` off | ON | log only: histogram of how many frames after its slot's write a texture is on the card | test/stream2 | no |
| `VICTORY_MAPPED_EXACT` | `0` off | ON | "how much of this texture is mapped" also looks between the first and last page (in turns, at most 1,024 blocks a frame); a line every 5 s counts the textures that had memory only there | test/stream2 5fe884eb9 (DexNom 921d8d3b4) | no |
| `VICTORY_MATERIAL_TEXTURE_DEMAND` | `1` | off | the textures a draw's material record names go ahead in the texture queue | test/stream2 (DexNom) | no |
| `VICTORY_TEXTURE_CATCH_UP_MS` | `<least>[,<most>]` ms (Neptune runs 2; most 20 unless said) | off | more texture passes within a frame when the title has written slots since the last one | test/stream2 (Neptune) | no |
| `VICTORY_STALE_KEEP_MS` | ms; 0 off | off | a slot goes on showing its old texture until the new one is made or its memory is mapped again, at most that long, while the card has room | test/stream2 (Neptune) | no |
| `VICTORY_STALE_KEEP_MAX_MS` | ms | 120000 | the cap of STALE_KEEP_MS; only read with it | test/stream2 | no |
| `VICTORY_VRAM_IDLE_MS` | `<normal>[,<moderate>[,<high>]]` ms (Neptune runs 15000,2500,1000) | unset: as before (5 s) | idle times after which host copies of textures are let go; NOT USED with `VICTORY_TEXTURE_KEEP=1` | test/stream2 (Neptune) | no |
| `VICTORY_TEXTURE_DROP_MIPS` | `1`-`4` | 0 | the host copy of a title-filled 2D texture (512 texels and up, 5 levels and up) leaves out its N largest levels; for 8 GB cards; the picture gets softer | test/res1 (soldsawn) | no |
| `VICTORY_COPY_PLAIN` | `1` | off | shader output goes back into title memory with a plain copy (the kernel's write behind it); small reads and reads from blocks that faulted go straight to the kernel; needs the guarded reads (on unless `VICTORY_GUARDED_READS=0`) | test/res1 (soldsawn) | no |
| `VICTORY_GROUP_SPLIT` | `1` | off | a wave64 compute program whose thread groups hold several console waves that share nothing runs each console wave as a host thread group of its own (else it does not translate: f133e35efed2a48a is one; its name is `CS_ShadowRenderTiledCascadesAlphaCBPPPC`, a sun-shadow tile writer, not the water's: test/pic1) | test/split (Corgansm/victory-corgi) | YES |
| `VICTORY_WAVE_PAIRS` | `1` the paired form after the exact forms and before the wave-wide one; `first` before every other retry form (for comparing pictures) | off | a console wave64 compute program on 32-lane host waves runs each console wave as ONE host wave, each thread two lanes (no barrier added inside a branch); f133e35efed2a48a takes it | test/pairs (fxceless PR #854) | YES (by program: only the programs whose answer changes translate again) |
| `VICTORY_WAVE_PAIRS_ONLY`, `VICTORY_WAVE_PAIRS_NOT` | `<hash>,...` | unset | the paired form for the listed programs alone, or not for the listed ones | test/pairs | YES (by program) |
| `VICTORY_PERMUTE_HALVES` | `0` off | ON | `ds_bpermute_b32` in a console wave64 program run as two host waves reads the lane its address names in either of them (64 LDS dwords between two group barriers) | test/loops2 664ee75e0 | YES |
| `VICTORY_SHORT_WAVE_LAST` | `0` off | ON | in a group that is not a multiple of 32 threads only the full host waves take their run from the counter; the short one takes the last run | test/loops2 a7afedf05 | YES |
| `VICTORY_LANE_MODULO` | `0` off | ON | `v_readlane_b32` and `v_writelane_b32` take a computed lane number modulo the console wave's size (lane 64 is lane 0) | test/loops2 533f14c8a | YES |
| `VICTORY_THREAD_TICKET` | `0` off | ON | a program with console lanes in a group of several 32-lane host waves takes its console threads from an LDS counter, not from the first lane's flat id rounded down (a 32 by 2 group gave half its threads the other half's numbers) | test/loops2 864343dff | YES |
| `VICTORY_AFTER_LOOP` | `1` | off | the loops a per-lane branch leaves count as that branch's region | test/loops2 9e9272327 (fxceless PR #1249) | YES |
| `VICTORY_REMAP_SAME` | `0`, `1`, `2` | 0 | a mapping of pages that are shown there already: 0 views taken away and made again (as before); 1 no view touched, the GPU side told of the bytes; 2 nor is the GPU side told | test/speed1 ec85139be (Neptune) | no |
| `VICTORY_ALIAS_RUNS` | `1` | off | a copy of page mappings makes one view for a run of pages that follow one another, not one a page | test/speed1 ec85139be (Neptune) | no |
| `VICTORY_IDLE_WITHOUT_RAYS` | `1` | off | the dispatches of compute program 39f0e9aeb156a61c (activates the screen probes' rays) are left out while the structure builder does not run (no `VICTORY_BVH_BUILDER=1`); the picture may change | test/speed1 4e2e1f6e1 (Belobug, Neptune) | no |
| `VICTORY_INDIRECT_BATCH` | `1` exact batches; `2` the source fork's default (a dispatch does not end a batch: KNOWN stale-argument risk); `3` as 1, but a dispatch that stores to no buffer does not end a batch | off | argument clamps of indirect executions of one record are made in batches by one compute pass of ours | test/clamps (avatarmanz 037596241) | no |
| `VICTORY_SECTION_MB` | MiB (`2` measured); unset or 0: one section a request, as before | off | the title's physical pages come from pagefile sections of that many MiB, not one section per request (6.1 GiB in one section, with views by the hundred thousand, stalls every map call) | test/all13y 95d04e060 (ka1qux 193d414c30) | no |
| `VICTORY_REMAP_LEAN` | `1` | off | section pages committed once through a view of ours are not committed again | test/all13y 95d04e060 (ka1qux b1d7012a44) | no |
| `VICTORY_SHADER_CACHE_KEY` | `exe` | the translator's fingerprint | what a build is known by in the translated-shader cache: `exe` = the executable's hash, as before (every new build translates everything again) | test/testkit d59bd65f6 | no (the cache only) |

Not switches: build profiles `release-lto` and `diag-lto` (thin LTO; test/res1; the runner's `diag` profile is
unchanged); `victory translate --digest` and `tools/shader-sweep.sh` (test/testkit: which programs a translation
switch changes, offline); `victory shader-cache <cache>/shaders --list`, `--prune-older-than <days> [--yes]`.

Warming offline (no game start): `victory.exe shader-cache <cache>/shaders --warm <folder of *.cdxil>... [--jobs N]
[--limit N]` translates the containers through this build's part of the cache under the `VICTORY_*` translation
switches of that process (set the same ones the run will have) and says how many were there already. Since the key
is the translator's, a build that changes nothing in crates/victory-shader finds the entries of the build before it;
entries keyed by executables (all16 and older) are read by no build from here on, so all17's first run with each
set of shader switches translates once more.

## Came with all16 (merged on top of all15b)
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

## Combining them (all16 notes)
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
- Checked offline (2026-11-25, unit tests of victory-gpu and victory-hle run with the switches in the environment):
  every new switch on at once passes, except that with `VICTORY_GUARD_REST` set two tests that assert the default
  (`watch::tests::a_range_that_begins_in_an_unmapped_block_is_guarded`,
  `watch::tests::a_write_through_another_view_is_reported_at_the_guarded_one`) fail by design: a resting block is
  told as written at every asking. Run `cargo test` with no `VICTORY_*` switch exported.

## Combining them in all20
- `VICTORY_HEAP_WORKER=1` with `VICTORY_HEAP_LOCKS=1`: allowed together. The heap worker's lock
  (`heap_worker::order`, taken in `host::Host::execute_late` with no lock of descriptor/host.rs and not the
  submission lock held) comes before every lock of that file; under it `take_ready` makes the uploaded slots through
  `mirror` (shared side of `made` first with HEAP_LOCKS, let go before the writer side: no upgrade), lets `made` go,
  and only then the submission lock is taken. HEAP_LOCKS changes no order, only who shares `made` and how `kept` is
  cut; the order comment at the top of descriptor/host.rs now names the worker's lock. Not run together in a game.
- `VICTORY_RT_SKIP_RESOLVE=1` beside `VICTORY_SKIP_RT=1`: both leave out the resolve programs (by name, and by the
  1,034-hash list that also holds the traced shadows); together they leave out the union, harmless. RT_SKIP_RESOLVE
  follows the structure builder, SKIP_RT does not. Both beside `VICTORY_IDLE_WITHOUT_RAYS`: one `no_rays` kind per
  pipeline (the activate program is no resolve), and a pipeline left out by any is not made ahead.
- `VICTORY_CARD_FETCH_INFLIGHT` reads nothing without `VICTORY_RELAY` (the run lines set RELAY=2).
- Translator changed (SDWA_SEXT, default on): the cache key folder with nothing set differs from all19's.

## Combining them in all19
- No pair refuses at start-up beyond `VICTORY_HEAP_WORKER=1` without `VICTORY_RELAY`/`VICTORY_LATE_WAIT` (refused
  with a warning, by its design; the run lines' `VICTORY_RELAY=2` satisfies it).
- `VICTORY_MESH_CLAMP_IN_DRAW` with `VICTORY_TARGET_VARIANTS`: a pipeline made again for the bound formats goes the
  same way as the title's own, so it is launched too. With `VICTORY_INDIRECT_BATCH`: mesh records the stage launches
  are not in a batch; the rest keep the batched clamp (test/meshclamp was made on test/clamps).
- `VICTORY_HEAP_WORKER` runs the same texture pass (`State::heap_textures`), so `VICTORY_CLEAR_CHECKED`,
  `VICTORY_STALE_KEEP_MS`, `VICTORY_MATERIAL_TEXTURE_DEMAND` and `VICTORY_TEXTURE_CATCH_UP_MS` act there as on
  the recording thread.
- Shader-changing switches now add MESH_CLAMP_IN_DRAW and the two new NAN_GUARD values. The translator changed
  (meshclamp, scrub), so the cache's key folder with nothing set differs from all18's.

## Combining them in all18
- No pair refuses at start-up; none has to stay apart. Merge resolutions that matter for pairs:
  - `VICTORY_CLEAR_CHECKED` with `VICTORY_STALE_KEEP_MS`: a slot found unmapped is first checked for its write
    (written again since the look: nothing is done, not even kept), then kept on if STALE_KEEP_MS asks, else made
    null. The pass then tells the queue the slot is away (all17's `texture_slot_away`), as before.
  - `VICTORY_SYNC_BARRIER` with `VICTORY_INDIRECT_BATCH=2`: a sync both ends the batch window (`State::synced`, as
    in all17) and, with the barrier switch, records the barrier.
  - `VICTORY_SKIP_RT` with `VICTORY_IDLE_WITHOUT_RAYS` and `VICTORY_SKIP_PIPELINES`: independent; a dispatch is left
    out if any of them names its pipeline. SKIP_RT's list holds none of the three programs `VICTORY_AMBIENT`
    follows, nor IDLE_WITHOUT_RAYS's 39f0e9aeb156a61c, so `VICTORY_AMBIENT` works beside `VICTORY_SKIP_RT=1`.
  - `VICTORY_AMBIENT` does nothing with `VICTORY_BVH_BUILDER=1` (the title's rays then find a structure).
  - `VICTORY_PRESENT_NEWEST` acts only on single-plane presents; composed presents (`VICTORY_COMPOSE_PLANES`,
    default on: the Series S interface over the game) pass through unchanged.
- One shader-changing switch came with all18, `VICTORY_TABLE_CONSTANTS_IN_PLACE` (opt-in; added to the cache key's
  list). Its lowering is in the translator, so the translator's fingerprint, and with it the cache's key folder,
  differ from all17's even with nothing set: the first run of this build translates everything (warmed offline).

## Combining them in all17
- No pair has to stay apart, and nothing refuses at startup. Four pairs needed care in the merge:
  - `VICTORY_WAVE_PAIRS=1` with `VICTORY_GROUP_SPLIT=1`: THE PAIRED FORM WINS for a program both could take (it is
    tried before the wave-wide form, where the split is; f133e35efed2a48a takes the paired form and is not
    split). A program the paired form refuses still gets the split. A paired program keeps the console's thread
    group count, so its dispatches need no change.
  - `VICTORY_GROUP_SPLIT=1` with `VICTORY_INDIRECT_BATCH`: a split program's indirect dispatches keep their own clamp
    (it multiplies z by the split; the batched clamp does not) and are counted as refused in the batch line.
  - `VICTORY_TEXTURE_DROP_MIPS` with `VICTORY_COPY_KEEP=1`: a whole-texture copy made level by level (host copies
    that leave out different numbers of levels) is not kept: the keeper goes, as at a resolve.
  - `VICTORY_STREAM_PRIORITY=1` with the new mapping path: the scatter call lifts the thread itself.
- `VICTORY_SECTION_MB` and `VICTORY_REMAP_LEAN` beside `VICTORY_REMAP_SAME` and `VICTORY_ALIAS_RUNS`: the cherry-pick
  applied cleanly; pages are compared by section and index everywhere, so a run of pages or a view never crosses
  a section, with any of the four on.
- The shader cache's key list holds every translation switch the merges brought (GROUP_SPLIT, AFTER_LOOP added to
  the new list by hand); the test `every_switch_the_translator_reads_is_accounted_for` fails if one is missing.
- One of two is used: with `VICTORY_TEXTURE_KEEP=1`, `VICTORY_VRAM_IDLE_MS` is ignored (by test/stream2's design).
- Streaming group judged safe together: `VICTORY_TEXTURE_KEEP=1 VICTORY_MATERIAL_TEXTURE_DEMAND=1
  VICTORY_TEXTURE_CATCH_UP_MS=2 VICTORY_STALE_KEEP_MS=10000`. Left out on purpose: STREAM_PRIORITY, DSTORAGE_* and
  PARTIAL_PROMPT (cs-all1's culprit is among them), VRAM_IDLE_MS (ignored beside TEXTURE_KEEP), DROP_MIPS (softer
  picture; for 8 GB cards), REMAP_SAME and ALIAS_RUNS (runtime mapping: a line of their own).
- Shader-changing switches now (all18 adds TABLE_CONSTANTS_IN_PLACE): LOOP_EXITS, NORM_EXACT, HEAP_BELOW_ZERO, LOCAL_LDS_SYNC, NAN_GUARD, GROUP_SPLIT,
  AFTER_LOOP, WAVE_PAIRS (and its two lists), PERMUTE_HALVES, SHORT_WAVE_LAST, LANE_MODULO, THREAD_TICKET. `tools/shader-sweep.sh` says offline which programs one of them changes.
- Swept offline (tools/shader-sweep.sh, the 3,605 programs of shaders-save-boat, VICTORY_HOST_MAX_WAVE=32):
  `VICTORY_GROUP_SPLIT=1 VICTORY_AFTER_LOOP=1` together change 3 compute programs (20b78ee12ecdb57d,
  ce79e9eba7ae0348, fa0bd10a35bfef3c) and let one more translate (f133e35efed2a48a, refused without them); all 4
  pass the validator and make pipelines on this card. Nothing else changes, nothing is lost.
- Tests on the software adapter take turns through ONE lock, `crate::warp_alone` (test/avfix); test/clamps' own
  `warp_turn` was not taken over.

## Shadow tiles with the stand-in (test/pic1, from the room captures pic1-cap2 / pic1-cap2off)
- `VICTORY_GROUP_SPLIT=1` makes `CS_ShadowRenderTiledCascadesAlphaCBPPPC` (f133e35efed2a48a) run. Without it the sun
  shadow tiles it owns keep stale values: the bed pillow wears 16-px staircase patches of sun through the blinds and the
  player is lit by sun through the roof (or black after a cut). With it the pillow is evenly shaded (cap2 740 against
  cap2off 766).
- Run it WITH `VICTORY_AMBIENT_REFLECT=1`: all20-split1 (GROUP_SPLIT without REFLECT) drew the player's skin black with
  the tattoos lighter than the skin; pic1-cap2 (GROUP_SPLIT + REFLECT=1) drew him lit in every room shot. Skin
  (PS 01999631eaafe981, lit in its own draw) reads the resolve's diffuse light through a placed R9G9B9E5 view filled
  by a copy after our raise; what REFLECT changes for it (the unbounded sky reflection in 0x1b7ff28ced0 reaching the
  skin's composite) is inferred, not traced.

