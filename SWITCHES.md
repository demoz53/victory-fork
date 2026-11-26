# SWITCHES.md - every switch added since test/all14n, in test/all23

`test/all23` = `test/all22` (7c275212b) + merges, in this order, of test/watch5 (2a61eb7d2, holds test/watch4),
test/async5 (f4d212aa8, holds test/async4 8798ddb11 and test/async3), test/leak1 (0af946da1), test/pic4 (fcf8630c2, holds test/pic3 6518051e4), test/stream3 (972fbdb77),
test/stable1 (f830f7c2e) and test/trim2 (46c8acccd; holds test/leak1). Left out (not finished when merged):
test/shader3, test/shader4.

ALL default-on changes since all21 (a run with nothing set does these; each line's switch restores all21's way):
- `VICTORY_GUARD_VIEWS_ONLY` on (all22, test/watch2): `VICTORY_GUARD_VIEWS_ONLY=0`.
- `VICTORY_AMBIENT_AIR_FINITE` on (all22, test/pic2; acts only with `VICTORY_AMBIENT` and `VICTORY_AMBIENT_AIR`):
  `VICTORY_AMBIENT_AIR_FINITE=0`.
- `VICTORY_PLACED_PADDING` on (all23, test/leak1): `VICTORY_PLACED_PADDING=0`.
- `VICTORY_OWNERLESS_VIEWS` 1,048,576 (all23, test/leak1): `VICTORY_OWNERLESS_VIEWS=0` (keeps all, as before).
- `VICTORY_RETIRE_AHEAD` 64 (all23, test/stable1): `VICTORY_RETIRE_AHEAD=0`.
- `VICTORY_PREWARM_READERS` on (all23, test/async5): a settle's prewarm leaves alone a one-sided copy that a flight
  in the air read: `VICTORY_PREWARM_READERS=0`.
- `VICTORY_WRITE_ALIAS_TEXTURES` on (all23, test/pic4): a write of ours into title memory also makes the textures
  placed at other views of those pages upload again: `VICTORY_WRITE_ALIAS_TEXTURES=0`.
- `VICTORY_VIEW_STATS` on, log only: the "dstorage views" line every 5 s while DirectStorage works (all23,
  test/stream3): `VICTORY_VIEW_STATS=0`.
- Fixes with no switch of their own (all23, test/watch5): the standing asking of `VICTORY_ASK_ONCE` (on since before
  all21) notes what it finds under a number above every asking made since (a page could be missed until its next
  write); `VICTORY_ASK_ONCE=0` turns that whole asking off. The other watch5/watch4 fixes act only with
  `VICTORY_JOURNAL_UNLOCKED`, `VICTORY_LIGHT_REFRESH` or `VICTORY_GUARD_OUTSIDE` set. test/async4's fixes act only with
  `VICTORY_ASYNC_SUBMIT` or `VICTORY_FILL_AHEAD` set (`FILL_AHEAD=2` is gone: it means 1 now).

`test/all22` = `test/all21` (530dbd273) + merges, in this order, of test/watch3 (a2b96a048, holds test/watch2),
test/async2 (57713c41b, holds test/async1), test/trim1 (e5d838e60), test/waits' tip (d41b5e680), test/spread
(c12a3e63f), test/pic2 (4aa55718b, holds test/pic1; based on all20) and test/prof (a8b7472ab; based on all20).
Two new default-on changes (rows 23 and 24 below), with their restore switches:
- `VICTORY_GUARD_VIEWS_ONLY` (on; `VICTORY_GUARD_VIEWS_ONLY=0` restores all21): the write watch's guards make view
  memory read-only and nothing else (the process heap went read-only: every thread waited, the 30-s watchdog).
- `VICTORY_AMBIENT_AIR_FINITE` (on, read only with `VICTORY_AMBIENT` and `VICTORY_AMBIENT_AIR`;
  `VICTORY_AMBIENT_AIR_FINITE=0` restores): the stand-in's air pass counts a NaN as nothing and bounds an infinity.

`test/all21` = `test/all20` (14a67b11a) + merges, in this order, of test/heapw's tip (4e380920d: the crash fix of
`VICTORY_HEAP_WORKER`, `resource::hold`), test/copies (bda62c82c) and test/waits (2c12a2d98, then again to its tip e95320672).
No new default-on change: every all21 switch is opt-in, so the default-on list and the restore line are all20's.

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
| 23 | `VICTORY_GUARD_VIEWS_ONLY` is on (the write watch's guards make view memory read-only and nothing else) | test/watch2 (all22; fxceless PR #1147) | `VICTORY_GUARD_VIEWS_ONLY=0` |
| 24 | `VICTORY_AMBIENT_AIR_FINITE` is on (with `VICTORY_AMBIENT` and `VICTORY_AMBIENT_AIR` only: the air pass stays finite) | test/pic2 4aa55718b (all22) | `VICTORY_AMBIENT_AIR_FINITE=0` |
| 25 | `VICTORY_PLACED_PADDING` is on (a placed view's shape leaves out the words the title does not write: resource words 1 and 13, an SRV's word 3 and the union words its dimension does not use; ends the placed-view leak) | test/leak1 (TaynanGT/Vice-Tay PR #27 f07e54a2b, ce914907f) | `VICTORY_PLACED_PADDING=0` |
| 26 | `VICTORY_OWNERLESS_VIEWS` is 1,048,576 (table entries of views no resource releases - constant buffer views and placed views that did not fit a record - are kept at most this many, oldest freed under a new generation; before: kept for the run) | test/leak1 (TaynanGT/Vice-Tay PR #27 d3626d257) | `VICTORY_OWNERLESS_VIEWS=0` |
| 27 | `VICTORY_RETIRE_AHEAD` is 64 (what a sweep or a slot change lets go waits for the running submissions and 64 more, not only the running ones) | test/stable1 (AntoninMichelq e90063015) | `VICTORY_RETIRE_AHEAD=0` |
| 28 | `VICTORY_WRITE_ALIAS_TEXTURES` is on (a write of ours into title memory also makes the textures placed at other views of those pages upload again) | test/pic4 fcf8630c2 (all23) | `VICTORY_WRITE_ALIAS_TEXTURES=0` |
| 29 | `VICTORY_PREWARM_READERS` is on (a settle's prewarm leaves alone a one-sided copy a flight in the air read) | test/async5 f4d212aa8 (all23) | `VICTORY_PREWARM_READERS=0` |

Everything restored (all15b behaviour, one line; all18 added BAKE_WIDE, CLEAR_CHECKED, TARGET_VARIANTS, all19 AMBIENT_SETTLE, all20 HEAP_FORGET and SDWA_SEXT, all22 GUARD_VIEWS_ONLY and AMBIENT_AIR_FINITE, all23 PLACED_PADDING, OWNERLESS_VIEWS, RETIRE_AHEAD, WRITE_ALIAS_TEXTURES, PREWARM_READERS):
`VICTORY_PLACED_PADDING=0 VICTORY_OWNERLESS_VIEWS=0 VICTORY_LOOP_EXITS=0 VICTORY_FAIR_DUE=0 VICTORY_RELOOK_MS=4000 VICTORY_REMAP_REOPEN=0 VICTORY_DYNAMIC_SLOTS=262144 VICTORY_TEXTURE_BACK_CHECK=0 VICTORY_DYNAMIC_ACROSS=0 VICTORY_TABLES_EARLY=0 VICTORY_ALIAS_ALWAYS=0 VICTORY_QUEUE_AGE=0 VICTORY_MAPPED_EXACT=0 VICTORY_PERMUTE_HALVES=0 VICTORY_SHORT_WAVE_LAST=0 VICTORY_LANE_MODULO=0 VICTORY_THREAD_TICKET=0 VICTORY_BAKE_WIDE=0 VICTORY_CLEAR_CHECKED=0 VICTORY_TARGET_VARIANTS=0 VICTORY_AMBIENT_SETTLE=0 VICTORY_HEAP_FORGET=0 VICTORY_SDWA_SEXT=0 VICTORY_GUARD_VIEWS_ONLY=0 VICTORY_AMBIENT_AIR_FINITE=0 VICTORY_RETIRE_AHEAD=0 VICTORY_WRITE_ALIAS_TEXTURES=0 VICTORY_PREWARM_READERS=0 VICTORY_SHADER_CACHE_KEY=exe`
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

## New in all23 (merged on top of all22)
From test/watch5 (holds test/watch4), test/async4 (holds test/async3), test/leak1, test/pic3, test/stream3 and
test/stable1, test/trim2; each row's "From" names its source.
| Switch | Values | Default | What it does | From | Shader |
|---|---|---|---|---|---|
| `VICTORY_GUARD_OUTSIDE` | `1`; `alt` (20 s on, 20 s off, said in the log with the guards' counts) | off | an asking of view memory notes the written blocks it guards again as read-only and clean under the guards' lock, and makes them read-only (`VirtualProtect`) after letting go of the guards' lock and the journal's, before its caller reads them. Another asking of those blocks waits for that; an opening meanwhile (fault, kernel write, remap) marks them disturbed and they end writable; light-refresh bits only once they stand. Blocks with other known views, or with a kernel write announced, are still protected under the lock; VIEWS_ONLY is kept (only view memory made read-only). Shortens the guards' lock holds that the title's threads wait on in the write-fault handler | fxceless PR #1315 (KainanP, 33c2b7f32) | no |
| `VICTORY_ASK_SKIP_VIEWS` | `1` | off | ranges whose first block the guards hold (view memory, which has no write watch) are left out of the kernel askings: the submission's one asking (ask_ahead runs, which otherwise fail and are asked again by halves down to single ranges; the lone ranges; ASK_ONCE_VIEWS ranges) and every asking (as GUARD_FIRST). Diagnostics builds: a "kernel calls" line after the journal-lock line (GetWriteWatch answered / failed / not made; hole checks made / from the cache; placeholder questions) | ours (test/watch5) | no |
| `VICTORY_HOLES_KNOWN` | `1` | off | a block the guards find uncommitted inside a placeholder reservation (asked once, `NtQueryVirtualMemory`, after fxceless 94ed363cc) is kept a hole until the runtime tells of a mapping there (`title_memory_remapped`), with no `VirtualQuery` at every asking; holes in ordinary reservations are asked as before. Counted on the "kernel calls" line | ours (test/watch5), placeholder question after fxceless 94ed363cc | no |
| `VICTORY_MERGE_MASK` | `1` (threshold 16 changed ranges a 64 KiB page), `<n>`, `alt` (20 s each way) | off | under flights' shader writes, a card copy's page with more changed ranges than the threshold costs the merge's budget (512 runs a submission) one run instead of one a run, and goes up as one row of a compute pass that writes the masked bytes from the CPU side and keeps the card side's others (byte-exact as the copies; the page hashed once). Frame line "card copies' pages that went up with a mask" | fxceless PR #1295 (KainanP, 55c9076f2) | no (our own pass, not the title's) |
| `VICTORY_CARD_ALL` | `1` | off | every title buffer's copy has two sides: the smallest go to the card's memory too, and one the card has no room for gets a card side in PC memory instead of one side only | avatarmanz/VIctory-7900XTX 5b422ba02 (via cobrez/Victory-Nvidia-4090-laptop) | no |
| (counter, no switch) | | always | frame line "the merge under flights' shader writes gave up per frame" by why: not write-watched, no answer, a card copy over the 512 runs, a PC copy without snapshot (each a settle for title memory changed under shader writes) | test/async3 | no |
| `VICTORY_SETTLE_PARTIAL` | `1` | off (relay only) | the settle for "title memory changed under shader writes" waits only until the newest flight that declared an overlapping range has landed, not for every flight in the air (all of them where that cannot be told). Frame line "settles ... that waited only for the flights writing it" | AntoninMichelq/VIctory-forkAMD 74768592a (Antonin Michel) | no |
| `VICTORY_STEP_INDEX` | `1`, `alt` | off | stale_under_flights, declared_writes and card_before find a step's copy by an index (one pending-writes question a copy, ranges deduplicated by a set) instead of a search per step: same copies, ranges, order | fxceless 84c5dc002 (fx issue #1317's set) | no |
| `VICTORY_LATE_FRAMES` | `<n>` | off (relay only) | a present waits until the flights sent before the present n frames earlier have landed (n = 1: a frame's work is home by the end of the next) | AntoninMichelq/VIctory-forkAMD 1bb10cb07 (fix/walls16) | no |
| `VICTORY_XMEM_PERSISTENT_VIEW` | `1` (also `true`/`yes`/`on`) | off | DirectStorage writes into the title's physical pages (in place, or copied for runs that cross sections) go through ONE kept view of each section, mapped whole at its first such write and kept until the section goes, instead of a view mapped, committed and unmapped (every unmap a TLB flush on every core) for every request and run of pages. Pages are committed once through it (the per-page bits of `VICTORY_REMAP_LEAN`, shared: a commit holds for every view of the section). A writer holds exactly its bytes; one whose bytes touch a live writer's takes a view of its own (counted `kept_refused`), as does a section the host will not map whole (a warn once a section) | fxceless PR #231 (HailLucifer666, 5c60b8906 / df57b2899 / 20d88e6fe / ad67d2557; idea also EFriseer) | no |
| `VICTORY_HEAP_RECHECK_EVERY` | `<frames>[,<looks a frame>]`; looks 128 unless said | off | once the look schedule of a texture picked by index (4, 12, 40, 120, 400 frames = 576; `VICTORY_HEAP_RECHECK=dense`: 5,376) is through, the slot is looked at again every `<frames>` frames for as long as it shows that texture instead of leaving the queue; at most `<looks>` such looks a frame (the rest put off 1 to `<frames>` frames, unhashed). A look reads at most 48 KiB (3 x 16 KiB; dense: 4 KiB a 64 KiB) and hashes it: 128 looks = 6 MiB a frame at most. Line: "changed after its look schedule ended" (first 50, then every 500th) with looks made and put off; the existing "changed after its upload" line shows such finds as `looks=5` | ryancase94/victory-Rycas 920423a97 (Rycas; the cap a frame is ours) | no |
| `VICTORY_TITLE_PRIORITIES` | `1` | off | the title's `SetThreadPriority` with "idle" (-15) or "lowest" (-2) gives "below normal" (-1); the rest as asked. Six title threads run at idle, the streaming pager among them; on a busy PC they ran once in seconds. The first 16 lifted are said in the log | avatarmanz/VIctory-7900XTX 0d065d352 (there default on, `=0` off) | no |
| `VICTORY_VIEW_STATS` | `0` | ON (log only) | every 5 s while DirectStorage works, an info line "dstorage views": requests, views of ours mapped per second, writes through a kept view, kept views refused, sections kept whole (since the start), commits through kept views, requests over 250 ms from a worker's start to their end and the slowest ms. `=0`: no line | test/stream3 | no |
| `VICTORY_RETIRE_AHEAD` | `<n>` submissions; `0` restores | ON (64: about 1.5 frames at the ~43 submissions a frame this build sends with the relay; Antonin's fork uses 16, about four of its frames) | `host::retire` (host copies let go by the sweeps, a descriptor slot's old resource, title buffer copies) keeps what it is given until the submissions running at the call AND n more have finished. Before, with no submission running it was dropped at once, while lists recorded but not yet submitted (the one being prepared, those the relay sends ahead) can name a host copy they do not hold: a page fault on freed memory, device removed 0x887A0005. Costs only memory held a few frames longer. Said once in the log. | AntoninMichelq/VIctory-forkAMD e90063015 (mich0206) | no |
| `VICTORY_D3D12_DRED` (new values) | `faults`; `light`; any other true value = full as before | off (diagnostics builds only) | `faults`: page-fault tracking only (no breadcrumbs, no markers): the faulting address and the runtime's live/recently freed allocations around it, by our names; cheap enough to leave on while playing. `light`: breadcrumbs too (which list stopped, at which command and of what kind, the pipelines it binds), no per-command marker texts. The removal report now always states the removal reason, whether a fault address was reported, and whether breadcrumbs came back (asked again for 0.5 s if empty); a removal found away from a submission's wait prints the data too. Every named object counted on the log. | fxceless/victory-fork 4cc775f57 | no |
| `VICTORY_D3D12_DRED_FRAME_NAMES` | `0` | on (with `light`/`faults`) | `0`: objects made while frames are drawn (list uploads and scratch, texture staging, read-back buffers) are not named in `light`/`faults` (one SetName each); always named in the full form | fxceless 4cc775f57 | no |
| `VICTORY_LOOKUP_CACHE` | `1` | off | each thread remembers (weakly) what its asks for a buffer's or window's host copy found, until any lookup table changes (a global count bumped where a buffer is registered or released, a copy taken out, a window made or let go): a copy made again at the same address is never found from memory. A window found so is checked and noted under the windows' lock. Memory line: "asks for a buffer's host copy since the last line: N answered from the asking thread's memory, M from the tables". An ask 745 -> 257 ns with 5 threads asking, 56 -> 44 ns alone | test/trim2 (fxceless 32be36d50, fork PR #1237) | no |
| `VICTORY_TEXTURE_ONCE` | `1` | off | a texture a draw reads (`readable`) and each state move of a title texture's host copy (`list::ensure_state`, `HostList::transition`) look the host copy up once and hand it on (before: three lookups per texture read, two per move) | test/trim2 (Marvisak/VIctory PR #3258, KainanP) | no |
| `VICTORY_DESC_CACHE` (extended) | `1` | off | now also: a host texture keeps its resource flags from when it is made; its legal states and "title fills it" answer from them (before: `GetDesc` up to three times per texture read and per move) | test/trim2 (#3258) | no |
| `VICTORY_TARGETS_ONCE` | `1` | off | a draw leaves out moving its render targets to the render-target state while the same targets are bound, the list's states number is unchanged (bumped at any new or changed entry, any settle, any reset; one counter for all lists) and no host copy of a title resource was made since. The depth target's move stays every draw (twin use, state by shaders) | test/trim2 (Marvisak/VIctory PR #3249, borosf) | no |
| `VICTORY_PLANE_NOW` | `1` | off | a depth plane whose twin a list wrote and that the same list then uses (a draw's depth test, a shader read, a copy) gets its pass right there, before the use, not when the list closes; the title's pipeline, root signatures, heaps, targets, viewports, scissors and topology are set again (`State::replay`) and the draw's binding is made again. The memory line's "times used before the pass" then stays where it was (about +1 a frame without it) and the new "drawn at once (VICTORY_PLANE_NOW)" counts these passes. GPU test: "equal" draws in the same list see the list's classes with it, the old plane without it | test/pic3 (Op-lite/victory-scarlett silent-defects audit C4, ideas) | no |
| `VICTORY_E5_STAND_IN` | `32` | off (16-bit float, as before) | R9G9B9E5 render targets get R32G32B32A32_FLOAT on the host instead of R16G16B16A16_FLOAT: 16-bit float turns values past 65,504 into infinity (the console's format saturates at 65,408), later passes make NaN of it (black or purple where the light is brightest). Twice the memory for those targets; their pipelines are made anew (other target format), so the first run with it compiles pipelines | test/pic3 (Ranxanat0r/cool_fox-dlss-ranxanat0r 3eb180e12, Humb8l) | no (pipelines differ, shader code does not) |
| `VICTORY_WRITE_ALIAS_TEXTURES` | `0` restores | ON | a write of ours into title memory (shader results put back, the title's copies, the title's kernel reads: `watch::writing`) already made the textures picked by index placed in the written range be looked at again; now also those placed at the other views of the written 64 KiB pages (at most `VICTORY_GUARD_ALIASES` views a page). The card writes skin and tattoo textures' memory through a buffer view; their uploads stayed stale (noise on skin, a new pattern each launch). Counter: the "textures picked by index" line's "pages at other views of memory we wrote looked at again". Test: the other view's page is listed, the written view's own pages not | test/pic4 (Belobug PR #139, Zetsukae a208d61db, faac6de43) | no |
| `VICTORY_PLACED_PADDING` | `0` restores | ON | a placed view's shape (the key its table entry is found by) leaves out the words the title does not write (resource words 1 and 13, an SRV's word 3, the union words its dimension does not use): the same view written again finds its entry instead of making a new one each time (the placed-view leak) | test/leak1 43e668fb8 (TaynanGT/Vice-Tay PR #27 d3626d257) | no |
| `VICTORY_OWNERLESS_VIEWS` | a number of entries; `0` keeps all, as before | ON: 1,048,576 | table entries of views no resource releases (constant buffer views, placed views that did not fit a record) are kept at most this many, oldest freed first; shape counts on the memory line | test/leak1 0af946da1 | no |
| `VICTORY_LOG_ASYNC` | `1` | off | log lines formatted on the logging thread, written to stderr by a thread of its own (`victory-log`); bounded at 8 MiB waiting: past it INFO lines are counted and dropped (reported), WARN/ERROR written at once after what came before; flushed at a crash, a panic, the host GPU's failure and the title's exit/TerminateProcess | p3n1cillin/yuriko e6a4c9f8e (GPL-3.0) | no |
| `VICTORY_PREWARM_READERS` | `0` restores | **ON** (fix) | a settle's prewarm (SETTLE_PREWARM, ONE_FLIGHT_PREWARM) leaves alone a one-sided copy that a flight still in the air read (the card reads the copy itself): the prewarm wrote the title's newer bytes under that read (Belobug #139; PR #203's test, Lux-Designs, ported as `prewarm_readers_tests`, fails without the fix) | test/async5 f4d212aa8 | no |

From test/watch4 (on top of test/watch3 a2b96a048):
`VICTORY_JOURNAL_UNLOCKED` fixed (72dc1fb20, review finding): an answer now waits for every asking in flight over its range.

How the watch switches combine: `VICTORY_JOURNAL_UNLOCKED` takes the kernel call and the whole guards' check off the JOURNAL lock (what the submitting thread waits for in `writes <- update_free`); `VICTORY_GUARD_OUTSIDE` takes the re-protection calls off the GUARDS lock (what `watch::install::on_write` waits for on the title's recording threads) and, without JOURNAL_UNLOCKED, off the journal lock too. Together: the journal lock covers bookkeeping only, the guards' lock the walk of the range only. LIGHT_REFRESH answers clean ranges before either lock's slow part.

From test/watch5 (on top of test/watch4 813c2805d):
Four review fixes, no new switch, each with a test that fails before it: (1) JOURNAL_UNLOCKED: an asking that keeps the journal's lock (the light refresh's self-check) settles before it answers; (2) LIGHT_REFRESH under JOURNAL_UNLOCKED: blocks passed over for an asking in flight get their bits when it lands (speed only); (3) ASK_ONCE (default on) / ASK_ONCE_VIEWS: what the standing asking notes goes under a number above askings made since, and a stamp never falls (a correctness fix of the default build too); (4) GUARD_OUTSIDE without JOURNAL_UNLOCKED: blocks armed outside the locks are noted again, above every number given, once they stand, and the asking answers under that number.


On VirtualProtect batching (the leader's item c): the guards already make each run of adjacent written blocks read-only in one call, split only where the run crosses views (`protect_runs`, one call per region with VIEWS_ONLY); about 2,050 calls a frame on all22-s-wft-prof. Left: the `VirtualQuery` that VIEWS_ONLY makes per run (`view_run`, ~2,100 a frame), which a set of blocks known to be views (cleared at remaps) could take away.

From test/async4 (on top of test/async3):
Fixes from review first (no switch of their own): ASYNC_SUBMIT's Running marker taken off on a panic (the process then ends instead of freezing); FILL_AHEAD holds a piece whose landing changes bytes a fill read even when the title also rewrote some (was hidden); FILL_AHEAD `=2` removed (a read-only title copy ahead of a miss left its destination wrong).

From test/stream3 (on top of all22; late assets and streaming stalls):
All three opt-in; none changes translated shader code. A run with nothing set behaves as all22, but for one new
log-only line, "dstorage views" (below).

Notes for stream3:
- `VICTORY_SECTION_MB=2` and `VICTORY_XMEM_PERSISTENT_VIEW=1` attack the same address-space and section locks from two
  sides: SECTION_MB makes each section 2 MiB (so the title's 6.1 GiB is ~3,100 sections and the large-page worker
  looks through a few dozen views of one), the kept view takes our per-request views away. Together: one kept view of
  2 MiB per section touched by DirectStorage (~3,100 views at most, the sections' total size in address space, no
  extra commit). Without SECTION_MB the kept view of the largest section is 6.1 GiB of address space at once.
- Requests whose destination pages cross from one section into the next are written run by run, each run through its
  section's kept view (with SECTION_MB=2 more requests cross: they still map nothing).
- `VICTORY_DSTORAGE_WORKERS` / `_HANDLES` / `_PRIORITY` act on the reading side and combine as they are; more
  workers writing one section's pages at once share its kept view while their bytes are apart.
  `VICTORY_DSTORAGE_RETRIES` still applies: a write that fails on the kept path takes a view of its own first.
- `VICTORY_HEAP_RECHECK_EVERY` keeps the slots it looks at in the texture queue: the frame line's queued and "waiting"
  counts include them (as they already included slots inside the schedule). Pick `<frames>` so that shown slots /
  frames stays under ~500 (the due list's catch-up mode starts at 1,024 due slots): 120 for tens of thousands.
- `VICTORY_TITLE_PRIORITIES=1` with `VICTORY_STREAM_PRIORITY=1`: the streaming thread is lifted from below normal to
  above normal as before (the lift asks for a priority under normal, which below normal still is).

## New in all22 (merged on top of all21)
From test/watch3 (holds test/watch2), test/async2 (holds test/async1), test/trim1, test/waits (CONSTANT_WAITS),
test/spread, test/pic2 (holds test/pic1) and test/prof; each row's "From" names its source.
| Switch | Values | Default | What it does | From | Shader |
|---|---|---|---|---|---|
| `VICTORY_GUARD_VIEWS_ONLY` | `0` restores | ON | the write watch's guards make view memory read-only and nothing else: free memory is no view (not noted absent, so not guarded once something is allocated there) and committed private memory and images inside a range to guard stay writable. Both ways made the process heap read-only, whose own code then faulted holding the heap lock while the guards' lock holder waited for the heap: every thread waits, no frame, the title's 30-s watchdog | fxceless PR #1147 (tippythetaidum, 772da7e19 + e8a93d2bc) | no |
| `VICTORY_ASK_ONCE_VIEWS` | `1` | off | with `VICTORY_ASK_ONCE` (on): the submission's one asking also takes in the VIEW memory its refreshes read: each refreshed range (cut to its copy) is asked once at submission start (kernel, else the guards; ranges the guards know are joined and guarded as one union) and the journal answers the refreshes inside, instead of a failed write-watch call plus a guards' check per refresh per flush. Title writes during preparing are seen by the next submission's asking (as for watched memory); a remap of view memory takes its ranges out of the asking. Measure on the 'write-watch askings' line: 'by one copy's refresh' should fall, 'for a submission's refreshes together' rise a little | fxceless PR #995 (tunczzz; only ask_once_views, without VIEWS_TRIM / AHEAD_INDEX) | no |
| `VICTORY_LIGHT_REFRESH` | `1`; `alt` (20 s on, 20 s off in turn, said in the log at each change; the bits are kept in both phases) | off | a lock-free bit per 64 KiB block that is read-only by our hand, clean, not faulted and not a hole, with every earlier write in the journal: an asking of view memory whose blocks all have their bit is answered from the journal with no guards' lock and no kernel call, and a host copy's refresh the journal shows has nothing new copies and allocates nothing. Bits set only under the journal's and the guards' locks right after the guards made the block clean; cleared wherever a block stops being clean (opened, faulted alias, remap). Holes get no bit (fxceless's hole bits and VICTORY_ABSENT_ONCE parts are not ported). Diagnostics builds: a 'light refresh' line after the 'write-watch askings' line (askings answered by the bits, refreshes skipped, askings that took the guards' lock); every build: an info line every 65,536 askings (`=1`) | fxceless 94ed363cc (after avatarmanz 2aa8134b5 / 74045e46), by hand | no |
| `VICTORY_LIGHT_REFRESH_CHECK` | `<n>`; `1` = 256; `all` = every one | off | with LIGHT_REFRESH: every n-th asking the bits would answer goes to the guards anyway under the same journal lock; a page they find written whose bit still stood is a mismatch (warned: first 16, then powers of two), one whose bit was cleared meanwhile is inconclusive; the guards' answer is used. Counted on the light refresh's lines | idea from CuteBooty/victory-8gb ea131ae01 (its check compares bytes; this one compares the guards' answer) | no |
| `VICTORY_JOURNAL_UNLOCKED` | `1` | off | the write watch's journal lock (`watch::JOURNAL`, one `Mutex<Journal>`: asking numbers, every page's last-written stamp, the standing asking ahead) is no longer held over an asking's `GetWriteWatch` (with reset) nor over the guards' check (`VirtualProtect`/`VirtualQuery` under the guards' lock). An asking takes its number and notes itself in flight under the lock, asks outside it, and notes what it found under it again; an answer of a range waits for every asking in flight over it when it reads the journal, whatever its number (a reset kernel asking hides its pages from the others; a2b96a048 waited only for lower-numbered ones, which let a higher-numbered asking that reached the kernel first hide a write from a lower one: fixed in test/watch4); stamps are never taken back. The light refresh's bits are not set for blocks an asking in flight covers; its self-check askings keep the lock throughout. Diagnostics builds: a "journal lock" line after the write-watch askings line (takes, takes that found it held, ms waiting, waits for askings in flight). Profile vf-bis-prof-s2: the submitting thread waited on this lock in `writes <- Shadow::update_free` for 22-40 % of its waiting | ours (test/watch3) | no |
| `VICTORY_ASYNC_SUBMIT` | `1` | off (needs `VICTORY_RELAY` or `VICTORY_LATE_WAIT`, and lists cut at their waits, the default; refused with a warning otherwise) | the title's submit call only puts its lists behind the queue's line; a thread of ours (`victory-submit`) prepares, settles and sends the lines in order while the title goes on. A part being sent keeps a marker at its line's head, so later submissions and signals stay behind it. The lines are run first, on the caller's thread, before: a present (so a present returns, as before, only once the frame is sent), a fence the title blocks on without an event, a synchronous submission of ours, and the title's page-mapping changes (`CopyPageMappingsBatchX`, done at once). Frame line: "submissions prepared off the title's submit calls per frame" (parts sent, their ms, and how long title threads stood at each of those four places) | our 99250881f (port of fxceless f85101b60) rebased onto all21 | no |
| `VICTORY_FLIP_LANDED` | `1` on, `0` off | unset: on with `VICTORY_ASYNC_SUBMIT=1`, off without it (needs `VICTORY_RELAY`/`VICTORY_LATE_WAIT`) | the title's blocking flip wait first waits (at most its own time-out, never past 1.5 s) until what it submitted before the present of the frame before the one it records has landed: shader writes copied back, read-backs, bottom-of-pipe values, fences. Frame line: "the title's flip waits per frame" (waits, those that found the frame before not landed, ms waited, time-outs) | fx PR #995 (tunczzz, 72dbd10) | no |
| `VICTORY_FILL_AHEAD` | `1` (`2` was removed in test/async4: a read-only title copy ahead of a miss left its destination wrong; 2 now means 1) | off | a fill of upload memory that only flights' shader writes meet (no bottom-of-pipe write, no read-back's range) is not waited for; it keeps a fingerprint, and each landing's merged bytes (`shadow::put_back`) are compared with the fills read ahead of that flight: a real miss holds that 64 KiB back (past 8 pieces every fill waits again). Frame line: "fills of upload memory over what flights may write, not waited for" (real misses, rewritten-since) | our aebce6702 (test/all13a: fxceless f85101b60 + fx #860, turbinedogbla-creator) brought onto all21 | no |
| `VICTORY_ALIAS_COPY` (new value) | `hold` | off (as before; `=1` unchanged) | carries a refresh over another copy's in-flight writes on the card as `=1`, and keeps each carried range in the air (as a read-back's range) until the carrying submission lands, so nothing refreshes either copy under the carry: `=1`'s race that darkened the picture | our 5b467e978 (test/all13a) brought onto all21 | no |
| (counters, no switch) | | always | frame lines "settles for a fill of upload memory per frame, by the first fill that met flights" (root CBV, table CBV, raw table constants, texture copies' sources, other, rectangle vertices; meeting a root UAV's rest-of-buffer declaration or another, diagnostics builds) and "shader writes declared per frame, by origin" (root UAVs, table views, numbered copies, heap views: count and MB) | test/async2 | no |
| `VICTORY_TIMING_LEAN` | `1` | off | fewer clock reads on the hot paths: list calls, title copies (submission steps), refreshed ranges and copy moves timed one call in 16 per thread and counted 16 times (counts exact, times estimates); per-slot heap-pass parts, dynamic-slot search, texture-alias time per bound view and wait checks not timed (print 0); `host::moment` (idle letting go) from the tick count. Untouched: VICTORY_FRAME_PARTS (timed in full), the frame line's GPU/copy-back/settle times, LOCK_STATS, WAIT_STATS | test/trim1 | no |
| `VICTORY_ALLOC_COUNT` | `0` | ON (counted, as before; diagnostics builds only) | `0`: the counting allocator counts nothing (memory line: "our own allocations" 0, big blocks "none"); read once without our heap at the first allocation. On path made cheaper always: the first 256 threads count on a stripe of their own with plain stores (0.3 ns a count against 8.7 ns for the locked additions of a shared stripe; a small block made and released costs 46 ns in the system heap) | test/trim1 | no |
| `VICTORY_EXECUTION_SPEED` | `1` | off | the process opts out of Windows' efficiency mode (power throttling EXECUTION_SPEED in the control mask, state 0: no EcoQoS clocks or slow-core preference when the window is behind or hidden), process-wide, all threads. Separate from `VICTORY_PROCESS_PRIORITY=above` (priority class) and `VICTORY_FAST_CORES` (fast-core CPU sets per thread kind): any combination works | test/trim1 (avatarmanz 51d3f4da0, fxceless a25370661) | no |
| `VICTORY_DROP_LATER` | `1` | off | on the submitting thread, what is let go after submissions ran goes to the existing reaper thread ("victory-reaper"): the objects retired until a submission finished (`bury`), and a landed list's draw state and uploads and any host list not kept for reuse (`list::recycle`). Descriptor slots are still given back on the spot; nothing is released earlier than before, only later and elsewhere | test/trim1 | no |
| `VICTORY_FAST_HASH` | `1` | off | the recording path's integer-keyed maps (a list's refreshed ranges, GPU-written ranges, textures, aliases, named writes; a draw state's fresh tables and root SRV/UAV slots; the texture staging, through, host-written and ahead maps; a heap's slot spans, view sizes and changed slots) hash with a multiply-rotate (FxHash-like, local, no crate) instead of SipHash; picked at start-up, one branch per hashed word. An insert or lookup of a 3-word key: 13.5 ns against 89.8 ns | test/trim1 | no |
| `VICTORY_DESC_CACHE` | `1` | off | a host copy's buffer width and flags come from its own record (read once when the copy is made) instead of a `GetDesc` call per buffer view made for a bind (buffer SRV/UAV views, root UAVs, title-memory views): 19.3 ns a call saved, one or two per view | test/trim1 | no |
| `VICTORY_BIND_LEAN` | `1` | off | diagnostics builds: a bound texture shader view no longer looks its host texture up again (table lock, reference counted, clock read) and asks its description only for the debug line "a shader reads a texture the title fills", unless debug lines are on (then as before). Log output unchanged at the usual levels | test/trim1 | no |
| `VICTORY_SPREAD` | `1`; `alt` (20 s as usual, 20 s widened, in turn, each change in the log) | off | every 500 ms a thread of ours looks at the title's threads: those the title pinned inside logical processors 0-15 (`SetThreadGroupAffinity`, which reaches Windows unchanged) are grouped by mask, and a group with more threads than processors also gets the spare processors (here 11 threads on 0x1F00 = logical 8-12 -> 0xFFFF1F00); smaller groups keep their mask. Log: `spread: the title's pinned threads by its mask: ...` | fxceless/victory-fork 39731795d (fxceless) | no |
| `VICTORY_SPREAD_SPARE` | hex mask (`0x` optional) | the slow cores (0xFFFF0000 here) | the processors SPREAD adds instead; only read with SPREAD (darkworldhm's sessions name every processor) | fxceless 39731795d | no |
| `VICTORY_CONSTANT_WAITS` | `lazy` | off (one full wait per copy) | a root or table constant view copied on the card (`card_constants`, under the default FILL_ON_CARD) waits for every earlier shader write only if a draw or dispatch that can store to buffer memory (or of unknown code) was recorded since the list's last full wait; copies between two such draws share one wait. WAIT_STATS counts these waits by site (root / table constants) and the ones left out | test/waits (the rule of fxceless PR #995 `VICTORY_BARRIER_LAZY_UAV`, tunczzz, for this wait only) | no |
| `VICTORY_AMBIENT_REFLECT_AT` | `48`; anything else 40 | 40 (read only with `VICTORY_AMBIENT_REFLECT`) | MOOT in the game (runs pic1-cap/cap2: dwords 40 and 48 name ONE resource, the resolve's (1,152)/(1,200) image): which image of `CS_UpdateHistory` the hold acts on: table dword 40 (u31, where the program stores the colour UNPACKED) or 48 (u36, where it stores it PACKED as R9G9B9E5 when two of its constants are set; read from its code). With REFLECT on, the log says once what both are (`the images CS_UpdateHistory writes`: format, size, title resource) and the resolve's diffuse light (`the resolve's diffuse light image`) | test/pic1 | no (a pass of ours) |
| `VICTORY_AMBIENT_AIR_FINITE` | `0` restores | ON (read only with `VICTORY_AMBIENT` and `VICTORY_AMBIENT_AIR`) | the air pass counts a NaN in the scattered-light volume as nothing and an infinity as 64512 (the 10-bit float's most) before bounding it: an infinite lamp value was scaled by 0 into NaN, drawn black or purple/pink where the light is brightest. A clear bug fix with a GPU test | test/pic2 (Ranxanat0r/cool_fox-dlss-ranxanat0r dfc88ffc7, Humb8l) | no (the stand-in's own HLSL) |
| `VICTORY_AMBIENT_MAGENTA` | `1` | off (read only with `VICTORY_AMBIENT`) | texels whose green is below half their red and blue (and red or blue above 0.02) are memory the probes held, not light: in the resolve's diffuse light the title's leftover there counts as nothing; in the reflections (only with `VICTORY_AMBIENT_REFLECT`) the held light around (its colour times REFLECT's multiple) replaces them: the purple curtain panel, lavender white cloth | test/pic2 (Belobug 81ebb7bdc; avatarmanz issue #71, idea) | no (the stand-in's own HLSL) |
| `VICTORY_AMBIENT_SKIP_DARK` | a share of the 64x36 cells, `0.1` | off (read only with `VICTORY_AMBIENT`) | a frame whose light of visible surfaces is dark in all but that share of the cells (a cut, a fade, a cleared image on the first frame of a shot) is not followed: the stand-in's cells and slow mean stay as the last lit frame left them (without it such a frame takes the light around to nothing at once by SETTLE's jump) | test/pic1 | no (the stand-in's own HLSL) |
| `VICTORY_PROFILE` (new form) | a number of milliseconds (`2`); a value that is not a number is the old form (a file for the stall profiler) | off | diagnostics builds: a sampler thread (above normal) looks at every thread every <ms>; a thread whose cycle count moved less than 100,000 since the last pass counts as waiting where it was last seen, any other is suspended only to read its registers, cycle count and up to 64 KiB of its stack (no allocation, log or lock meanwhile), and the copy is unwound after it runs again. Log lines `PROFILE ...` every 30 s (the last 30 s) and at the title's exit (the whole run; not after `Stop-Process`): the critical thread (most time running or waiting in our code: running and waiting ms a frame, its top 3 wait sites); totals per role (unnamed threads with title frames are `title/submits`, `title/records`, `title/audio`, `title/loads` or `title/other` by the names of their functions); per thread with over 1% of all running samples (6 in full, 30 more in one line), its share, running vs waiting, the scheduler's states (running / ready but not running / blocked, its main wait reason; read with NtQuerySystemInformation every 20 ms, less often when the read costs more than 2% of a core), its affinity and ideal processor, running by leaf module (title, victory, system, d3d, nvidia, other), its top 12 functions of OUR crates (std, core, alloc, hashbrown, windows and the copystat allocator shims passed over and named as `(in Condvar::wait)`), its top 8 wait sites and top 5 contended-lock sites as `ours <- our caller (in library call)`; then a process-wide top 25. The first line of each report gives the sampler's own cost (time a pass, threads suspended a pass and for how long, the state read's cost). Under 250 lines a report | test/prof | no |

With `VICTORY_FAST_CORES`: FAST_CORES selects CPU sets (fast cores 0-15), SPREAD sets affinity masks; Windows runs a
thread on the part of its mask that is also selected. So a title thread that submits, selected by FAST_CORES=3 or 4,
keeps to the fast processors of its (widened) mask, 8-12, and SPREAD changes nothing for it; the title's other
pinned threads (its recording and job threads) get the full widened mask either way. Our own threads are not pinned
and never touched. `VICTORY_PROCESS_PRIORITY=above` is independent (priority class, not processors). While a test
runs, the governor keeps compilers on 16-31 at idle priority: those are SPREAD's default spare processors, so give a
SPREAD run no build beside it (or `VICTORY_SPREAD_SPARE=FFFFC000`: fast core 7 = logical 14-15, which fxceless found free of pinned threads, plus the slow cores).

## New in all21 (merged on top of all20)
| Switch | Values | Default | What it does | From | Shader |
|---|---|---|---|---|---|
| `VICTORY_STREAM_COPY` | `1` | off | refreshes into host copies and snapshots of shader-written ranges (4 KiB and more) written with AVX2 streaming stores, fault-safe (avatarmanz #98, Joly0) | test/copies (bda62c82c) | no |
| `VICTORY_COPY_THREADS` | `2`-`8` | off | a refresh copy of 2 MiB or more cut into n pieces on a pool of its own (fast cores with FAST_CORES>=2); the caller waits for all | test/copies (bda62c82c) | no |
| `VICTORY_GUARD_FIRST` | `1` | off | view memory the guards already hold goes to the guards without a failing write-watch call first (under the journal lock) | test/copies (bda62c82c) | no |
| `VICTORY_TITLE_COPY_PLAIN` | `1` | off | the title's buffer copies done at submission (and copy-backs into title memory) are made in place, the kernel's copy only when a page stops it | test/copies (bda62c82c) | no |
| `VICTORY_WAIT_STATS` | `1` | off | one line after the 5 s frame line: global UAV barriers a frame by cause (title groups, title barriers, clamps before/after, batched clamps, syncs, ours), one-resource UAV barriers, what the title's barrier groups hold, settles by why with ms standing, "title memory read" settles by reader, and the submitting thread's waits for staging, synchronous submissions and the submission lock | test/waits 8c87fc403 | no |
| `VICTORY_CLAMP_WAITS` | `0` | full wait (as before) | the wait after each in-place indirect clamp is a UAV barrier on the list's budget buffer only (none if no record ran); the wait before stays | test/waits c61be0389 (lordon1a 32b5e420b, after Op-lite f748fb557a, Trexies 6b358fe83) | no |
| `VICTORY_BARRIER_GROUPS` | `1` | off (fixed in e95320672: named barriers only on textures made for unordered access, else the global barrier; game test pending) | the title's barrier groups: no barrier when none of their barriers orders shader writes; UAV barriers on the named textures' host copies (and twins) when only textures do; one global barrier otherwise (any global or buffer barrier, a texture with no host copy) | test/waits f81db3d21 (after Marvisak PR #642, nathanbzrr 6264e86a6) | no |
| `VICTORY_FILL_TITLE` | `1` | off | a fill of upload memory or title buffer copy of at most 64 KiB does not settle flights when the title wrote that view memory since a copy took it, or no shader write landed in its pages in 600 frames (bottom-of-pipe writes and read-backs still waited for) | test/waits 49337278e (avatarmanz 9fea7434c) | no |

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

