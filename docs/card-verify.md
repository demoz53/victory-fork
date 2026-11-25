# Checking host copies in the card's memory (`VICTORY_CARD_VERIFY`)

A diagnostics build only (`--features diagnostics`). Off, and in any other
build, nothing changes: one `OnceLock` read per present.

Title buffers of `VICTORY_CARD_BUFFERS_MIN_MB` or more are copied into the
card's own memory (`VICTORY_CARD_BUFFERS`, see `shadow.rs`). Such a copy has
two sides: the card's (what shaders read and write) and the CPU's (what title
memory is copied into, and the reference the card's writes are found against).
The code between them (`card_before`, `card_back`, `card_merge`,
`merge_piece`, `card_dirty`) is meant to keep them equal. Nothing checked
that end to end.

| Variable | Effect |
|----------|--------|
| `VICTORY_CARD_VERIFY=<n>` | every `n`-th presented frame, check every copy in the card's memory (0 or unset: off) |
| `VICTORY_CARD_VERIFY_MAX_MB=<mb>` | the most MB of copies checked in one frame (default 256; 0: no limit). A copy larger than what is left is skipped, and the next check starts with it, so every copy that fits the limit is reached in turn |

## What is compared, and when

At the start of the present, with the submission lock held, every flight in
the air is settled (with `VICTORY_RELAY` the landing thread's work, waited
for): no shader of ours runs, every fetch of a compare's flagged pieces is
merged and every plan is done with. Then the card's side of each copy is
copied into a read-back buffer in stretches of whole 64 KiB pieces and
compared with the CPU's side piece by piece.

Left out of the comparison: the offsets the CPU's side holds that have not
gone up to the card yet (`Card::dirty`); they differ by design until the next
submission, and are counted as `left out`. A copy that still names a pending
shader write is skipped (`skipped: in flight`).

## The log

One line per copy, then one for the frame (all `INFO`):

```
card verify: frame 100 copy 0xd1a00000+0xa600000 (buffer): ok, 2688 pieces equal, 0 B left out (not sent to the card yet), 41.3 ms
card verify: frame 100 copy 0xd3000000+0x400000 (buffer): MISMATCH 2 of 64 pieces, 17 B differ, 0 B left out (...), first 2: [0xd3012340(9 B), 0xd3030000(8 B)], 1.2 ms
card verify: frame 100 copy 0xd4000000+0x2000000 (buffer): skipped: in flight
card verify: frame 100 summary: 97 copies in the card's memory, 40 checked (250.0 MB, ...), ...
```

`first` names, for the first eight mismatching pieces, the title address of the
first differing byte and how many bytes of that piece differ. Zero
mismatches everywhere is what a sound card path gives; anything else names the
copy and the address to look at.

## Three-way classes of the differing pieces

For a piece the card-vs-CPU compare finds differing, each differing byte is
also compared with the title's own memory, read live (`ReadProcessMemory`,
with the submission lock held and the flights landed; the title's other
threads keep running, so this read is a snapshot). Per byte: **A** title ==
CPU != card (the card's byte is newer: shaders wrote it and it did not come
back), **B** title == card != CPU (the CPU's side is stale: a merge missed
it), **C** the title's byte equals neither (the title wrote it; neither side
has it). A piece whose bytes are all of one class is of it, else **D**
(mixed). A second line per mismatching copy gives pieces per class, how many
have a byte inside a range declared as written in the last two frames
(`Shadow::note_declared`; the first sample of each class is marked `*` if so),
the first four samples as `copy start + offset`, bytes per class, bytes
inside declared ranges, and pieces whose title memory could not be read. A
differing byte outside every recently declared range is a shader write nobody
declared.
