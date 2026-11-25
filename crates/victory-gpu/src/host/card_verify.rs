//! Diagnostics (`diagnostics` feature only): `VICTORY_CARD_VERIFY=N` checks,
//! every N-th presented frame, that each host copy held in the card's memory
//! ([`crate::shadow::cards`]) holds the same bytes as its CPU side.
//!
//! A copy in the card's memory has two sides: the card's (what shaders read
//! and write) and the CPU's (`Shadow::data`, which title memory is copied
//! into and which is the reference the card's writes are found against).
//! Every path of `card_before` / `card_back` / `card_merge` exists to keep
//! them equal, except for bytes the CPU's side holds that have not gone up to
//! the card yet (`Card::dirty`). Nothing checked it end to end.
//!
//! When the sides are comparable: with the submission lock held, after every
//! flight has been settled (`Host::settle`: with `VICTORY_RELAY` that is the
//! landing thread's work, waited for), so no shader of ours runs, every
//! fetch has been merged and `card_done` has run. A copy that still names a
//! pending shader write (`Shadow::writes_pending`) is skipped, and the
//! offsets in `Card::dirty` are left out (counted as `excluded`). Everything
//! the CPU wrote to a copy went through a submission, which the lock holds
//! off, so the CPU side does not move while it is read.
//!
//! Off (the variable unset or 0): one `OnceLock` read per present.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::Instant;

use super::*;
use crate::shadow::{CardVerifyView, ClassReport, VerifyReport, carded_copies, verify_classify, verify_compare};

/// The card's side comes down to the PC in stretches of this many bytes.
const CHUNK: u64 = 64 << 20;

/// `VICTORY_CARD_VERIFY` (every N-th frame; 0 or unset: off) and
/// `VICTORY_CARD_VERIFY_MAX_MB` (the most bytes of copies checked in one
/// frame; 256 unless set; 0: no limit).
fn settings() -> (u64, u64) {
    static SETTINGS: OnceLock<(u64, u64)> = OnceLock::new();
    *SETTINGS.get_or_init(|| {
        let number = |name: &str| std::env::var(name).ok().and_then(|v| v.trim().parse::<u64>().ok());
        let every = number("VICTORY_CARD_VERIFY").unwrap_or(0);
        let max = number("VICTORY_CARD_VERIFY_MAX_MB").unwrap_or(256).saturating_mul(1 << 20);
        if every > 0 {
            tracing::info!(
                every,
                max_mb = max >> 20,
                "card verify: every {every} frames each host copy in the card's memory is compared with its CPU side \
                 (VICTORY_CARD_VERIFY; at most {} MB of copies a frame, VICTORY_CARD_VERIFY_MAX_MB; 0 = all)",
                max >> 20
            );
        }
        (every, max)
    })
}

/// Where the next check starts in the copies (ordered by address) when the
/// limit left some out last time.
static NEXT: AtomicUsize = AtomicUsize::new(0);
/// The last frame checked: a frame is presented once, but be sure.
static LAST: AtomicU64 = AtomicU64::new(0);

/// Called by [`Host::present`] with the frame's number.
pub(super) fn frame(host: &Host, frame: u64) {
    let (every, max) = settings();
    if every == 0 || frame == 0 || !frame.is_multiple_of(every) || LAST.swap(frame, Ordering::Relaxed) == frame {
        return;
    }
    if !crate::shadow::cards() {
        tracing::info!(frame, "card verify: copies are not held in the card's memory (VICTORY_CARD_BUFFERS=0): nothing to check");
        return;
    }
    let began = Instant::now();
    // The submission lock: nothing is prepared or sent while this runs.
    let mut submitted = host.submitted.lock().unwrap_or_else(|e| e.into_inner());
    // Every flight in the air lands (merged, copied back, `card_done` run).
    host.settle(&mut submitted, WHY_SYNC);
    // SAFETY: plain object creation; the list, fence and event are ours and
    // used under the submission lock only.
    let Some(mut tool) = (unsafe { Tool::new(host) }) else {
        tracing::warn!(frame, "card verify: no list, fence or buffer to read the card's side with");
        return;
    };
    let mut copies = carded_copies();
    copies.sort_by_key(|c| c.card_verify_view().map_or(0, |v| v.start));
    let mut totals = Totals::default();
    let count = copies.len();
    let first = if count == 0 { 0 } else { NEXT.load(Ordering::Relaxed) % count };
    let mut resume = None;
    let mut left = if max == 0 { u64::MAX } else { max };
    for i in 0..count {
        let copy = &copies[(first + i) % count];
        let Some(view) = copy.card_verify_view() else {
            continue;
        };
        let what = format!("{:#x}+{:#x} ({})", view.start, view.len, if view.window { "window" } else { "buffer" });
        if view.in_flight {
            totals.in_flight += 1;
            tracing::info!("card verify: frame {frame} copy {what}: skipped: in flight");
            continue;
        }
        if view.len as u64 > left {
            // The next check starts here, so every copy is reached in turn.
            resume.get_or_insert((first + i) % count);
            totals.over_limit += 1;
            continue;
        }
        left = left.saturating_sub(view.len as u64);
        let mut report = VerifyReport::default();
        let mut classes = ClassReport::default();
        let waited = Instant::now();
        match tool.check(&view, &mut report, &mut classes) {
            Ok(moved) => {
                totals.add(&report, view.len as u64);
                let first: Vec<String> = report.first.iter().map(|(at, n)| format!("{at:#x}({n} B)")).collect();
                if report.bad_pieces == 0 {
                    tracing::info!(
                        "card verify: frame {frame} copy {what}: ok, {} pieces equal, {} B left out (not sent to the card yet), {:.1} ms{}",
                        report.pieces,
                        report.excluded,
                        waited.elapsed().as_secs_f64() * 1000.0,
                        if moved { ", NOTE: its unsent ranges changed meanwhile" } else { "" }
                    );
                } else {
                    tracing::info!(
                        "card verify: frame {frame} copy {what}: MISMATCH {} of {} pieces, {} B differ, {} B left out (not sent to the card yet), \
                         first {}: [{}], {:.1} ms{}",
                        report.bad_pieces,
                        report.pieces,
                        report.bad_bytes,
                        report.excluded,
                        report.first.len(),
                        first.join(", "),
                        waited.elapsed().as_secs_f64() * 1000.0,
                        if moved { ", NOTE: its unsent ranges changed meanwhile" } else { "" }
                    );
                    tracing::info!("card verify: frame {frame} copy {what}: {}", class_line(&classes, view.start));
                }
            }
            Err(e) => {
                totals.failed += 1;
                tracing::warn!("card verify: frame {frame} copy {what}: skipped: {e}");
            }
        }
    }
    if let Some(at) = resume {
        NEXT.store(at, Ordering::Relaxed);
    }
    tracing::info!(
        "card verify: frame {frame} summary: {count} copies in the card's memory, {} checked ({:.1} MB, {} pieces), {} with a mismatch \
         ({} pieces, {} B differ), skipped {} in flight, {} over the limit (VICTORY_CARD_VERIFY_MAX_MB={}; the next check starts \
         there), {} failed; {} B left out as not sent to the card yet; {:.0} ms in all (every {every} frames)",
        totals.checked,
        totals.bytes as f64 / (1 << 20) as f64,
        totals.pieces,
        totals.mismatched,
        totals.bad_pieces,
        totals.bad_bytes,
        totals.in_flight,
        totals.over_limit,
        max >> 20,
        totals.failed,
        totals.excluded,
        began.elapsed().as_secs_f64() * 1000.0,
    );
}

/// The line that tells a mismatching copy's pieces by who holds the card's
/// bytes (see [`ClassReport`]); samples are `copy start + offset`, `*` when a
/// range declared as written in the last two frames holds the byte.
fn class_line(c: &ClassReport, start: usize) -> String {
    const NAMES: [&str; 4] = [
        "A title==CPU!=card (card newer: not returned)",
        "B title==card!=CPU (CPU side stale: merge missed)",
        "C title!=card, title!=CPU (title wrote it: neither side has it)",
        "D mixed",
    ];
    let mut out = String::from("classes of the differing pieces (title memory read live, the title's threads still run):");
    for (i, name) in NAMES.iter().enumerate() {
        let first: Vec<String> = c.first[i].iter().map(|&(at, held)| format!("{start:#x}+{at:#x}{}", if held { "*" } else { "" })).collect();
        out += &format!(" {name}: {} pieces ({} with a declared byte) [{}];", c.pieces[i], c.declared[i], first.join(", "));
    }
    out + &format!(
        " bytes A/B/C {}/{}/{}, {} B inside declared ranges, title unreadable in {} pieces",
        c.bytes[0], c.bytes[1], c.bytes[2], c.bytes_declared, c.unread
    )
}

/// Sums of a frame's check.
#[derive(Default)]
struct Totals {
    checked: u64,
    mismatched: u64,
    bytes: u64,
    pieces: u64,
    bad_pieces: u64,
    bad_bytes: u64,
    excluded: u64,
    in_flight: u64,
    over_limit: u64,
    failed: u64,
}

impl Totals {
    fn add(&mut self, report: &VerifyReport, bytes: u64) {
        self.checked += 1;
        self.mismatched += u64::from(report.bad_pieces > 0);
        self.bytes += bytes;
        self.pieces += report.pieces;
        self.bad_pieces += report.bad_pieces;
        self.bad_bytes += report.bad_bytes;
        self.excluded += report.excluded;
    }
}

/// What the card's side is read with: a list, a fence, an event, and a
/// buffer in the PC's memory the stretches come down to.
struct Tool<'a> {
    host: &'a Host,
    allocator: ID3D12CommandAllocator,
    list: ID3D12GraphicsCommandList,
    fence: ID3D12Fence,
    event: HANDLE,
    sent: u64,
    buffer: ID3D12Resource,
    mapping: usize,
}

impl<'a> Tool<'a> {
    /// # Safety
    /// Under the submission lock, with nothing of ours in the air.
    unsafe fn new(host: &'a Host) -> Option<Self> {
        let (allocator, list) = host.make_prelude()?;
        // SAFETY: plain object creation.
        let (fence, event) = unsafe {
            let fence: ID3D12Fence = host.device.CreateFence(0, D3D12_FENCE_FLAG_NONE).ok()?;
            (fence, CreateEventW(None, false, false, None).ok()?)
        };
        let (buffer, mapping) = host.shared_buffer(CHUNK)?;
        name(&buffer, || "card verify read-back".into());
        Some(Tool {
            host,
            allocator,
            list,
            fence,
            event,
            sent: 0,
            buffer,
            mapping: mapping as usize,
        })
    }

    /// Reads the card's side of the copy in stretches that end on piece
    /// boundaries, and compares each with the CPU's side. Whether the
    /// copy's unsent ranges ([`CardVerifyView::dirty`]) changed meanwhile.
    fn check(&mut self, view: &CardVerifyView, report: &mut VerifyReport, classes: &mut ClassReport) -> Result<bool, String> {
        let (start, len) = (view.start, view.len);
        let mut at = 0usize;
        while at < len {
            // A stretch of whole pieces, at least one.
            let mut to = at;
            while to < len && (to == at || (to - at) as u64 + PIECE_BYTES <= CHUNK) {
                to = (((start + to) / PIECE_BYTES as usize + 1) * PIECE_BYTES as usize - start).min(len);
            }
            self.read(&view.buffer, at, to - at)?;
            // SAFETY: the mapping of the read-back, `to - at` bytes of it
            // written by the copy that has run; the CPU side's mapping holds
            // `len` bytes, and it does not move under the submission lock.
            let (card, cpu) = unsafe {
                (
                    std::slice::from_raw_parts(self.mapping as *const u8, to - at),
                    std::slice::from_raw_parts((view.cpu + at) as *const u8, to - at),
                )
            };
            let bad = report.bad_pieces;
            verify_compare(start, at, card, cpu, &view.dirty, report);
            if report.bad_pieces > bad {
                // The pieces that differ, against the title's own memory.
                let mut live = |address: usize, n: usize| {
                    let mut bytes = vec![0u8; n];
                    read_into(address as *const u8, &mut bytes).then_some(bytes)
                };
                verify_classify(start, at, card, cpu, &view.dirty, &view.declared, &mut live, classes);
            }
            at = to;
        }
        // The unsent ranges were taken before the read: the same now?
        let mut now = self.host_dirty(view);
        now.sort_unstable();
        Ok(now != view.dirty)
    }

    /// The copy's unsent ranges now, found again by its CPU side's address.
    fn host_dirty(&self, view: &CardVerifyView) -> Vec<(usize, usize)> {
        carded_copies()
            .iter()
            .filter_map(|c| c.card_verify_view())
            .find(|v| v.cpu == view.cpu)
            .map_or_else(|| view.dirty.clone(), |v| v.dirty)
    }

    /// Copies `len` bytes at `at` of the card's side into the read-back
    /// buffer and waits for the card.
    fn read(&mut self, card: &ID3D12Resource, at: usize, len: usize) -> Result<(), String> {
        self.sent += 1;
        // SAFETY: the list's last run was waited for; live buffers; the
        // fence and event are ours and the queue is the host's, under the
        // submission lock.
        let ran = unsafe {
            (|| -> windows::core::Result<()> {
                self.allocator.Reset()?;
                self.list.Reset(&self.allocator, None)?;
                self.list.CopyBufferRegion(&self.buffer, 0, card, at as u64, len as u64);
                self.list.Close()?;
                self.host.queue.ExecuteCommandLists(&[Some(self.list.cast()?)]);
                self.host.queue.Signal(&self.fence, self.sent)?;
                self.fence.SetEventOnCompletion(self.sent, self.event)?;
                WaitForSingleObject(self.event, INFINITE);
                Ok(())
            })()
        };
        ran.map_err(|e| format!("the card's side could not be read ({e})"))?;
        // SAFETY: a plain query.
        if let Err(e) = unsafe { self.host.device.GetDeviceRemovedReason() } {
            return Err(format!("the device was removed ({e})"));
        }
        Ok(())
    }
}

impl Drop for Tool<'_> {
    fn drop(&mut self) {
        // SAFETY: our own event, not waited on any more.
        let _ = unsafe { windows::Win32::Foundation::CloseHandle(self.event) };
    }
}

const PIECE_BYTES: u64 = 64 << 10;
