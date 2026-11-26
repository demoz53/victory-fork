//! Host pipelines are made by workers, not by the title thread that first
//! records a draw with them. Translating a shader and the host driver's
//! compile take from milliseconds to seconds each (structurisation of the
//! biggest compute programs ~0.4 s, some driver compiles 3.4 s), and
//! the title's first world frame wants over 1,200 new pipelines: made
//! on its recording threads, that frame took more than 30 s and the
//! title's GPU watchdog ended the run (decision 0007, "Pipelines made off
//! the title's threads").
//!
//! A first use queues the pipeline and waits for it within a budget per
//! frame, so the cheap ones (menus, one-off passes such as a colour-grading
//! bake) are there at once as before; past the budget a draw or dispatch
//! whose pipeline is not ready is dropped, as an untranslated one is.
//!
//! The default budget is half a second per title frame. A pass the title
//! runs only once and whose pipeline takes longer is lost.
//!
//! Passes over the screen (no vertex input, no depth) have two workers and a
//! budget a frame of their own ([`screen_passes`], [`screen_budget`]: 8 s, was
//! 20 s): the title fills its
//! lighting tables with such draws once, at the world's first frame, and behind
//! a world's worth of pipelines (a fresh cache) they were lost: no diffuse light
//! indoors and a white bloom (upstream VIctory #2433, fixed in #2435 by
//! pushkarreddyy, after Belobug 18e57db).
//!
//! A pipeline a worker has had for [`stuck_after`] (15 s) without coming
//! back is not waited for any more ([`stuck`]).

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock};
use std::time::{Duration, Instant};

use super::Shader;
use super::host::{GraphicsState, HostPipeline, TargetFormats};

/// What a host pipeline is made from, and the host pipeline once made.
pub(crate) struct Pending {
    shaders: Vec<Arc<Shader>>,
    pub(crate) graphics: Option<Box<GraphicsState>>,
    queue_state: AtomicU8,
    made: OnceLock<Option<HostPipeline>>,
    completion: (Mutex<()>, Condvar),
    /// What identifies it across runs (shader containers and state), and
    /// whether a draw or dispatch has requested it yet.
    key: String,
    used: AtomicBool,
    /// A pass over the screen ([`GraphicsState::screen_pass`], [`screen_passes`]).
    screen: bool,
    /// When a worker began making it ([`stuck`]).
    begun: OnceLock<Instant>,
    /// Whether [`stuck`] has named it and started a worker in place of the
    /// one making it.
    replaced: AtomicBool,
    /// The title frame (plus one) in which a dispatch's long wait for it
    /// ran out ([`wait_made`]): the other asks of that frame (a dispatch
    /// asks for its pipeline more than once) do not wait again.
    gave_up: AtomicU64,
    /// The formats its state declares for what a draw draws into (zeros
    /// without a graphics state).
    pub(crate) formats: TargetFormats,
    /// The same shaders and state for other formats, bound at a draw
    /// ([`for_formats`]): a chain from the title's pipeline, one link per
    /// set of formats met, never shortened.
    variant: OnceLock<Arc<Pending>>,
}

// SAFETY: D3D12 objects are free-threaded; the graphics state (whose input
// elements point into its own names) is only read after creation.
unsafe impl Send for Pending {}
// SAFETY: as above.
unsafe impl Sync for Pending {}

impl Pending {
    pub(crate) fn new(shaders: Vec<Arc<Shader>>, graphics: Option<Box<GraphicsState>>) -> Pending {
        let mut key = blake3::Hasher::new();
        for s in &shaders {
            key.update(s.stage.as_bytes());
            key.update(s.hash.as_bytes());
        }
        key.update(graphics.as_ref().map(|g| g.describe()).unwrap_or_default().as_bytes());
        let graphics_screen = graphics.as_ref().is_some_and(|g| g.screen_pass());
        Pending {
            formats: graphics.as_ref().map_or([0; 9], |g| g.formats()),
            variant: OnceLock::new(),
            shaders,
            graphics,
            queue_state: AtomicU8::new(NOT_QUEUED),
            made: OnceLock::new(),
            completion: (Mutex::new(()), Condvar::new()),
            key: key.finalize().to_hex()[..32].to_owned(),
            used: AtomicBool::new(false),
            screen: screen_passes() && graphics_screen,
            begun: OnceLock::new(),
            replaced: AtomicBool::new(false),
            gave_up: AtomicU64::new(0),
        }
    }

    /// Its shaders as `stage hash`, the hashes `VICTORY_SKIP_PIPELINES` takes.
    fn names(&self) -> String {
        let names: Vec<String> = self
            .shaders
            .iter()
            .map(|s| format!("{} {}", s.stage, s.hash))
            .collect();
        names.join(", ")
    }

    /// Its shaders.
    pub(crate) fn shaders(&self) -> &[Arc<Shader>] {
        &self.shaders
    }

    /// What identifies it across runs: its shader containers and state
    /// (the key of the early list and of the pipeline records).
    pub(crate) fn key(&self) -> &str {
        &self.key
    }

    fn make(&self) {
        self.made.get_or_init(|| {
            // Before the driver sees it, for an offline warm of the library
            // (`super::warm`): a record for each pipeline a run makes.
            super::warm::record(&self.key, &self.shaders, self.graphics.as_deref());
            HostPipeline::new(&self.shaders, self.graphics.as_deref())
        });
        let (lock, done) = &self.completion;
        let _held = lock.lock().unwrap_or_else(|e| e.into_inner());
        done.notify_all();
    }
}

/// Why a pipeline has no host pipeline (diagnostics): its shaders that do
/// not translate, else "refused" or "being made"; `None` if it has one.
#[cfg(feature = "diagnostics")]
pub(crate) fn missing(p: &Pending) -> Option<String> {
    match p.made.get() {
        Some(Some(_)) => None,
        // Its worker may have died making it, or its compile never ends.
        None if stuck(p) => Some(format!("stuck, its compile never came back {:?}", p.shaders)),
        None => Some("being made".into()),
        Some(None) => {
            let failed: Vec<String> = p
                .shaders
                .iter()
                .filter(|s| matches!(s.translated.get(), Some(None)))
                .map(|s| format!("{} {}", s.stage, s.hash))
                .collect();
            Some(if failed.is_empty() {
                format!("refused {:?}", p.shaders)
            } else {
                failed.join(" + ")
            })
        }
    }
}

/// Waiting allowed per title frame, over all threads.
fn wait_budget() -> Duration {
    static BUDGET: OnceLock<Duration> = OnceLock::new();
    *BUDGET.get_or_init(|| {
        let millis = std::env::var("VICTORY_PIPELINE_WAIT_MS")
            .ok()
            .and_then(|value| value.trim().parse::<u64>().ok())
            .unwrap_or(500);
        tracing::info!(millis, "host pipeline wait budget per frame");
        Duration::from_millis(millis)
    })
}

/// Whether passes over the screen get workers and a wait budget of their own
/// (`VICTORY_SCREEN_PASSES=0`: they queue with every other pipeline, as before).
fn screen_passes() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_SCREEN_PASSES").is_none_or(|v| v != "0"))
}

/// Waiting allowed per title frame for passes over the screen
/// (`VICTORY_SCREEN_PASS_WAIT_MS`, 8000 by default; it was 20 s). The title's
/// watchdog fires after 30 s without a finished frame, and a frame also
/// waits for other things; one pass that took the whole 20 s every frame
/// left the title little room. When a frame's wait runs out on a pass,
/// [`wait`] names the pipeline in the log.
fn screen_budget() -> Duration {
    static BUDGET: OnceLock<Duration> = OnceLock::new();
    *BUDGET.get_or_init(|| {
        let millis = std::env::var("VICTORY_SCREEN_PASS_WAIT_MS")
            .ok()
            .and_then(|value| value.trim().parse::<u64>().ok())
            .unwrap_or(8000);
        Duration::from_millis(millis)
    })
}

/// How long a pipeline may be in a worker's hands before it counts as
/// stuck (`VICTORY_PIPELINE_STUCK_S`, 15 by default; `0`: never, as before).
fn stuck_after() -> Option<Duration> {
    #[cfg(test)]
    return Some(Duration::from_millis(50));
    #[cfg(not(test))]
    {
        static AFTER: OnceLock<Option<Duration>> = OnceLock::new();
        *AFTER.get_or_init(|| {
            let seconds = std::env::var("VICTORY_PIPELINE_STUCK_S")
                .ok()
                .and_then(|v| v.trim().parse::<u64>().ok())
                .unwrap_or(15);
            (seconds > 0).then_some(Duration::from_secs(seconds))
        })
    }
}

/// How often a wait looks again whether its pipeline is [`stuck`].
const STUCK_POLL: Duration = Duration::from_millis(250);

/// Whether a worker began making `p` longer ago than [`stuck_after`] and
/// has not come back (DexNom/VIctory PR #37, LloydNicholson, after
/// victory-fork 36a8541): then it is not waited for any more, by [`wait`]
/// or [`now`]. Under Proton, dxil-spirv recursed without end on three of
/// the title's compute programs: the worker's stack overflowed, the thread
/// was gone, the pipeline stayed "being made", and every frame spent its
/// whole budget on it. The first time it is found stuck it is named in the
/// log and a worker of its kind is started in place of the one making it. A
/// compile that is only slow is still used once made.
fn stuck(p: &Pending) -> bool {
    let (Some(after), Some(begun)) = (stuck_after(), p.begun.get()) else {
        return false;
    };
    if p.made.get().is_some() || begun.elapsed() < after {
        return false;
    }
    if !p.replaced.swap(true, Ordering::Relaxed) {
        tracing::warn!(
            pipeline = %p.key,
            shaders = %p.names(),
            seconds = after.as_secs_f32(),
            screen = p.screen,
            "a host pipeline's compile has not come back: it is not waited for any more, and a pipeline worker is started in place of the one making it (VICTORY_SKIP_PIPELINES takes its shader hashes)"
        );
        if WORKERS_STARTED.load(Ordering::Relaxed) > 0 {
            spawn_worker(usize::from(p.screen));
        }
    }
    true
}

/// Workers for passes over the screen.
const SCREEN_WORKERS: usize = 2;

/// Pipelines queued and not made yet.
static WAITING: AtomicU64 = AtomicU64::new(0);

/// Pipelines queued and not made yet (for the frame line).
pub(crate) fn waiting() -> u64 {
    WAITING.load(Ordering::Relaxed)
}

/// The host pipeline if it is made (or is made within the frame's budget);
/// queues it on first use.
pub(crate) fn ready(p: &Arc<Pending>) -> Option<&HostPipeline> {
    // A still-pending pipeline was requested by the title too. Prewarm it
    // on the next launch, even if this run never got to use it.
    if !p.used.load(Ordering::Relaxed) && !p.used.swap(true, Ordering::Relaxed) {
        super::library::note_used(&p.key);
    }
    if p.made.get().is_none() {
        start(p, REQUESTED);
        let frame = crate::frame::current();
        match compute_wait() {
            Some(most) if p.graphics.is_none() => {
                // Asked again in a frame whose long wait for it ran out:
                // not waited for again in that frame.
                if p.gave_up.load(Ordering::Relaxed) != frame + 1 {
                    match reserve_compute(frame, most) {
                        Some(take) => wait_made(p, take, frame),
                        None => wait(p, frame),
                    }
                }
            }
            _ => wait(p, frame),
        }
    }
    p.made.get()?.as_ref()
}

/// How long a dispatch waits for its compute pipeline, whatever the frame's
/// budget ([`wait_budget`]): `VICTORY_PIPELINE_COMPUTE_WAIT` in ms (10000 by
/// default; `0`: within the frame's budget, as before). A draw still waits
/// within the budget and is dropped while its pipeline is not there.
///
/// A dropped draw is drawn again in the next frame. A dropped dispatch may
/// not be: the title runs some compute passes once, when a scene or a
/// character is set up, and what they leave is read for as long as it
/// lasts (avatarmanz issue #75: a character's skin missing for a whole
/// session on first launches of a build, right on later ones, whose
/// pipeline library had the pipeline; fxceless pull request #1312,
/// Exopus-0x65f7ad0: one dispatch of a program first met dropped after the
/// half second of budget, a 4.4 s compile, the picture wrong for minutes).
/// Bounded, not endless as #1312: the title's watchdog ends the process
/// after 30 s without a finished frame, and a compile that never comes back
/// ([`stuck`]) is not waited for at all.
fn compute_wait() -> Option<Duration> {
    #[cfg(test)]
    return Some(Duration::from_millis(200));
    #[cfg(not(test))]
    {
        static MOST: OnceLock<Option<Duration>> = OnceLock::new();
        *MOST.get_or_init(|| {
            let millis = std::env::var("VICTORY_PIPELINE_COMPUTE_WAIT")
                .ok()
                .and_then(|v| v.trim().parse::<u64>().ok())
                .unwrap_or(10_000);
            (millis > 0).then_some(Duration::from_millis(millis))
        })
    }
}

/// What one title frame may spend in [`wait_made`] over all its dispatches
/// and threads (`VICTORY_PIPELINE_COMPUTE_WAIT_FRAME_MS`, 6000): many
/// programs met for the first time in one frame must not add up to the
/// title's 30-s watchdog, beside the frame's other waits (the screen
/// passes' 8 s, the others' 0.5 s). Reserved before a wait
/// ([`ComputeBudget::reserve`]), so that threads waiting at once share it.
fn compute_frame_budget() -> Duration {
    static MOST: OnceLock<Duration> = OnceLock::new();
    *MOST.get_or_init(|| {
        let millis = std::env::var("VICTORY_PIPELINE_COMPUTE_WAIT_FRAME_MS")
            .ok()
            .and_then(|v| v.trim().parse::<u64>().ok())
            .unwrap_or(6_000);
        Duration::from_millis(millis)
    })
}

/// How long a dispatch waits for a pipeline no worker has begun yet (the
/// queue backed up): `VICTORY_PIPELINE_COMPUTE_QUEUED_MS`, 2000. [`stuck`]
/// bounds only a pipeline a worker began; once one begins it during the
/// wait, the wait may go on to its whole reservation.
fn compute_queued_wait() -> Duration {
    #[cfg(test)]
    return Duration::from_millis(100);
    #[cfg(not(test))]
    {
        static MOST: OnceLock<Duration> = OnceLock::new();
        *MOST.get_or_init(|| {
            let millis = std::env::var("VICTORY_PIPELINE_COMPUTE_QUEUED_MS")
                .ok()
                .and_then(|v| v.trim().parse::<u64>().ok())
                .unwrap_or(2_000);
            Duration::from_millis(millis)
        })
    }
}

/// A frame's budget for [`wait_made`]: the frame it belongs to and what is
/// reserved of it (by waits running now, or what finished waits took).
#[derive(Default)]
struct ComputeBudget {
    frame: u64,
    reserved: Duration,
}

impl ComputeBudget {
    /// Takes at most `most` of frame `frame`'s `budget` for one wait;
    /// `None` when nothing is left. A newer frame starts afresh; an older
    /// one (a thread still in a frame gone by) gets nothing.
    fn reserve(&mut self, frame: u64, budget: Duration, most: Duration) -> Option<Duration> {
        if frame > self.frame {
            *self = ComputeBudget { frame, reserved: Duration::ZERO };
        } else if frame < self.frame {
            return None;
        }
        let take = budget.saturating_sub(self.reserved).min(most);
        if take.is_zero() {
            return None;
        }
        self.reserved += take;
        Some(take)
    }

    /// Gives back what a wait of frame `frame` reserved (`took`) and did
    /// not use (`waited`).
    fn finish(&mut self, frame: u64, took: Duration, waited: Duration) {
        if frame == self.frame {
            self.reserved = self.reserved.saturating_sub(took.saturating_sub(waited));
        }
    }
}

static COMPUTE_BUDGET: Mutex<ComputeBudget> = Mutex::new(ComputeBudget { frame: 0, reserved: Duration::ZERO });

/// [`ComputeBudget::reserve`] of the shared budget.
fn reserve_compute(frame: u64, most: Duration) -> Option<Duration> {
    COMPUTE_BUDGET.lock().unwrap_or_else(|e| e.into_inner()).reserve(frame, compute_frame_budget(), most)
}

/// Dispatches that waited for their compute pipeline past the frame's
/// budget ([`compute_wait`]): how many waited, the longest wait (ms), and
/// how many were dropped all the same (the wait ran out or the compile was
/// stuck).
static COMPUTE_WAITED: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];

/// Waits for `p` until it is made, its compile is [`stuck`], or `most`
/// (reserved of frame `frame`'s budget) has passed; a pipeline no worker
/// has begun is waited for at most [`compute_queued_wait`] unless one
/// begins it meanwhile. What was not used goes back to the frame.
fn wait_made(p: &Pending, most: Duration, frame: u64) {
    let started = Instant::now();
    if !stuck(p) {
        let (lock, done) = &p.completion;
        let mut held = lock.lock().unwrap_or_else(|e| e.into_inner());
        while p.made.get().is_none() {
            let bound = if p.begun.get().is_some() { most } else { most.min(compute_queued_wait()) };
            let rest = bound.saturating_sub(started.elapsed());
            if rest.is_zero() {
                break;
            }
            held = done.wait_timeout(held, rest.min(STUCK_POLL)).unwrap_or_else(|e| e.into_inner()).0;
            if stuck(p) {
                break;
            }
        }
        drop(held);
    }
    let waited = started.elapsed().min(most);
    COMPUTE_BUDGET.lock().unwrap_or_else(|e| e.into_inner()).finish(frame, most, waited);
    let made = p.made.get().is_some();
    if !made {
        p.gave_up.store(frame + 1, Ordering::Relaxed);
    }
    let ms = waited.as_millis() as u64;
    let n = COMPUTE_WAITED[0].fetch_add(1, Ordering::Relaxed) + 1;
    let longest = COMPUTE_WAITED[1].fetch_max(ms, Ordering::Relaxed).max(ms);
    let dropped = COMPUTE_WAITED[2].fetch_add(u64::from(!made), Ordering::Relaxed) + u64::from(!made);
    if (!made && (dropped <= 16 || dropped.is_power_of_two())) || (made && (ms >= 500 || n.is_power_of_two())) {
        tracing::info!(
            pipeline = %p.key,
            shaders = %p.names(),
            waited_ms = ms,
            made,
            begun = p.begun.get().is_some(),
            "a dispatch waited for its compute pipeline (VICTORY_PIPELINE_COMPUTE_WAIT; 0: within the frame's budget); since the start {n} waited, the longest {longest} ms, {dropped} dropped all the same"
        );
    }
    #[cfg(feature = "diagnostics")]
    crate::copystat::add(crate::copystat::C::PipelineWaitUs, started.elapsed().as_micros() as u64);
}

/// Pipelines made again for the formats bound at a draw, since the start
/// (the frame line, [`variants_made`]).
static VARIANTS: AtomicU64 = AtomicU64::new(0);

/// Pipelines made again for the formats bound at a draw, since the start.
pub(crate) fn variants_made() -> u64 {
    VARIANTS.load(Ordering::Relaxed)
}

/// Whether a draw into views of other formats than its pipeline declares
/// gets a host pipeline for those formats ([`for_formats`]). On by
/// default; `VICTORY_TARGET_VARIANTS=0` sends such a draw to the host with
/// the title's pipeline, as before.
fn variants_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_TARGET_VARIANTS").is_none_or(|v| v != "0"))
}

/// The host pipeline for a draw with `p` into views of the formats
/// `bound`, which `p` does not declare (`host::formats_differ`): that of
/// the same shaders and state for those formats. It is a pipeline like the
/// title's own: queued at its first draw, made by a worker, waited for
/// within the frame's budget, and the draw is dropped while it is not
/// there ([`ready`]).
#[cold]
pub(crate) fn for_formats<'a>(p: &'a Arc<Pending>, bound: &TargetFormats) -> Option<&'a HostPipeline> {
    match p.graphics.as_deref() {
        Some(g) if variants_on() => ready(variant(p, g, &super::host::formats_for(&p.formats, bound))),
        _ => ready(p),
    }
}

/// The pipeline of `p`'s shaders and its state `g` for the formats
/// `bound`: found in the chain that starts at `p`, or added to it (logged
/// then: once per pipeline and set of formats).
fn variant<'a>(p: &'a Arc<Pending>, g: &GraphicsState, bound: &TargetFormats) -> &'a Arc<Pending> {
    let mut at = p;
    // (`bound` is what `host::formats_for` gives, and a state made for it
    // declares exactly it, so the walk ends at the link made for it; the
    // count is a bound all the same: past it the title's own pipeline.)
    for _ in 0..64 {
        let next = at.variant.get_or_init(|| {
            let so_far = VARIANTS.fetch_add(1, Ordering::Relaxed) + 1;
            tracing::info!(
                pipeline = %p.key,
                shaders = %p.names(),
                declared = ?p.formats,
                bound = ?bound,
                so_far,
                "a pipeline is made again for the render target formats bound at its draw (DXGI formats of the eight render targets, then depth; VICTORY_TARGET_VARIANTS=0: the draw goes out with the title's pipeline)"
            );
            let v = Pending::new(p.shaders.clone(), Some(Box::new(g.with_formats(bound))));
            // Not for the early list: no title pipeline is made with its key.
            v.used.store(true, Ordering::Relaxed);
            Arc::new(v)
        });
        if next.formats == *bound {
            return next;
        }
        at = next;
    }
    p
}

/// Makes `p` on this thread if it is not made yet (waiting for a worker
/// that is making it); whether it has a host pipeline.
pub(crate) fn now(p: &Arc<Pending>) -> bool {
    // Its worker never came back: making it here would wait for that
    // worker for good (the one-time initialiser is still its).
    if stuck(p) {
        return false;
    }
    p.make();
    p.made.get().is_some_and(|made| made.is_some())
}

/// At the title's creation of a pipeline: starts it at once if an earlier
/// run drew with it (during loading, so the world's first frames find it).
pub(crate) fn early(p: &Arc<Pending>) {
    if !p.shaders.is_empty() && super::library::wanted_early(&p.key) {
        super::library::EARLY.fetch_add(1, Ordering::Relaxed);
        start(p, EARLY);
    }
}

/// Whether the frame's budget is one for all threads together: a thread
/// that waits takes what is left of it and gives back what it did not
/// use. Counted only when a wait had ended, the budget was one per thread:
/// each of the title's five recording threads began its first wait of a
/// frame with nothing counted yet and waited the whole half second, 2.5 s
/// of waiting in every frame in which pipelines were being made (runs
/// 34-49: "2504 ms waiting for pipelines" in each such frame).
/// On by default; `VICTORY_PIPELINE_WAIT_SHARED=0` counts as before.
fn shared_budget() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_PIPELINE_WAIT_SHARED").is_none_or(|v| v != "0"))
}

/// The newest frame that has requested a wait, and its spent or
/// reserved budget. A late caller from an older frame cannot reopen it
/// or erase the reservations of a newer frame.
#[derive(Default)]
struct WaitBudget {
    frame: Option<u64>,
    spent: Duration,
}

impl WaitBudget {
    fn reserve(&mut self, frame: u64, budget: Duration, shared: bool) -> Duration {
        if self.frame.is_some_and(|current| frame < current) {
            return Duration::ZERO;
        }
        if self.frame != Some(frame) {
            self.frame = Some(frame);
            self.spent = Duration::ZERO;
        }
        let left = budget.saturating_sub(self.spent);
        if shared {
            self.spent += left;
        }
        left
    }

    fn finish(&mut self, frame: u64, reserved: Duration, elapsed: Duration, shared: bool) {
        if self.frame == Some(frame) {
            let waited = elapsed.min(reserved);
            if shared {
                self.spent = self.spent.saturating_sub(reserved - waited);
            } else {
                self.spent = self.spent.saturating_add(waited);
            }
        }
    }
}

/// Waits for `p` while frame `frame` has budget left.
fn wait(p: &Pending, frame: u64) {
    let budget = if p.screen { screen_budget() } else { wait_budget() };
    if budget.is_zero() || stuck(p) {
        return;
    }
    // The frame the budget belongs to, and what was waited of it (or is
    // taken by a thread waiting now): one for passes over the screen, one
    // for the others.
    static SPENT: [Mutex<WaitBudget>; 2] = [const {
        Mutex::new(WaitBudget {
            frame: None,
            spent: Duration::ZERO,
        })
    }; 2];
    let spent = &SPENT[usize::from(p.screen)];
    let shared = shared_budget();
    let left = spent.lock().unwrap_or_else(|e| e.into_inner()).reserve(frame, budget, shared);
    if left.is_zero() {
        return;
    }
    let started = Instant::now();
    let (lock, done) = &p.completion;
    let mut held = lock.lock().unwrap_or_else(|e| e.into_inner());
    let polls = stuck_after().is_some();
    while p.made.get().is_none() {
        let rest = left.saturating_sub(started.elapsed());
        if rest.is_zero() {
            break;
        }
        let step = if polls { rest.min(STUCK_POLL) } else { rest };
        held = done.wait_timeout(held, step).unwrap_or_else(|e| e.into_inner()).0;
        if stuck(p) {
            break;
        }
    }
    drop(held);
    let waited = started.elapsed().min(left);
    // The frame's wait for passes over the screen ran out on this one.
    if p.screen && waited >= left && p.made.get().is_none() {
        tracing::warn!(
            pipeline = %p.key,
            shaders = %p.names(),
            waited_ms = waited.as_millis() as u64,
            budget_ms = budget.as_millis() as u64,
            "a frame's wait for passes over the screen ran out on this host pipeline (VICTORY_SCREEN_PASS_WAIT_MS)"
        );
    }
    {
        let mut spent = spent.lock().unwrap_or_else(|e| e.into_inner());
        // A frame that began meanwhile has its own budget.
        spent.finish(frame, left, waited, shared);
    }
    #[cfg(feature = "diagnostics")]
    crate::copystat::add(
        crate::copystat::C::PipelineWaitUs,
        started.elapsed().as_micros() as u64,
    );
}

const NOT_QUEUED: u8 = 0;
const EARLY: u8 = 1;
const REQUESTED: u8 = 2;
const MAKING: u8 = 3;

struct Queue {
    early: VecDeque<Arc<Pending>>,
    requested: VecDeque<Arc<Pending>>,
}

impl Queue {
    const fn new() -> Self {
        Self {
            early: VecDeque::new(),
            requested: VecDeque::new(),
        }
    }

    /// Queues an unrequested pipeline or promotes one queued at creation.
    /// Returns whether it is new to the queue, or `None` if nothing changed.
    fn push(&mut self, p: &Arc<Pending>, priority: u8) -> Option<bool> {
        let prev = p.queue_state.load(Ordering::Relaxed);
        if prev >= priority {
            return None;
        }
        p.queue_state.store(priority, Ordering::Relaxed);
        if priority == REQUESTED {
            self.requested.push_back(p.clone());
        } else {
            self.early.push_back(p.clone());
        }
        Some(prev == NOT_QUEUED)
    }

    fn pop(&mut self) -> Option<Arc<Pending>> {
        let p = self.requested.pop_front().or_else(|| {
            // An early entry can have been promoted and claimed already.
            std::iter::from_fn(|| self.early.pop_front())
                .find(|p| p.queue_state.load(Ordering::Relaxed) == EARLY)
        })?;
        p.queue_state.store(MAKING, Ordering::Relaxed);
        Some(p)
    }
}

/// The queues of the workers: other pipelines, and passes over the screen.
static QUEUE: [Mutex<Queue>; 2] = [const { Mutex::new(Queue::new()) }; 2];
static AVAILABLE: [Condvar; 2] = [const { Condvar::new() }; 2];

/// Start or promote work for a draw without waiting behind the entire early
/// list. Queue changes and worker claims happen under the same lock.
fn start(p: &Arc<Pending>, priority: u8) {
    if p.queue_state.load(Ordering::Relaxed) >= priority {
        return;
    }
    if !workers() {
        if p.queue_state.swap(MAKING, Ordering::Relaxed) == NOT_QUEUED {
            p.make();
        }
        return;
    }
    let kind = usize::from(p.screen);
    let mut queue = QUEUE[kind].lock().unwrap_or_else(|e| e.into_inner());
    if let Some(new) = queue.push(p, priority) {
        if new {
            PEAK.fetch_max(WAITING.fetch_add(1, Ordering::Relaxed) + 1, Ordering::Relaxed);
        }
        AVAILABLE[kind].notify_one();
    }
}

/// The next pipeline for a worker of `kind` (0: other pipelines, 1: passes over the screen).
fn next(kind: usize) -> Arc<Pending> {
    let mut queue = QUEUE[kind].lock().unwrap_or_else(|e| e.into_inner());
    loop {
        if let Some(p) = queue.pop() {
            return p;
        }
        queue = AVAILABLE[kind].wait(queue).unwrap_or_else(|e| e.into_inner());
    }
}

/// Number of background pipeline compile workers for a machine with `cores`
/// logical processors.
fn pipeline_worker_count(cores: usize) -> usize {
    (cores / 2).clamp(2, 12)
}

/// Starts the workers (half the logical processors, 2 to 12, and
/// [`SCREEN_WORKERS`] for passes over the screen) on first use.
fn workers() -> bool {
    static RUNNING: OnceLock<bool> = OnceLock::new();
    *RUNNING.get_or_init(|| {
        let mut running = false;
        let workers =
            pipeline_worker_count(std::thread::available_parallelism().map_or(4, |n| n.get()));
        let screen = if screen_passes() { SCREEN_WORKERS } else { 0 };
        for i in 0..workers + screen {
            running |= spawn_worker(usize::from(i >= workers));
        }
        running
    })
}

/// Workers started since the start (each named by its number).
static WORKERS_STARTED: AtomicUsize = AtomicUsize::new(0);

/// Stack of each pipeline worker in MiB (`VICTORY_PIPELINE_STACK_MB`, 64 by
/// default; `0`: the default 2 MiB, as before): room for the driver's
/// shader compiler, which runs on it, recursively (DexNom/VIctory PR #37,
/// after victory-fork a240d4b: NVIDIA's compiler overflowed 2 MiB on a world
/// compute program). Reserved address space, committed only as used.
fn pipeline_stack() -> Option<usize> {
    static MB: OnceLock<usize> = OnceLock::new();
    let mb = *MB.get_or_init(|| {
        std::env::var("VICTORY_PIPELINE_STACK_MB")
            .ok()
            .and_then(|v| v.trim().parse().ok())
            .unwrap_or(64)
    });
    (mb > 0).then_some(mb << 20)
}

/// Starts one more pipeline worker of `kind` (0: other pipelines, 1: passes
/// over the screen); whether it started.
fn spawn_worker(kind: usize) -> bool {
    let i = WORKERS_STARTED.fetch_add(1, Ordering::Relaxed);
    let name = if kind == 1 {
        format!("victory-pipelines-screen-{i}")
    } else {
        format!("victory-pipelines-{i}")
    };
    let mut builder = std::thread::Builder::new().name(name);
    if let Some(stack) = pipeline_stack() {
        builder = builder.stack_size(stack);
    }
    let spawned = builder.spawn(move || {
        // Below the title's own threads (loading, recording).
        // SAFETY: a plain call on our own thread.
        unsafe {
            use windows::Win32::System::Threading::*;
            let _ = SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL);
        }
        loop {
            let p = next(kind);
            let _ = p.begun.set(Instant::now());
            hold(i, Some(p.names()), Instant::now());
            p.make();
            hold(i, None, Instant::now());
            if WAITING.fetch_sub(1, Ordering::Relaxed) == 1 {
                drained();
            } else {
                super::library::save(false);
            }
        }
    });
    if let Err(e) = &spawned {
        tracing::warn!(error = %e, "no pipeline worker");
    }
    spawned.is_ok()
}

/// How long a worker may hold one pipeline before [`note_slow`] names it.
const SLOW: Duration = Duration::from_secs(60);

/// A pipeline a worker is making: its shaders ([`Pending::names`]), since
/// when, and whether [`note_slow`] has named it.
struct Held {
    shaders: String,
    since: Instant,
    named: bool,
}

/// What each worker holds, by worker.
static HELD: Mutex<Vec<Option<Held>>> = Mutex::new(Vec::new());

/// Worker `worker` now holds the pipeline of `shaders`, or none.
fn hold(worker: usize, shaders: Option<String>, since: Instant) {
    let mut held = HELD.lock().unwrap_or_else(|e| e.into_inner());
    if held.len() <= worker {
        held.resize_with(worker + 1, || None);
    }
    held[worker] = shaders.map(|shaders| Held {
        shaders,
        since,
        named: false,
    });
}

/// The pipelines held for [`SLOW`] or longer at `now` and not named yet,
/// with the seconds held; each is named once.
fn slow(now: Instant) -> Vec<(String, u64)> {
    let mut held = HELD.lock().unwrap_or_else(|e| e.into_inner());
    held.iter_mut()
        .flatten()
        .filter(|h| !h.named && now.saturating_duration_since(h.since) >= SLOW)
        .map(|h| {
            h.named = true;
            (
                h.shaders.clone(),
                now.saturating_duration_since(h.since).as_secs(),
            )
        })
        .collect()
}

/// Names, once each, the pipelines a worker has been making for a minute
/// or more (from the frame line, every second). Some driver compiles never
/// end: on an RX 9070 XT, AMD's compiler kept a worker on
/// `compute 39f0e9aeb156a61c` for whole runs while its own heap grew past
/// 30 GB, and the frame line only said "(1 being made)".
pub(crate) fn note_slow() {
    for (shaders, seconds) in slow(Instant::now()) {
        tracing::warn!(
            shaders = %shaders,
            seconds,
            "a host pipeline is still being made: the driver's compile may never end (VICTORY_SKIP_PIPELINES takes its shader hashes)"
        );
    }
}

/// The most pipelines waiting at once since the workers last ran out.
static PEAK: AtomicU64 = AtomicU64::new(0);

/// The workers ran out of work: logged after a batch of 10 or more, and
/// the cache saved (at most every 20 s).
fn drained() {
    use super::library::{EARLY, LOADED, MADE};
    if PEAK.swap(0, Ordering::Relaxed) >= 10 {
        tracing::info!(
            made = MADE.load(Ordering::Relaxed),
            loaded = LOADED.load(Ordering::Relaxed),
            started_early = EARLY.load(Ordering::Relaxed),
            "host pipelines all made"
        );
    }
    super::library::save(false);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn old_frame_cannot_reset_newer_budget() {
        for shared in [false, true] {
            let ms = Duration::from_millis;
            let mut state = WaitBudget::default();
            let first = state.reserve(10, ms(500), shared);
            assert_eq!(first, ms(500));
            state.finish(10, first, ms(300), shared);
            let next = state.reserve(11, ms(500), shared);
            assert_eq!(next, ms(500));
            assert_eq!(state.reserve(10, ms(500), shared), Duration::ZERO);
            state.finish(10, first, ms(400), shared);
            state.finish(11, next, ms(120), shared);
            assert_eq!(state.reserve(11, ms(500), shared), ms(380));
        }
    }

    #[test]
    fn shared_reservations_do_not_overbook() {
        let ms = Duration::from_millis;
        let mut state = WaitBudget::default();
        let first = state.reserve(0, ms(500), true);
        assert_eq!(state.reserve(0, ms(500), true), Duration::ZERO);
        state.finish(0, first, ms(80), true);
        let second = state.reserve(0, ms(500), true);
        assert_eq!(second, ms(420));
        state.finish(0, second, ms(600), true);
        assert_eq!(state.reserve(0, ms(500), true), Duration::ZERO);
        assert_eq!(state.reserve(1, ms(500), true), ms(500));
    }

    /// A first use queues the pipeline and, within the budget, gets the
    /// worker's answer (here: no host pipeline for no shaders).
    #[test]
    fn first_use_is_made_by_a_worker() {
        let p = Arc::new(Pending::new(Vec::new(), None));
        assert!(ready(&p).is_none());
        assert_ne!(p.queue_state.load(Ordering::Relaxed), NOT_QUEUED);
        let started = Instant::now();
        while p.made.get().is_none() && started.elapsed() < Duration::from_secs(10) {
            std::thread::yield_now();
        }
        assert!(matches!(p.made.get(), Some(None)));
    }

    /// A draw into other formats than its pipeline declares finds the
    /// pipeline for them, which exists once per set of formats and is
    /// queued by its first draw like any other (fxceless a35c55739).
    #[test]
    fn a_pipeline_for_bound_formats_is_made_once() {
        // The title's pipeline declares one render target, format 24.
        let p = Arc::new(Pending::new(Vec::new(), Some(Box::new(GraphicsState::declaring(24)))));
        assert_eq!(p.formats, [24, 0, 0, 0, 0, 0, 0, 0, 0]);
        let g = p.graphics.as_deref().unwrap();
        let (a, b) = ([26, 0, 0, 0, 0, 0, 0, 0, 0], [28, 0, 0, 0, 0, 0, 0, 0, 0]);
        let before = variants_made();
        let first = variant(&p, g, &a);
        assert_eq!(first.formats, a);
        assert_ne!(first.key, p.key);
        assert!(Arc::ptr_eq(first, variant(&p, g, &a)));
        let second = variant(&p, g, &b);
        assert_eq!(second.formats, b);
        assert!(!Arc::ptr_eq(first, second));
        assert_ne!(second.key, first.key);
        assert!(Arc::ptr_eq(first, variant(&p, g, &a)));
        assert!(Arc::ptr_eq(second, variant(&p, g, &b)));
        assert!(variants_made() >= before + 2);
        assert_eq!(first.queue_state.load(Ordering::Relaxed), NOT_QUEUED);
        assert_eq!(p.queue_state.load(Ordering::Relaxed), NOT_QUEUED);
        // Variants stay off the early list.
        assert!(first.used.load(Ordering::Relaxed));
        assert!(!p.used.load(Ordering::Relaxed));
    }

    /// A pipeline whose worker never came back is not waited for, `now`
    /// does not block on it, and a worker is started in its worker's place
    /// (DexNom/VIctory PR #37's test).
    #[test]
    fn a_stuck_pipeline_is_not_waited_for_and_its_worker_is_replaced() {
        assert!(workers());
        let p = Pending::new(Vec::new(), None);
        // A worker began it and never returned (the one-time initialiser
        // is never run here).
        p.queue_state.store(MAKING, Ordering::Relaxed);
        let _ = p.begun.set(Instant::now());
        let p = Arc::new(p);
        assert!(!stuck(&p));
        std::thread::sleep(stuck_after().unwrap() + Duration::from_millis(20));
        let before = WORKERS_STARTED.load(Ordering::Relaxed);
        let started = Instant::now();
        assert!(ready(&p).is_none());
        assert!(!now(&p));
        assert!(started.elapsed() < Duration::from_millis(100), "waited for a stuck pipeline");
        assert_eq!(WORKERS_STARTED.load(Ordering::Relaxed), before + 1);
        // Once only.
        assert!(stuck(&p));
        assert_eq!(WORKERS_STARTED.load(Ordering::Relaxed), before + 1);
        assert!(p.made.get().is_none());
    }

    /// Threads waiting at once share a frame's compute budget: what they
    /// reserve together never passes it, what a wait did not use goes
    /// back, and a new frame starts afresh.
    #[test]
    fn compute_waits_share_the_frames_budget() {
        let ms = Duration::from_millis;
        let budget = std::sync::Arc::new(Mutex::new(ComputeBudget::default()));
        let threads: Vec<_> = (0..8)
            .map(|_| {
                let budget = budget.clone();
                std::thread::spawn(move || budget.lock().unwrap().reserve(7, ms(6000), ms(10_000)))
            })
            .collect();
        let taken: Vec<Duration> = threads.into_iter().filter_map(|t| t.join().unwrap()).collect();
        assert_eq!(taken.iter().sum::<Duration>(), ms(6000), "reserved together: {taken:?}");
        assert_eq!(taken.len(), 1, "the first takes it all (its bound is longer than the budget)");
        let mut b = budget.lock().unwrap();
        assert_eq!(b.reserve(7, ms(6000), ms(10)), None);
        // The wait used 1 s of its 6: 5 s come back, for two waits of 2 s
        // and one of what is left.
        b.finish(7, ms(6000), ms(1000));
        assert_eq!(b.reserve(7, ms(6000), ms(2000)), Some(ms(2000)));
        assert_eq!(b.reserve(7, ms(6000), ms(2000)), Some(ms(2000)));
        assert_eq!(b.reserve(7, ms(6000), ms(2000)), Some(ms(1000)));
        assert_eq!(b.reserve(7, ms(6000), ms(2000)), None);
        // A thread of a frame gone by gets nothing; a new frame all of it.
        assert_eq!(b.reserve(6, ms(6000), ms(2000)), None);
        assert_eq!(b.reserve(8, ms(6000), ms(2000)), Some(ms(2000)));
        b.finish(7, ms(2000), ms(0));
        assert_eq!(b.reserved, ms(2000), "an old frame's return is not the new frame's");
    }

    /// Several threads reserving and waiting at once never hold more than
    /// the budget between them.
    #[test]
    fn concurrent_compute_reservations_stay_inside_the_budget() {
        let ms = Duration::from_millis;
        let budget = std::sync::Arc::new(Mutex::new(ComputeBudget::default()));
        let held = std::sync::Arc::new(Mutex::new(Duration::ZERO));
        let threads: Vec<_> = (0..16)
            .map(|_| {
                let (budget, held) = (budget.clone(), held.clone());
                std::thread::spawn(move || {
                    for _ in 0..200 {
                        let Some(take) = budget.lock().unwrap().reserve(1, ms(600), ms(100)) else {
                            std::thread::yield_now();
                            continue;
                        };
                        {
                            let mut h = held.lock().unwrap();
                            *h += take;
                            assert!(*h <= ms(600), "held {h:?} past the budget");
                        }
                        std::thread::yield_now();
                        *held.lock().unwrap() -= take;
                        budget.lock().unwrap().finish(1, take, Duration::ZERO);
                    }
                })
            })
            .collect();
        for t in threads {
            t.join().unwrap();
        }
        assert_eq!(budget.lock().unwrap().reserved, Duration::ZERO);
    }

    /// A dispatch's wait that ran out is not repeated by the same frame's
    /// other asks for that pipeline.
    #[test]
    fn a_dispatch_asks_once_a_frame() {
        let q = Pending::new(Vec::new(), None);
        q.queue_state.store(MAKING, Ordering::Relaxed);
        let q = Arc::new(q);
        let started = Instant::now();
        assert!(ready(&q).is_none());
        let first = started.elapsed();
        assert!(first >= compute_queued_wait().min(compute_wait().unwrap()));
        let again = Instant::now();
        assert!(ready(&q).is_none());
        // (A frame may have begun in between: then it is waited for once more.)
        if q.gave_up.load(Ordering::Relaxed) == crate::frame::current() + 1 {
            assert!(again.elapsed() < Duration::from_millis(50), "waited again in the same frame");
        }
    }

    /// A dispatch waits for its compute pipeline past the frame's budget,
    /// and gets it once a worker has made it; bounded ([`compute_wait`]).
    #[test]
    fn a_dispatch_waits_for_its_compute_pipeline() {
        let p = Pending::new(Vec::new(), None);
        // Taken by a worker that has not finished it yet.
        p.queue_state.store(MAKING, Ordering::Relaxed);
        let p = Arc::new(p);
        let maker = {
            let p = p.clone();
            std::thread::spawn(move || {
                std::thread::sleep(Duration::from_millis(60));
                p.make();
            })
        };
        let started = Instant::now();
        let _ = ready(&p);
        assert!(p.made.get().is_some(), "dispatch dropped before its pipeline was made");
        assert!(started.elapsed() >= Duration::from_millis(50));
        maker.join().unwrap();
        // One never made is waited for at most the bound.
        let q = Pending::new(Vec::new(), None);
        q.queue_state.store(MAKING, Ordering::Relaxed);
        let q = Arc::new(q);
        let started = Instant::now();
        assert!(ready(&q).is_none());
        assert!(started.elapsed() < compute_wait().unwrap() + Duration::from_secs(2));
    }

    /// A pipeline a worker began just now is not stuck, nor is one made.
    #[test]
    fn a_pipeline_being_made_is_not_stuck_yet() {
        let p = Pending::new(Vec::new(), None);
        p.queue_state.store(MAKING, Ordering::Relaxed);
        let _ = p.begun.set(Instant::now());
        let p = Arc::new(p);
        assert!(!stuck(&p));
        p.make();
        std::thread::sleep(stuck_after().unwrap() + Duration::from_millis(20));
        assert!(!stuck(&p), "made: not stuck");
    }

    #[test]
    fn pending_pipeline_is_recorded_for_the_next_launch() {
        let mut p = Pending::new(Vec::new(), None);
        p.key = "pending-pipeline-test".to_owned();
        // Simulate a worker that has taken the job but cannot finish it yet.
        p.queue_state.store(MAKING, Ordering::Relaxed);
        let p = Arc::new(p);

        assert!(!super::super::library::wanted_early(&p.key));
        assert!(ready(&p).is_none());
        assert!(p.made.get().is_none());
        assert!(super::super::library::wanted_early(&p.key));
    }

    #[test]
    fn draws_promote_early_pipelines_without_claiming_them_twice() {
        let earlier = Arc::new(Pending::new(Vec::new(), None));
        let promoted = Arc::new(Pending::new(Vec::new(), None));
        let requested = Arc::new(Pending::new(Vec::new(), None));
        let mut queue = Queue::new();

        assert_eq!(queue.push(&earlier, EARLY), Some(true));
        assert_eq!(queue.push(&promoted, EARLY), Some(true));
        assert_eq!(queue.push(&requested, REQUESTED), Some(true));
        assert_eq!(queue.push(&promoted, REQUESTED), Some(false));
        assert_eq!(queue.push(&promoted, REQUESTED), None);

        assert!(Arc::ptr_eq(&queue.pop().unwrap(), &requested));
        assert!(Arc::ptr_eq(&queue.pop().unwrap(), &promoted));
        assert_eq!(queue.push(&promoted, REQUESTED), None);
        assert!(Arc::ptr_eq(&queue.pop().unwrap(), &earlier));
        assert!(queue.pop().is_none());
    }

    #[test]
    fn pipeline_worker_count_scales_with_cores() {
        assert_eq!(pipeline_worker_count(2), 2);
        assert_eq!(pipeline_worker_count(4), 2);
        assert_eq!(pipeline_worker_count(8), 4);
        assert_eq!(pipeline_worker_count(16), 8);
        assert_eq!(pipeline_worker_count(28), 12);
        assert_eq!(pipeline_worker_count(64), 12);
    }

    /// A worker still on one pipeline a minute later is named once, by the
    /// shader hashes `VICTORY_SKIP_PIPELINES` takes; a later one is not yet.
    #[test]
    fn a_pipeline_held_a_minute_is_named_once() {
        // Above any worker's index (2 to 8 workers).
        let (old, young) = (60, 61);
        let now = Instant::now();
        hold(old, Some("compute 39f0e9aeb156a61c".to_owned()), now);
        hold(
            young,
            Some("pixel 00000000000000a1".to_owned()),
            now + Duration::from_secs(30),
        );
        let named = |at| -> Vec<String> {
            slow(at)
                .into_iter()
                .map(|(shaders, _)| shaders)
                .filter(|s| s == "compute 39f0e9aeb156a61c" || s == "pixel 00000000000000a1")
                .collect()
        };
        assert_eq!(named(now + SLOW), ["compute 39f0e9aeb156a61c"]);
        assert!(named(now + SLOW + Duration::from_secs(5)).is_empty());
        hold(old, None, now);
        hold(young, None, now);
    }
}
