//! Keeps the card's clocks up between submissions (`VICTORY_CARD_AWAKE=1`; off
//! by default). The card stands idle a part of every frame, some fifty times
//! for a millisecond or two each, while the CPU prepares the next submission
//! (run all15b-on1, the boat scene: nothing of ours queued for 67 of a frame's
//! 252 ms, in 49 submissions). The driver takes that for a light load and
//! lowers the clocks, and what comes next starts slow: in the fork this is
//! ported from, a captured frame whose lists take 28.6 ms sent in one go took
//! 37.8 ms with 1.2 ms between its submissions, and 28.6 ms again with this.
//!
//! One group of 64 threads counts on a compute queue of its own, over and
//! over: it takes one or two of the card's few thousand wave slots, and the
//! card is never idle. Nothing is sent once no submission has gone out for
//! a quarter of a second (a title that stands still; not a menu, which goes on
//! drawing). The title's lists, their order and what they draw are untouched.
//!
//! Ported from avatarmanz/VIctory-7900XTX (commits 5f86fa302 and 040aa272c,
//! GPL-3.0-or-later, as this), without its replay options.
//!
//! MEASURED ON THIS PROJECT'S RTX 4060 Ti: IT COSTS. The test
//! `paced_submissions_with_and_without_the_card_kept_awake` (2026-11-25,
//! while a game test ran beside it, so not on a quiet card), the median
//! submission of 60, on a queue of its own:
//!
//! | work     | in one go | 1.5 ms apart | awake, 400,000 steps | awake, 4,000 steps |
//! |----------|-----------|--------------|----------------------|--------------------|
//! | 0.13 ms  | 0.13      | 0.20         | 1.50 apart, 2.19 go  | 0.34 apart, 0.41 go |
//! | 1.56 ms  | 1.56      | 1.61         | 5.65 apart, 5.51 go  | 2.23 apart, 2.64 go |
//!
//! Standing idle for 1.5 ms between submissions cost this card 0.05-0.07 ms
//! a submission whatever the work (a delay, not lower clocks; the other
//! process may have kept the clocks up). The counting pass did not take "one
//! or two wave slots" beside the work: the card ran the two queues in turns,
//! and work of 1.56 ms took 5.5 ms. So on this card the switch is expected
//! to slow the frame, as it stands, and the other fork's own runs on this PC
//! had it on. It stays off; `VICTORY_CARD_AWAKE_STEPS` and the measurement
//! are there to look again on a quiet card.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Duration;

use windows::Win32::Foundation::CloseHandle;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::System::Threading::{CreateEventW, INFINITE, WaitForSingleObject};
use windows::core::Interface;

/// The counting pass; `STEPS` stands for [`steps`].
const COUNT: &str = r#"
#define RS "UAV(u0)"
RWByteAddressBuffer sum : register(u0);
[RootSignature(RS)]
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint x = id.x;
    [loop] for (uint i = 0; i < STEPSu; i++) x = x * 1664525u + 1013904223u;
    sum.Store(id.x * 4u, x);
}
"#;

/// The other fork's pass: about 2 ms at full clocks, 500 sends a second.
const STEPS: u32 = 400_000;

/// `VICTORY_CARD_AWAKE_STEPS` (1,000 to 4,000,000; else [`STEPS`]).
fn steps_of(value: Option<&str>) -> u32 {
    value.and_then(|v| v.trim().parse().ok()).filter(|n| (1_000..=4_000_000).contains(n)).unwrap_or(STEPS)
}

/// How long a counting pass counts (`VICTORY_CARD_AWAKE_STEPS`; 400,000 as
/// in the other fork). A submission that arrives while a pass counts may
/// have to wait for the whole pass where the card runs a second queue's work
/// in turns with the first and not beside it; the measurement
/// `paced_submissions_with_and_without_the_card_kept_awake` says what this
/// card does. A shorter pass holds a submission up for less; whether it
/// still keeps the clocks up is for the same measurement on a quiet card.
fn steps() -> u32 {
    static STEPS_ASKED: OnceLock<u32> = OnceLock::new();
    *STEPS_ASKED.get_or_init(|| steps_of(std::env::var("VICTORY_CARD_AWAKE_STEPS").ok().as_deref()))
}

/// Quiet this long after the last submission.
const QUIET_MS: u64 = 250;

fn now_ms() -> u64 {
    static START: OnceLock<std::time::Instant> = OnceLock::new();
    START.get_or_init(std::time::Instant::now).elapsed().as_millis() as u64
}

/// When a submission was last sent ([`now_ms`] plus one; 0: never).
static LAST: AtomicU64 = AtomicU64::new(0);

/// Whether the card is to be kept busy at `now`, the last submission having
/// gone out at `last` ([`LAST`]).
fn wanted_at(mode: Mode, last: u64, now: u64) -> bool {
    last != 0 && now + 1 < last + QUIET_MS && (mode != Mode::ByTurns || now / TURN_MS % 2 == 1)
}

fn wanted() -> bool {
    wanted_at(mode(), LAST.load(Ordering::Relaxed), now_ms())
}

/// With `VICTORY_CARD_AWAKE=turns`: this long on, this long off.
const TURN_MS: u64 = 20_000;

#[derive(Clone, Copy, PartialEq, Debug)]
enum Mode {
    Off,
    On,
    /// On and off by turns, to compare the two in one run (the log says when).
    ByTurns,
}

fn mode_of(value: Option<&str>) -> Mode {
    match value {
        Some("1") => Mode::On,
        Some("turns") => Mode::ByTurns,
        _ => Mode::Off,
    }
}

fn mode() -> Mode {
    static MODE: OnceLock<Mode> = OnceLock::new();
    *MODE.get_or_init(|| mode_of(std::env::var("VICTORY_CARD_AWAKE").ok().as_deref()))
}

/// A submission of the title's goes to `device`'s drawing queue.
pub(super) fn sent(device: &ID3D12Device) {
    if mode() == Mode::Off {
        return;
    }
    let now = now_ms();
    let before = LAST.swap(now + 1, Ordering::Relaxed);
    if mode() == Mode::ByTurns && before.saturating_sub(1) / TURN_MS != now / TURN_MS {
        tracing::info!(on = now / TURN_MS % 2 == 1, "the card is kept awake, by turns (VICTORY_CARD_AWAKE=turns)");
    }
    static STARTED: std::sync::Once = std::sync::Once::new();
    STARTED.call_once(|| {
        tracing::info!("the card's clocks are kept up between submissions (VICTORY_CARD_AWAKE)");
        // Runs as long as the process does.
        std::mem::forget(start(device, steps(), wanted));
    });
}

/// The thread that keeps the card busy; it ends when this is dropped.
struct Awake {
    stop: std::sync::Arc<AtomicBool>,
    /// The passes sent to the card so far.
    #[cfg(test)]
    passes: std::sync::Arc<AtomicU64>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl Drop for Awake {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

/// Keeps the card of `device` busy while `wanted`, with passes of `steps`.
fn start(device: &ID3D12Device, steps: u32, wanted: impl Fn() -> bool + Send + 'static) -> Option<Awake> {
    let stop = std::sync::Arc::new(AtomicBool::new(false));
    let passes = std::sync::Arc::new(AtomicU64::new(0));
    let (device, ended, sent) = (device.clone(), stop.clone(), passes.clone());
    let thread = std::thread::Builder::new()
        .name("victory-card-awake".into())
        .spawn(move || {
            if let Err(e) = run(&device, steps, &wanted, &ended, &sent) {
                tracing::warn!(error = %e, "the card is not kept awake");
            }
        })
        .ok()?;
    Some(Awake {
        stop,
        #[cfg(test)]
        passes,
        thread: Some(thread),
    })
}

fn run(
    device: &ID3D12Device,
    steps: u32,
    wanted: &dyn Fn() -> bool,
    stop: &AtomicBool,
    passes: &AtomicU64,
) -> windows::core::Result<()> {
    // sets: own list (bindings on a list of ours that records nothing of the
    // title's: no draw state's memory of bindings is about it).
    let source = COUNT.replace("STEPS", &steps.to_string());
    let code = super::compile(&source, windows::core::s!("main"), windows::core::s!("cs_5_1"))?;
    // SAFETY: plain object creation; the code lives through the calls, the
    // root signature is inside it. Two closed lists, sent in turn, whose
    // objects live until what was sent last has run.
    unsafe {
        let root: ID3D12RootSignature = device.CreateRootSignature(0, &code)?;
        let pso: ID3D12PipelineState = device.CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
            pRootSignature: std::mem::transmute_copy(&root),
            CS: D3D12_SHADER_BYTECODE {
                pShaderBytecode: code.as_ptr().cast(),
                BytecodeLength: code.len(),
            },
            ..Default::default()
        })?;
        let queue: ID3D12CommandQueue = device.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
            Type: D3D12_COMMAND_LIST_TYPE_COMPUTE,
            Priority: D3D12_COMMAND_QUEUE_PRIORITY_NORMAL.0,
            ..Default::default()
        })?;
        let mut sum: Option<ID3D12Resource> = None;
        device.CreateCommittedResource(
            &D3D12_HEAP_PROPERTIES {
                Type: D3D12_HEAP_TYPE_DEFAULT,
                ..Default::default()
            },
            D3D12_HEAP_FLAG_NONE,
            &D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                Width: 256,
                Height: 1,
                DepthOrArraySize: 1,
                MipLevels: 1,
                SampleDesc: windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                ..Default::default()
            },
            D3D12_RESOURCE_STATE_COMMON,
            None,
            &mut sum,
        )?;
        let sum = sum.ok_or_else(windows::core::Error::empty)?;
        let allocator: ID3D12CommandAllocator = device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE)?;
        let mut lists = Vec::new();
        for _ in 0..2 {
            let list: ID3D12GraphicsCommandList =
                device.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COMPUTE, &allocator, &pso)?;
            list.SetComputeRootSignature(&root);
            list.SetComputeRootUnorderedAccessView(0, sum.GetGPUVirtualAddress());
            list.Dispatch(1, 1, 1);
            list.Close()?;
            lists.push(Some(list.cast::<ID3D12CommandList>()?));
        }
        let fence: ID3D12Fence = device.CreateFence(0, D3D12_FENCE_FLAG_NONE)?;
        let event = CreateEventW(None, false, false, None)?;
        let mut value = 0u64;
        let mut result = Ok(());
        while !stop.load(Ordering::Relaxed) && result.is_ok() {
            if !wanted() {
                std::thread::sleep(Duration::from_millis(2));
                continue;
            }
            // Never more than two on the way.
            if value >= 2 && fence.GetCompletedValue() + 2 <= value {
                result = fence.SetEventOnCompletion(value - 1, event);
                if result.is_ok() {
                    WaitForSingleObject(event, 1000);
                    result = device.GetDeviceRemovedReason();
                }
                continue;
            }
            queue.ExecuteCommandLists(&lists[(value % 2) as usize..][..1]);
            value += 1;
            passes.store(value, Ordering::Relaxed);
            result = queue.Signal(&fence, value);
        }
        if result.is_ok() && fence.GetCompletedValue() < value && fence.SetEventOnCompletion(value, event).is_ok() {
            WaitForSingleObject(event, INFINITE);
        }
        let _ = CloseHandle(event);
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_switch_is_off_unless_asked_for() {
        assert_eq!(mode_of(None), Mode::Off);
        assert_eq!(mode_of(Some("0")), Mode::Off);
        assert_eq!(mode_of(Some("1")), Mode::On);
        assert_eq!(mode_of(Some("turns")), Mode::ByTurns);
    }

    #[test]
    fn a_pass_counts_as_long_as_the_other_forks_unless_a_length_in_bounds_is_asked() {
        assert_eq!(steps_of(None), STEPS);
        assert_eq!(steps_of(Some("4000")), 4_000);
        assert_eq!(steps_of(Some("12")), STEPS);
        assert_eq!(steps_of(Some("many")), STEPS);
    }

    #[test]
    fn the_card_is_left_alone_before_the_first_submission_and_after_a_quiet_while() {
        assert!(!wanted_at(Mode::On, 0, 5));
        assert!(wanted_at(Mode::On, 1001, 1000));
        assert!(wanted_at(Mode::On, 1001, 1000 + QUIET_MS - 2));
        assert!(!wanted_at(Mode::On, 1001, 1000 + QUIET_MS));
        // By turns: off in the first 20 s, on in the next.
        assert!(!wanted_at(Mode::ByTurns, 1001, 1000));
        assert!(wanted_at(Mode::ByTurns, TURN_MS + 1001, TURN_MS + 1000));
    }

    /// The counting passes run on the card beside a list of the drawing
    /// queue's kind, which still does its work; the thread ends when it is
    /// let go, with the device as it was.
    #[test]
    fn the_card_counts_while_wanted_and_other_work_still_runs() {
        let Some(host) = crate::host::host() else {
            return; // no host GPU here
        };
        let awake = start(&host.device, STEPS, || true).expect("a thread");
        let (seen, back) = host.shared_buffer(4096).unwrap();
        let (from, data) = host.shared_buffer(4096).unwrap();
        // SAFETY: mappings of 4096 bytes; a list, a queue and a fence of the
        // test's own, and nothing is read before the copy has run.
        let read = unsafe {
            for i in 0..4096 {
                data.add(i).write((i * 7 + i / 251) as u8);
            }
            let device = &host.device;
            let queue: ID3D12CommandQueue = device
                .CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
                    Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
                    ..Default::default()
                })
                .unwrap();
            let allocator: ID3D12CommandAllocator = device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).unwrap();
            let list: ID3D12GraphicsCommandList =
                device.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &allocator, None).unwrap();
            list.CopyBufferRegion(&seen, 0, &from, 0, 4096);
            list.Close().unwrap();
            queue.ExecuteCommandLists(&[Some(list.cast().unwrap())]);
            let fence: ID3D12Fence = device.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
            queue.Signal(&fence, 1).unwrap();
            // No event: returns when the fence is there.
            fence.SetEventOnCompletion(1, windows::Win32::Foundation::HANDLE::default()).unwrap();
            std::slice::from_raw_parts(back, 4096).to_vec()
        };
        assert!(read.iter().enumerate().all(|(i, &b)| b == (i * 7 + i / 251) as u8));
        // Some passes have gone to the card by now, or go within a second.
        let started = std::time::Instant::now();
        while awake.passes.load(Ordering::Relaxed) < 3 && started.elapsed() < Duration::from_secs(5) {
            std::thread::sleep(Duration::from_millis(5));
        }
        assert!(awake.passes.load(Ordering::Relaxed) >= 3, "no counting passes were sent");
        drop(awake);
        // SAFETY: a plain query of the device.
        assert!(unsafe { host.device.GetDeviceRemovedReason() }.is_ok());
    }

    /// Compute work over the whole card; `STEPS` stands for how long each of
    /// its million threads counts.
    const WORK: &str = r#"
#define RS "UAV(u0)"
RWByteAddressBuffer sums : register(u0);
[RootSignature(RS)]
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint x = id.x;
    [loop] for (uint i = 0; i < STEPSu; i++) x = x * 1664525u + 1013904223u;
    sums.Store(id.x * 4u, x);
}
"#;

    /// Sends the same work (`work` steps a thread) `n` times on a queue of
    /// the test's own, each time waiting for the card and then standing idle
    /// for `gap`: the milliseconds the card took for all of them, and for
    /// the slowest half's fastest (the median).
    fn paced(host: &crate::host::Host, work: u32, n: usize, gap: Duration) -> (f64, f64) {
        const THREADS: u64 = 64 * 16_384;
        let source = WORK.replace("STEPS", &work.to_string());
        let code = crate::host::compile(&source, windows::core::s!("main"), windows::core::s!("cs_5_1")).unwrap();
        let sums = host.uav_buffer(THREADS * 4).unwrap();
        let device = &host.device;
        // SAFETY: objects of the test's own; a closed list is sent again only
        // after the card has run it.
        unsafe {
            let root: ID3D12RootSignature = device.CreateRootSignature(0, &code).unwrap();
            let pso: ID3D12PipelineState = device
                .CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                    pRootSignature: std::mem::transmute_copy(&root),
                    CS: D3D12_SHADER_BYTECODE {
                        pShaderBytecode: code.as_ptr().cast(),
                        BytecodeLength: code.len(),
                    },
                    ..Default::default()
                })
                .unwrap();
            let queue: ID3D12CommandQueue = device
                .CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
                    Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
                    ..Default::default()
                })
                .unwrap();
            let allocator: ID3D12CommandAllocator = device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).unwrap();
            let list: ID3D12GraphicsCommandList =
                device.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &allocator, &pso).unwrap();
            list.SetComputeRootSignature(&root);
            list.SetComputeRootUnorderedAccessView(0, sums.GetGPUVirtualAddress());
            list.Dispatch(16_384, 1, 1);
            list.Close().unwrap();
            let lists = [Some(list.cast::<ID3D12CommandList>().unwrap())];
            let fence: ID3D12Fence = device.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
            let mut took = Vec::with_capacity(n);
            for i in 0..n as u64 {
                let sent = std::time::Instant::now();
                queue.ExecuteCommandLists(&lists);
                queue.Signal(&fence, i + 1).unwrap();
                // No event: returns when the fence is there.
                fence.SetEventOnCompletion(i + 1, windows::Win32::Foundation::HANDLE::default()).unwrap();
                took.push(sent.elapsed().as_secs_f64() * 1e3);
                // Idle as the card is while a submission is prepared (the CPU is not).
                let idle = std::time::Instant::now();
                while idle.elapsed() < gap {
                    std::hint::spin_loop();
                }
            }
            let all = took.iter().sum();
            took.sort_by(f64::total_cmp);
            (all, took[n / 2])
        }
    }

    /// A measurement, not a check: what the same work costs this card sent
    /// in one go and sent with the card idle for 1.5 ms between submissions
    /// (as between the title's), for light work (256 steps a thread) and
    /// heavier (8,192): a card that lowers its clocks takes longer apart
    /// than in one go, and by more for the heavier work (a fixed delay is
    /// the same for both). Then the same with the card kept awake, by the
    /// other fork's pass and by one a hundredth as long: these say whether
    /// keeping it awake brings the time back, and what a pass costs a
    /// submission that has to wait for it.
    ///
    /// On a quiet card, in a process of its own:
    /// `cargo test -p victory-gpu --lib paced_submissions -- --ignored --nocapture`
    #[test]
    #[ignore = "a measurement: wants a quiet card and a process of its own"]
    fn paced_submissions_with_and_without_the_card_kept_awake() {
        let Some(host) = crate::host::host() else {
            return; // no host GPU here
        };
        const N: usize = 60;
        let gap = Duration::from_micros(1500);
        for work in [256, 8_192] {
            paced(host, work, 20, Duration::ZERO); // the pipeline is made, the card is up
            let mut rows = vec![
                ("in one go".to_owned(), paced(host, work, N, Duration::ZERO)),
                ("1.5 ms apart".to_owned(), paced(host, work, N, gap)),
            ];
            for pass in [STEPS, STEPS / 100] {
                let awake = start(&host.device, pass, || true).expect("a thread");
                std::thread::sleep(Duration::from_millis(50));
                rows.push((format!("1.5 ms apart, awake by passes of {pass} steps"), paced(host, work, N, gap)));
                rows.push((format!("in one go, awake by passes of {pass} steps"), paced(host, work, N, Duration::ZERO)));
                drop(awake);
            }
            for (what, (all, median)) in rows {
                eprintln!("work of {work} steps, {N} submissions {what}: {all:.1} ms on the card, {median:.2} ms the median one");
            }
        }
        // SAFETY: a plain query of the device.
        assert!(unsafe { host.device.GetDeviceRemovedReason() }.is_ok());
    }
}
