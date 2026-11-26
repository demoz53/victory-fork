//! The title's display planes and what is in them when they are shown.
//!
//! The title keeps a small ring of textures it presents in turn (two of
//! 3840x2160 in the Series X profile) and draws each frame's final picture
//! into the one it is about to show. In the open world, on this layer, it
//! was seen to present a plane it had not drawn into since that plane was
//! last shown: the window then went back to a picture up to 13 s old, and
//! on to a new one at the next present ("flipping between two states":
//! window shots 0.4 s apart were pixel for pixel the same old picture every
//! second or third shot, with new pictures between them).
//!
//! `VICTORY_PRESENT_TRACE=1` (log only): one line a present: which plane,
//! whether a submission drew, copied or stored into it since it was last
//! shown (`new`), or it is the plane shown before (`repeat`), or another
//! plane was written after it (`BACK`: the picture goes back in time), and
//! a count of each every 32 presents.
//!
//! `VICTORY_PRESENT_NEWEST=1` (off by default): a present that would go
//! back shows the plane written last instead, so the window never shows an
//! older picture than the one before. Nothing of the title's is changed:
//! only which of its own finished pictures our window copies.
//!
//! Writes are counted when a list is submitted, not when it is recorded:
//! the title records the next frame's lists while it presents this one.

use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

/// Whether `VICTORY_PRESENT_TRACE` is set.
pub(crate) fn trace_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| crate::host::env_flag("VICTORY_PRESENT_TRACE"))
}

/// Whether `VICTORY_PRESENT_NEWEST` is set.
pub(crate) fn newest_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        let on = crate::host::env_flag("VICTORY_PRESENT_NEWEST");
        if on {
            tracing::info!(
                "VICTORY_PRESENT_NEWEST=1: a present of a plane not written since it was last shown shows the plane written last"
            );
        }
        on
    })
}

/// Whether planes are followed at all.
fn on() -> bool {
    trace_on() || newest_on()
}

/// Planes followed at most: a title has two or three; more than this and
/// the oldest shown is forgotten.
const MOST: usize = 8;

/// What a present does to the picture in the window.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Kind {
    /// The plane was written since it was last shown (or is shown for the
    /// first time).
    New,
    /// Not written since it was last shown, and nothing newer is known:
    /// the same picture again.
    Repeat,
    /// Not written since it was last shown, and another plane like it was
    /// written after it: an older picture than one the title has finished.
    Back,
}

/// One display plane: `T` is what holds its host texture.
struct Plane<T> {
    /// The title's resource.
    resource: usize,
    /// Its host copy when it was first shown.
    copy: T,
    /// Width, height and format: planes are only compared with their like.
    like: (u64, u32, i32),
    /// The number of the last submission that wrote it (0: none yet).
    written: u64,
    /// `written` when it was last shown.
    shown_written: u64,
    /// The frame it was last shown as.
    shown_frame: u64,
}

/// The planes shown so far and the writes submitted to them.
struct Planes<T> {
    planes: Vec<Plane<T>>,
    /// Submissions that wrote a plane so far.
    writes: u64,
    /// Presents of each [`Kind`] since the last summary, and substitutions.
    counts: [u32; 4],
}

impl<T: Clone> Planes<T> {
    const fn new() -> Self {
        Self { planes: Vec::new(), writes: 0, counts: [0; 4] }
    }

    /// A submission wrote `resources`.
    fn written(&mut self, resources: &[usize]) {
        if !self.planes.iter().any(|p| resources.contains(&p.resource)) {
            return;
        }
        self.writes += 1;
        for plane in &mut self.planes {
            if resources.contains(&plane.resource) {
                plane.written = self.writes;
            }
        }
    }

    /// The plane `resource` is presented as `frame`: what that does to the
    /// picture, and with `newest` the copy of the plane to show instead
    /// when it would go back. `make` gives a plane not seen before its
    /// host copy and likeness; none: the plane is not followed.
    fn present(
        &mut self,
        frame: u64,
        resource: usize,
        newest: bool,
        make: impl FnOnce() -> Option<(T, (u64, u32, i32))>,
        same: impl Fn(&T) -> bool,
    ) -> Option<(Kind, Option<(usize, T)>)> {
        let at = match self.planes.iter().position(|p| p.resource == resource) {
            // The title made another texture at the address of one it let go.
            Some(at) if !same(&self.planes[at].copy) => {
                let (copy, like) = make()?;
                self.planes[at] = Plane { resource, copy, like, written: 0, shown_written: 0, shown_frame: 0 };
                at
            }
            Some(at) => at,
            None => {
                let (copy, like) = make()?;
                if self.planes.len() >= MOST
                    && let Some(oldest) = (0..self.planes.len()).min_by_key(|&i| self.planes[i].shown_frame)
                {
                    self.planes.remove(oldest);
                }
                self.planes.push(Plane { resource, copy, like, written: 0, shown_written: 0, shown_frame: 0 });
                self.planes.len() - 1
            }
        };
        let plane = &self.planes[at];
        let first = plane.shown_frame == 0;
        let newer = self
            .planes
            .iter()
            .enumerate()
            .filter(|&(i, p)| i != at && p.like == plane.like && p.written > plane.written && p.shown_frame != 0)
            .max_by_key(|(_, p)| p.written)
            .map(|(i, _)| i);
        let kind = if first || plane.written != plane.shown_written {
            Kind::New
        } else if newer.is_some() {
            Kind::Back
        } else {
            Kind::Repeat
        };
        self.counts[kind as usize] += 1;
        let instead = match (kind, newer) {
            (Kind::Back, Some(i)) if newest => {
                self.counts[3] += 1;
                Some((self.planes[i].resource, self.planes[i].copy.clone()))
            }
            _ => None,
        };
        let plane = &mut self.planes[at];
        plane.shown_written = plane.written;
        plane.shown_frame = frame;
        Some((kind, instead))
    }
}

/// The title resources of the planes followed, for [`is_plane`] without a
/// lock (0: none).
static KEYS: [AtomicUsize; MOST] = [const { AtomicUsize::new(0) }; MOST];

/// Presents so far, for the summary.
static PRESENTS: AtomicU64 = AtomicU64::new(0);

static PLANES: Mutex<Planes<std::sync::Arc<crate::resource::HostTexture>>> = Mutex::new(Planes::new());

/// Whether title resource `resource` is a display plane shown before:
/// lists note their writes of such ([`crate::host::HostList::plane_written`]).
/// Always false with both switches off.
pub(crate) fn is_plane(resource: usize) -> bool {
    on() && resource != 0 && KEYS.iter().any(|k| k.load(Ordering::Relaxed) == resource)
}

/// A submission's lists wrote the planes `resources` (render target,
/// unordered access or copy destination).
pub(crate) fn written(resources: &[usize]) {
    if resources.is_empty() {
        return;
    }
    PLANES.lock().unwrap_or_else(|e| e.into_inner()).written(resources);
}

/// The title presents plane `resource` as `frame`: traced
/// (`VICTORY_PRESENT_TRACE`), and with `VICTORY_PRESENT_NEWEST` the host
/// copy of the plane to show instead, when this one would bring an older
/// picture back.
pub(crate) fn present(
    frame: u64,
    resource: crate::object::This,
) -> Option<std::sync::Arc<crate::resource::HostTexture>> {
    if !on() || resource.is_null() {
        return None;
    }
    let copy = crate::resource::host_copy(resource)?;
    let mut planes = PLANES.lock().unwrap_or_else(|e| e.into_inner());
    let (kind, instead) = planes.present(
        frame,
        resource as usize,
        newest_on(),
        || {
            // SAFETY: a live resource.
            let d = unsafe { copy.texture.GetDesc() };
            Some((copy.clone(), (d.Width, d.Height, d.Format.0)))
        },
        |held| std::sync::Arc::ptr_eq(held, &copy),
    )?;
    for (key, plane) in KEYS.iter().zip(planes.planes.iter().map(|p| p.resource).chain(std::iter::repeat(0))) {
        key.store(plane, Ordering::Relaxed);
    }
    let presents = PRESENTS.fetch_add(1, Ordering::Relaxed) + 1;
    if trace_on() {
        let others: Vec<String> = planes
            .planes
            .iter()
            .map(|p| format!("{:#x} written {} shown {} (frame {})", p.resource, p.written, p.shown_written, p.shown_frame))
            .collect();
        tracing::info!(
            frame,
            plane = format_args!("{:#x}", resource as usize),
            kind = ?kind,
            shown_instead = ?instead.as_ref().map(|(r, _)| format!("{r:#x}")),
            planes = ?others,
            "present trace (VICTORY_PRESENT_TRACE)"
        );
    }
    if presents.is_multiple_of(32) {
        let [new, repeat, back, replaced] = std::mem::take(&mut planes.counts);
        if trace_on() || back != 0 {
            tracing::info!(
                new,
                repeat,
                back,
                replaced,
                planes = planes.planes.len(),
                "the last 32 presents: planes written since they were last shown, shown again unwritten, and going back to an older picture than another plane holds (VICTORY_PRESENT_NEWEST=1 shows the newest instead: replaced)"
            );
        }
    }
    instead.map(|(_, copy)| copy)
}

#[cfg(test)]
mod tests {
    use super::*;

    const LIKE: (u64, u32, i32) = (3840, 2160, 28);

    fn show(p: &mut Planes<u8>, frame: u64, resource: usize, newest: bool) -> (Kind, Option<usize>) {
        let (kind, instead) = p
            .present(frame, resource, newest, || Some((resource as u8, LIKE)), |_| true)
            .expect("followed");
        (kind, instead.map(|(r, _)| r))
    }

    #[test]
    fn planes_drawn_in_turn_are_new_each_time() {
        let mut p = Planes::<u8>::new();
        for frame in 1..=6u64 {
            let resource = if frame % 2 == 1 { 0xa } else { 0xb };
            p.written(&[resource]);
            assert_eq!(show(&mut p, frame, resource, true), (Kind::New, None));
        }
    }

    #[test]
    fn a_plane_shown_again_unwritten_is_a_repeat_until_another_is_newer() {
        let mut p = Planes::<u8>::new();
        p.written(&[0xa]);
        assert_eq!(show(&mut p, 1, 0xa, false).0, Kind::New);
        // Nothing drawn: the same picture, and nothing newer to show.
        assert_eq!(show(&mut p, 2, 0xa, true), (Kind::Repeat, None));
        // The other plane is shown first with nothing known of it.
        assert_eq!(show(&mut p, 3, 0xb, true), (Kind::New, None));
        // The title draws into B only and presents A and B in turn: A goes
        // back to an old picture each time; traced, and with the switch B
        // is shown.
        p.written(&[0xb]);
        assert_eq!(show(&mut p, 4, 0xa, false), (Kind::Back, None));
        assert_eq!(show(&mut p, 5, 0xb, true), (Kind::New, None));
        assert_eq!(show(&mut p, 6, 0xa, true), (Kind::Back, Some(0xb)));
        // Neither is drawn: A still goes back, B repeats.
        assert_eq!(show(&mut p, 7, 0xb, true), (Kind::Repeat, None));
        assert_eq!(show(&mut p, 8, 0xa, true), (Kind::Back, Some(0xb)));
        // Once A is drawn again it is shown as it is.
        p.written(&[0xa]);
        assert_eq!(show(&mut p, 9, 0xa, true), (Kind::New, None));
        assert_eq!(show(&mut p, 10, 0xb, true), (Kind::Back, Some(0xa)));
    }

    #[test]
    fn planes_of_another_size_are_not_shown_instead() {
        let mut p = Planes::<u8>::new();
        let small = (1920, 1080, 28);
        p.present(1, 0xa, true, || Some((0, LIKE)), |_| true);
        p.present(2, 0xc, true, || Some((0, small)), |_| true);
        p.written(&[0xc]);
        p.present(3, 0xc, true, || None, |_| true);
        let (kind, instead) = p.present(4, 0xa, true, || None, |_| true).expect("followed");
        assert_eq!((kind, instead.map(|(r, _)| r)), (Kind::Repeat, None));
    }

    #[test]
    fn a_plane_made_again_at_the_same_address_starts_anew() {
        let mut p = Planes::<u8>::new();
        p.written(&[0xa]);
        show(&mut p, 1, 0xa, true);
        show(&mut p, 2, 0xb, true);
        p.written(&[0xb]);
        show(&mut p, 3, 0xb, true);
        // Another host copy behind 0xa: its first showing is new, not a
        // going back.
        let (kind, instead) = p.present(4, 0xa, true, || Some((9, LIKE)), |_| false).expect("followed");
        assert_eq!((kind, instead.map(|(r, _)| r)), (Kind::New, None));
    }

    #[test]
    fn only_as_many_planes_as_most_are_followed() {
        let mut p = Planes::<u8>::new();
        for i in 0..MOST as u64 + 3 {
            show(&mut p, i + 1, 0x100 + i as usize, false);
        }
        assert_eq!(p.planes.len(), MOST);
        assert!(p.planes.iter().all(|pl| pl.resource >= 0x103));
    }
}
