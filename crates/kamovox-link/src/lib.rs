//! Shared memory between the KamoVox APO (loaded by `audiodg.exe` on a
//! Windows microphone) and the KamoVox app.
//!
//! The APO copies the raw microphone into the `mic` ring; the app processes it
//! and writes the transformed voice into the `voice` ring, which the APO hands
//! to every program recording from that microphone. Each ring has one writer
//! and one reader, and both use monotonic sample counters.

use std::ptr::null_mut;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

use windows_sys::Win32::Foundation::{CloseHandle, LocalFree, GetLastError, ERROR_ALREADY_EXISTS, HANDLE, INVALID_HANDLE_VALUE};
use windows_sys::Win32::Security::Authorization::ConvertStringSecurityDescriptorToSecurityDescriptorW;
use windows_sys::Win32::Security::{PSECURITY_DESCRIPTOR, SECURITY_ATTRIBUTES};
use windows_sys::Win32::System::Memory::{
    CreateFileMappingW, MapViewOfFile, OpenFileMappingW, UnmapViewOfFile, FILE_MAP_READ, FILE_MAP_WRITE,
    MEMORY_MAPPED_VIEW_ADDRESS, PAGE_READWRITE,
};

pub const NAME: &str = "Global\\KamoVoxLink";
const MAGIC: u32 = 0x4B4D_5658;
const VERSION: u32 = 1;
/// Samples per ring (about 0.68 s at 48 kHz).
pub const RING: usize = 1 << 15;
/// A side whose heartbeat is older than this is considered gone.
pub const ALIVE_MS: u64 = 300;

#[repr(C)]
pub struct Header {
    magic: AtomicU32,
    version: AtomicU32,
    pub sample_rate: AtomicU32,
    pub channels: AtomicU32,
    /// Frames per APO call.
    pub period: AtomicU32,
    /// Set by the app: 1 while it is producing the voice.
    pub app_active: AtomicU32,
    /// Set by the app: samples to keep queued as a safety margin.
    pub cushion: AtomicU32,
    /// Set by the app: largest chunk it writes at once (AI blocks).
    pub burst: AtomicU32,
    pub underruns: AtomicU32,
    _pad: u32,
    pub apo_beat: AtomicU64,
    pub app_beat: AtomicU64,
    /// Id of the APO instance that owns the rings.
    pub owner: AtomicU64,
    pub mic_write: AtomicU64,
    pub voice_write: AtomicU64,
}

#[repr(C)]
struct Layout {
    header: Header,
    mic: [f32; RING],
    voice: [f32; RING],
}

pub fn now_ms() -> u64 {
    unsafe { windows_sys::Win32::System::SystemInformation::GetTickCount64() }
}

pub fn alive(beat: &AtomicU64) -> bool {
    now_ms().saturating_sub(beat.load(Ordering::Acquire)) < ALIVE_MS
}

pub struct Link {
    handle: HANDLE,
    view: *mut Layout,
}

unsafe impl Send for Link {}
unsafe impl Sync for Link {}

fn wide(s: &str) -> Vec<u16> {
    s.encode_utf16().chain(Some(0)).collect()
}

/// `KAMOVOX_LINK` overrides the object name so tests can run unelevated.
fn name() -> Vec<u16> {
    wide(&std::env::var("KAMOVOX_LINK").unwrap_or_else(|_| NAME.to_string()))
}

impl Link {
    /// Creates (or joins) the link. Called by the APO, which runs as a service
    /// and may create global objects; everyone, including low-integrity
    /// processes, gets read/write access.
    pub fn create() -> Result<Link, u32> {
        unsafe {
            let sddl = wide("D:(A;;GA;;;WD)(A;;GA;;;SY)(A;;GA;;;BA)(A;;GA;;;LS)S:(ML;;NW;;;LW)");
            let mut sd: PSECURITY_DESCRIPTOR = null_mut();
            if ConvertStringSecurityDescriptorToSecurityDescriptorW(sddl.as_ptr(), 1, &mut sd, null_mut()) == 0 {
                return Err(GetLastError());
            }
            let sa = SECURITY_ATTRIBUTES {
                nLength: size_of::<SECURITY_ATTRIBUTES>() as u32,
                lpSecurityDescriptor: sd,
                bInheritHandle: 0,
            };
            let size = size_of::<Layout>();
            let name = name();
            let handle = CreateFileMappingW(INVALID_HANDLE_VALUE, &sa, PAGE_READWRITE, 0, size as u32, name.as_ptr());
            let err = GetLastError();
            LocalFree(sd);
            if handle.is_null() {
                return Err(err);
            }
            let link = Self::map(handle).ok_or_else(|| GetLastError())?;
            if err != ERROR_ALREADY_EXISTS || link.header().magic.load(Ordering::Acquire) != MAGIC {
                let h = link.header();
                h.version.store(VERSION, Ordering::Relaxed);
                h.magic.store(MAGIC, Ordering::Release);
            }
            Ok(link)
        }
    }

    /// Opens the link created by the APO. `None` until the APO has run at
    /// least once since the last time every handle was closed.
    pub fn open() -> Option<Link> {
        unsafe {
            let name = name();
            let handle = OpenFileMappingW(FILE_MAP_READ | FILE_MAP_WRITE, 0, name.as_ptr());
            if handle.is_null() {
                return None;
            }
            let link = Self::map(handle)?;
            let h = link.header();
            (h.magic.load(Ordering::Acquire) == MAGIC && h.version.load(Ordering::Relaxed) == VERSION).then_some(link)
        }
    }

    unsafe fn map(handle: HANDLE) -> Option<Link> {
        let view = unsafe { MapViewOfFile(handle, FILE_MAP_READ | FILE_MAP_WRITE, 0, 0, size_of::<Layout>()) };
        if view.Value.is_null() {
            unsafe { CloseHandle(handle) };
            return None;
        }
        Some(Link { handle, view: view.Value as *mut Layout })
    }

    pub fn header(&self) -> &Header {
        unsafe { &(*self.view).header }
    }

    fn ring(&self, voice: bool) -> *mut f32 {
        unsafe {
            if voice {
                (*self.view).voice.as_mut_ptr()
            } else {
                (*self.view).mic.as_mut_ptr()
            }
        }
    }

    fn push(&self, voice: bool, data: &[f32]) {
        let counter = if voice { &self.header().voice_write } else { &self.header().mic_write };
        let ring = self.ring(voice);
        let w = counter.load(Ordering::Relaxed);
        for (i, &s) in data.iter().enumerate() {
            unsafe { ring.add((w as usize + i) % RING).write_volatile(s) };
        }
        counter.store(w + data.len() as u64, Ordering::Release);
    }

    /// Copies up to `out.len()` samples from `*pos`, catching up when the
    /// reader fell more than a ring behind. Returns how many were copied.
    fn pull(&self, voice: bool, pos: &mut u64, out: &mut [f32]) -> usize {
        let counter = if voice { &self.header().voice_write } else { &self.header().mic_write };
        let ring = self.ring(voice);
        let w = counter.load(Ordering::Acquire);
        if *pos > w || w - *pos > RING as u64 / 2 {
            *pos = w.saturating_sub(out.len() as u64);
        }
        let n = ((w - *pos) as usize).min(out.len());
        for (i, o) in out[..n].iter_mut().enumerate() {
            *o = unsafe { ring.add((*pos as usize + i) % RING).read_volatile() };
        }
        *pos += n as u64;
        n
    }

    pub fn push_mic(&self, data: &[f32]) {
        self.push(false, data)
    }

    pub fn pull_mic(&self, pos: &mut u64, out: &mut [f32]) -> usize {
        self.pull(false, pos, out)
    }

    pub fn push_voice(&self, data: &[f32]) {
        self.push(true, data)
    }

    pub fn pull_voice(&self, pos: &mut u64, out: &mut [f32]) -> usize {
        self.pull(true, pos, out)
    }

    /// One voice sample at an absolute position (the caller tracks availability).
    #[inline]
    pub fn voice_at(&self, pos: u64) -> f32 {
        unsafe { self.ring(true).add(pos as usize % RING).read_volatile() }
    }

    pub fn mic_at(&self, pos: u64) -> f32 {
        unsafe { self.ring(false).add(pos as usize % RING).read_volatile() }
    }

    pub fn mic_position(&self) -> u64 {
        self.header().mic_write.load(Ordering::Acquire)
    }

    pub fn voice_position(&self) -> u64 {
        self.header().voice_write.load(Ordering::Acquire)
    }
}

impl Drop for Link {
    fn drop(&mut self) {
        unsafe {
            UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS { Value: self.view as _ });
            CloseHandle(self.handle);
        }
    }
}
