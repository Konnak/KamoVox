//! KamoVox system-effect APO for capture endpoints.
//!
//! Windows loads this DLL inside `audiodg.exe` for the microphone it is
//! installed on. Every audio period it copies the microphone into the shared
//! link and, while the KamoVox app is producing a voice, replaces the
//! microphone signal with that voice. Without the app it is a passthrough.
//!
//! The audio engine creates APOs as aggregated COM objects, so the object is
//! written by hand with explicit vtables and a non-delegating `IUnknown`.

#![allow(non_snake_case)]

use std::cell::UnsafeCell;
use std::ffi::c_void;
use std::io::Write;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

use kamovox_link::{alive, now_ms, Link};
use windows::Win32::Foundation::{
    CLASS_E_CLASSNOTAVAILABLE, E_INVALIDARG, E_NOINTERFACE, E_OUTOFMEMORY, E_POINTER, S_FALSE, S_OK,
};
use windows::Win32::Media::Audio::Apo::{
    IAudioMediaType, IAudioProcessingObject, IAudioProcessingObjectConfiguration, IAudioProcessingObjectRT,
    IAudioSystemEffects, APOERR_FORMAT_NOT_SUPPORTED, APOERR_NUM_CONNECTIONS_INVALID, APO_CONNECTION_DESCRIPTOR,
    APO_CONNECTION_PROPERTY, APO_FLAG, APO_REG_PROPERTIES, UNCOMPRESSEDAUDIOFORMAT,
};
use windows::Win32::System::Com::{CoTaskMemAlloc, IClassFactory};
use windows_core::{Interface, GUID, HRESULT, IUnknown};

/// {7A3E5C21-4B9D-4E6F-A1C8-3D2F9B7E6A54}
pub const CLSID: GUID = GUID::from_u128(0x7a3e5c21_4b9d_4e6f_a1c8_3d2f9b7e6a54);
const FLOAT: GUID = GUID::from_u128(0x00000003_0000_0010_8000_00aa00389b71);
const BUFFER_VALID: i32 = 1;
const BUFFER_SILENT: i32 = 2;

type Raw = *mut c_void;

#[repr(C)]
struct UnkVtbl {
    qi: unsafe extern "system" fn(Raw, *const GUID, *mut Raw) -> HRESULT,
    add_ref: unsafe extern "system" fn(Raw) -> u32,
    release: unsafe extern "system" fn(Raw) -> u32,
}

#[repr(C)]
struct ApoVtbl {
    unk: UnkVtbl,
    reset: unsafe extern "system" fn(Raw) -> HRESULT,
    get_latency: unsafe extern "system" fn(Raw, *mut i64) -> HRESULT,
    get_registration: unsafe extern "system" fn(Raw, *mut *mut APO_REG_PROPERTIES) -> HRESULT,
    initialize: unsafe extern "system" fn(Raw, u32, *const u8) -> HRESULT,
    is_input_supported: unsafe extern "system" fn(Raw, Raw, Raw, *mut Raw) -> HRESULT,
    is_output_supported: unsafe extern "system" fn(Raw, Raw, Raw, *mut Raw) -> HRESULT,
    input_channels: unsafe extern "system" fn(Raw, *mut u32) -> HRESULT,
}

#[repr(C)]
struct RtVtbl {
    unk: UnkVtbl,
    process: unsafe extern "system" fn(Raw, u32, *const *const APO_CONNECTION_PROPERTY, u32, *mut *mut APO_CONNECTION_PROPERTY),
    calc_input: unsafe extern "system" fn(Raw, u32) -> u32,
    calc_output: unsafe extern "system" fn(Raw, u32) -> u32,
}

#[repr(C)]
struct CfgVtbl {
    unk: UnkVtbl,
    lock: unsafe extern "system" fn(Raw, u32, *const *const APO_CONNECTION_DESCRIPTOR, u32, *const *const APO_CONNECTION_DESCRIPTOR) -> HRESULT,
    unlock: unsafe extern "system" fn(Raw) -> HRESULT,
}

#[repr(C)]
struct FactoryVtbl {
    unk: UnkVtbl,
    create: unsafe extern "system" fn(Raw, Raw, *const GUID, *mut Raw) -> HRESULT,
    lock_server: unsafe extern "system" fn(Raw, i32) -> HRESULT,
}

/// COM object: one vtable pointer per interface, followed by the data.
#[repr(C)]
struct Apo {
    inner: &'static UnkVtbl,
    apo: &'static ApoVtbl,
    rt: &'static RtVtbl,
    cfg: &'static CfgVtbl,
    sfx: &'static UnkVtbl,
    refs: AtomicU32,
    /// Controlling `IUnknown`: the aggregating object, or our own `inner`.
    outer: Raw,
    state: UnsafeCell<State>,
}

const SLOT_INNER: usize = 0;
const SLOT_APO: usize = 1;
const SLOT_RT: usize = 2;
const SLOT_CFG: usize = 3;
const SLOT_SFX: usize = 4;

unsafe fn obj<const SLOT: usize>(this: Raw) -> &'static Apo {
    unsafe { &*((this as *mut *const c_void).sub(SLOT) as *const Apo) }
}

unsafe fn slot(apo: &Apo, index: usize) -> Raw {
    unsafe { (apo as *const Apo as *mut *const c_void).add(index) as Raw }
}

unsafe fn unk_vtbl(p: Raw) -> &'static UnkVtbl {
    unsafe { &**(p as *const *const UnkVtbl) }
}

unsafe extern "system" fn d_qi<const SLOT: usize>(this: Raw, iid: *const GUID, out: *mut Raw) -> HRESULT {
    unsafe {
        let outer = obj::<SLOT>(this).outer;
        (unk_vtbl(outer).qi)(outer, iid, out)
    }
}

unsafe extern "system" fn d_add_ref<const SLOT: usize>(this: Raw) -> u32 {
    unsafe {
        let outer = obj::<SLOT>(this).outer;
        (unk_vtbl(outer).add_ref)(outer)
    }
}

unsafe extern "system" fn d_release<const SLOT: usize>(this: Raw) -> u32 {
    unsafe {
        let outer = obj::<SLOT>(this).outer;
        (unk_vtbl(outer).release)(outer)
    }
}

const fn delegating<const SLOT: usize>() -> UnkVtbl {
    UnkVtbl { qi: d_qi::<SLOT>, add_ref: d_add_ref::<SLOT>, release: d_release::<SLOT> }
}

unsafe extern "system" fn inner_qi(this: Raw, iid: *const GUID, out: *mut Raw) -> HRESULT {
    unsafe {
        if iid.is_null() || out.is_null() {
            return E_POINTER;
        }
        let apo = obj::<SLOT_INNER>(this);
        let iid = &*iid;
        let index = if *iid == IUnknown::IID {
            SLOT_INNER
        } else if *iid == IAudioProcessingObject::IID {
            SLOT_APO
        } else if *iid == IAudioProcessingObjectRT::IID {
            SLOT_RT
        } else if *iid == IAudioProcessingObjectConfiguration::IID {
            SLOT_CFG
        } else if *iid == IAudioSystemEffects::IID {
            SLOT_SFX
        } else {
            *out = std::ptr::null_mut();
            return E_NOINTERFACE;
        };
        let p = slot(apo, index);
        (unk_vtbl(p).add_ref)(p);
        *out = p;
        S_OK
    }
}

unsafe extern "system" fn inner_add_ref(this: Raw) -> u32 {
    unsafe { obj::<SLOT_INNER>(this).refs.fetch_add(1, Ordering::AcqRel) + 1 }
}

unsafe extern "system" fn inner_release(this: Raw) -> u32 {
    unsafe {
        let apo = obj::<SLOT_INNER>(this);
        let left = apo.refs.fetch_sub(1, Ordering::AcqRel) - 1;
        if left == 0 {
            drop(Box::from_raw(apo as *const Apo as *mut Apo));
        }
        left
    }
}

static INNER_VTBL: UnkVtbl = UnkVtbl { qi: inner_qi, add_ref: inner_add_ref, release: inner_release };
static SFX_VTBL: UnkVtbl = delegating::<SLOT_SFX>();

static APO_VTBL: ApoVtbl = ApoVtbl {
    unk: delegating::<SLOT_APO>(),
    reset: apo_reset,
    get_latency: apo_latency,
    get_registration: apo_registration,
    initialize: apo_initialize,
    is_input_supported: apo_format_supported,
    is_output_supported: apo_format_supported,
    input_channels: apo_input_channels,
};

static RT_VTBL: RtVtbl = RtVtbl {
    unk: delegating::<SLOT_RT>(),
    process: rt_process,
    calc_input: rt_same,
    calc_output: rt_same,
};

static CFG_VTBL: CfgVtbl = CfgVtbl { unk: delegating::<SLOT_CFG>(), lock: cfg_lock, unlock: cfg_unlock };

unsafe extern "system" fn apo_reset(_: Raw) -> HRESULT {
    S_OK
}

unsafe extern "system" fn apo_latency(_: Raw, out: *mut i64) -> HRESULT {
    if out.is_null() {
        return E_POINTER;
    }
    unsafe { *out = 0 };
    S_OK
}

fn wide_into(dst: &mut [u16; 256], s: &str) {
    for (d, c) in dst.iter_mut().zip(s.encode_utf16().take(255)) {
        *d = c;
    }
}

unsafe extern "system" fn apo_registration(_: Raw, out: *mut *mut APO_REG_PROPERTIES) -> HRESULT {
    unsafe {
        if out.is_null() {
            return E_POINTER;
        }
        let p = CoTaskMemAlloc(size_of::<APO_REG_PROPERTIES>()) as *mut APO_REG_PROPERTIES;
        if p.is_null() {
            return E_OUTOFMEMORY;
        }
        let mut props = APO_REG_PROPERTIES {
            clsid: CLSID,
            Flags: APO_FLAG(0xf),
            u32MajorVersion: 1,
            u32MinorVersion: 0,
            u32MinInputConnections: 1,
            u32MaxInputConnections: 1,
            u32MinOutputConnections: 1,
            u32MaxOutputConnections: 1,
            u32MaxInstances: u32::MAX,
            u32NumAPOInterfaces: 1,
            iidAPOInterfaceList: [IAudioProcessingObject::IID],
            ..Default::default()
        };
        wide_into(&mut props.szFriendlyName, "KamoVox");
        wide_into(&mut props.szCopyrightInfo, "KamoVox");
        p.write(props);
        *out = p;
        S_OK
    }
}

unsafe extern "system" fn apo_initialize(_: Raw, size: u32, data: *const u8) -> HRESULT {
    if size > 0 && data.is_null() {
        return E_INVALIDARG;
    }
    S_OK
}

fn describe(t: &IAudioMediaType) -> Option<UNCOMPRESSEDAUDIOFORMAT> {
    let mut f = UNCOMPRESSEDAUDIOFORMAT::default();
    unsafe { t.GetUncompressedAudioFormat(&mut f) }.ok()?;
    Some(f)
}

unsafe extern "system" fn apo_format_supported(_: Raw, opposite: Raw, requested: Raw, out: *mut Raw) -> HRESULT {
    unsafe {
        if requested.is_null() || out.is_null() {
            return E_POINTER;
        }
        *out = std::ptr::null_mut();
        let Some(req) = IAudioMediaType::from_raw_borrowed(&requested) else { return E_POINTER };
        let Some(f) = describe(req) else { return APOERR_FORMAT_NOT_SUPPORTED };
        if f.guidFormatType != FLOAT || f.dwBytesPerSampleContainer != 4 {
            return APOERR_FORMAT_NOT_SUPPORTED;
        }
        // In-place: input and output must match; suggest the other side's format.
        if let Some(opp) = IAudioMediaType::from_raw_borrowed(&opposite) {
            if let Some(o) = describe(opp) {
                if o.dwSamplesPerFrame != f.dwSamplesPerFrame || o.fFramesPerSecond != f.fFramesPerSecond {
                    *out = opp.clone().into_raw();
                    return S_FALSE;
                }
            }
        }
        *out = req.clone().into_raw();
        S_OK
    }
}

unsafe extern "system" fn apo_input_channels(this: Raw, out: *mut u32) -> HRESULT {
    unsafe {
        if out.is_null() {
            return E_POINTER;
        }
        *out = (*obj::<SLOT_APO>(this).state.get()).channels as u32;
        S_OK
    }
}

unsafe extern "system" fn cfg_lock(
    this: Raw,
    n_in: u32,
    ins: *const *const APO_CONNECTION_DESCRIPTOR,
    n_out: u32,
    outs: *const *const APO_CONNECTION_DESCRIPTOR,
) -> HRESULT {
    unsafe {
        if n_in != 1 || n_out != 1 || ins.is_null() || outs.is_null() {
            return APOERR_NUM_CONNECTIONS_INVALID;
        }
        let d = &**ins;
        let Some(fmt) = (*d.pFormat).as_ref() else { return E_POINTER };
        let Some(f) = describe(fmt) else { return APOERR_FORMAT_NOT_SUPPORTED };
        if f.guidFormatType != FLOAT || f.dwBytesPerSampleContainer != 4 {
            return APOERR_FORMAT_NOT_SUPPORTED;
        }
        let state = &mut *obj::<SLOT_CFG>(this).state.get();
        state.configure(f.dwSamplesPerFrame.max(1) as usize, f.fFramesPerSecond as u32, d.u32MaxFrameCount as usize);
        S_OK
    }
}

unsafe extern "system" fn cfg_unlock(this: Raw) -> HRESULT {
    unsafe { (*obj::<SLOT_CFG>(this).state.get()).release_ownership() };
    S_OK
}

unsafe extern "system" fn rt_same(_: Raw, frames: u32) -> u32 {
    frames
}

unsafe extern "system" fn rt_process(
    this: Raw,
    n_in: u32,
    ins: *const *const APO_CONNECTION_PROPERTY,
    n_out: u32,
    outs: *mut *mut APO_CONNECTION_PROPERTY,
) {
    unsafe {
        if n_in == 0 || n_out == 0 || ins.is_null() || outs.is_null() {
            return;
        }
        let input = &**ins;
        let output = &mut **outs;
        let state = &mut *obj::<SLOT_RT>(this).state.get();
        let frames = (input.u32ValidFrameCount as usize).min(state.max_frames);
        let ch = state.channels;
        let src = std::slice::from_raw_parts(input.pBuffer as *const f32, frames * ch);
        let silent = input.u32BufferFlags.0 == BUFFER_SILENT;
        if output.pBuffer == input.pBuffer {
            state.process_in_place(output.pBuffer as *mut f32, frames, silent);
        } else {
            let dst = std::slice::from_raw_parts_mut(output.pBuffer as *mut f32, frames * ch);
            if silent {
                dst.fill(0.0);
            } else {
                dst.copy_from_slice(src);
            }
            state.process_in_place(dst.as_mut_ptr(), frames, false);
        }
        output.u32ValidFrameCount = frames as u32;
        output.u32BufferFlags.0 = BUFFER_VALID;
    }
}

#[derive(Clone, Copy, PartialEq)]
enum Play {
    /// App not producing a voice: pass the microphone through.
    Off,
    /// Waiting for the first processed audio.
    Waiting,
    /// Holding back `n` samples of silence as a jitter cushion.
    Priming(u32),
    Playing,
}

struct State {
    id: u64,
    channels: usize,
    sample_rate: u32,
    max_frames: usize,
    mono: Vec<f32>,
    link: Option<Link>,
    link_tried: bool,
    read: u64,
    play: Play,
    last: f32,
    /// 0 = microphone, 1 = KamoVox voice.
    mix: f32,
    mix_step: f32,
}

static NEXT_ID: AtomicU64 = AtomicU64::new(1);

fn log(msg: &str) {
    let path = std::path::Path::new("C:\\ProgramData\\KamoVox\\apo.log");
    if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(path) {
        let _ = writeln!(f, "[{}] {msg}", now_ms());
    }
}

impl State {
    fn new() -> Self {
        Self {
            id: (u64::from(std::process::id()) << 32) | NEXT_ID.fetch_add(1, Ordering::Relaxed),
            channels: 1,
            sample_rate: 48_000,
            max_frames: 0,
            mono: Vec::new(),
            link: None,
            link_tried: false,
            read: 0,
            play: Play::Off,
            last: 0.0,
            mix: 0.0,
            mix_step: 0.0,
        }
    }

    fn configure(&mut self, channels: usize, sample_rate: u32, max_frames: usize) {
        self.channels = channels;
        self.sample_rate = sample_rate.max(8_000);
        self.max_frames = max_frames;
        self.mono = vec![0.0; max_frames];
        self.mix_step = 1.0 / (self.sample_rate as f32 * 0.01);
        self.play = Play::Off;
        self.last = 0.0;
        self.mix = 0.0;
        if self.link.is_none() && !self.link_tried {
            self.link_tried = true;
            match Link::create() {
                Ok(l) => {
                    log(&format!("pronto: {channels} canais, {sample_rate} Hz, até {max_frames} amostras"));
                    self.link = Some(l);
                }
                Err(e) => log(&format!("falha ao criar o link: erro {e}")),
            }
        }
    }

    fn release_ownership(&mut self) {
        if let Some(l) = &self.link {
            let _ = l.header().owner.compare_exchange(self.id, 0, Ordering::AcqRel, Ordering::Relaxed);
        }
    }

    fn process_in_place(&mut self, buf: *mut f32, frames: usize, silent: bool) {
        let ch = self.channels;
        let Some(link) = &self.link else {
            if silent {
                unsafe { std::slice::from_raw_parts_mut(buf, frames * ch).fill(0.0) };
            }
            return;
        };
        let data = unsafe { std::slice::from_raw_parts_mut(buf, frames * ch) };
        // Windows opens one effect per program. The extra capture KamoVox keeps
        // open can be silent while Discord still receives the microphone, and
        // that silent client used to own the link. Measure first: a silent
        // flag means the samples themselves must not be trusted.
        let energy = if silent {
            0.0
        } else {
            data.iter().fold(0f32, |m, s| m.max(s.abs()))
        };
        if silent {
            data.fill(0.0);
        }
        let h = link.header();
        let owner = h.owner.load(Ordering::Acquire);
        let owner_alive = owner != 0 && alive(&h.apo_beat);
        let mine = owner == self.id;
        // Take the microphone from a client that is only forwarding silence.
        let steal = owner_alive && !mine && energy > 0.002 && energy > recent_mic_peak(link) * 4.0;
        if mine || !owner_alive || steal {
            if !mine {
                if steal {
                    log("o microfone real assumiu o envio");
                }
                h.owner.store(self.id, Ordering::Release);
                self.play = Play::Off;
                self.mix = 0.0;
                self.read = 0;
            }
            h.apo_beat.store(now_ms(), Ordering::Release);
            h.sample_rate.store(self.sample_rate, Ordering::Relaxed);
            h.channels.store(ch as u32, Ordering::Relaxed);
            h.period.store(frames as u32, Ordering::Relaxed);

            let scale = 1.0 / ch as f32;
            for (m, frame) in self.mono.iter_mut().zip(data.chunks_exact(ch)) {
                *m = frame.iter().sum::<f32>() * scale;
            }
            link.push_mic(&self.mono[..frames]);
        }

        // Every program hears the voice. Only the owner feeds the microphone in.
        let active = h.app_active.load(Ordering::Acquire) == 1 && alive(&h.app_beat);
        if !active {
            self.play = Play::Off;
        } else if self.play == Play::Off {
            self.play = Play::Waiting;
            self.read = link.voice_position();
        }
        if self.play == Play::Off && self.mix == 0.0 {
            return;
        }

        let cushion = h.cushion.load(Ordering::Relaxed);
        let burst = h.burst.load(Ordering::Relaxed) as u64;
        let written = link.voice_position();
        if self.read > written {
            self.read = written;
        }
        let mut avail = written - self.read;
        let ring = kamovox_link::RING as u64;
        if avail > ring {
            self.read = written - (cushion as u64 + frames as u64).min(ring / 2);
            avail = written - self.read;
        }
        let limit = (cushion as u64 + burst + 2 * frames as u64 + u64::from(self.sample_rate) / 20).min(ring);
        if self.play == Play::Playing && avail > limit {
            self.read = written - (cushion as u64 + frames as u64).min(avail);
            avail = written - self.read;
        }

        let count_gaps = h.owner.load(Ordering::Acquire) == self.id;
        let target = if active { 1.0 } else { 0.0 };
        for frame in data.chunks_exact_mut(ch) {
            let voice = match self.play {
                Play::Off => 0.0,
                Play::Waiting => {
                    if avail > cushion as u64 {
                        self.read = written - cushion as u64;
                        avail = cushion as u64;
                        self.play = Play::Playing;
                    }
                    0.0
                }
                Play::Priming(n) => {
                    self.play = if n <= 1 { Play::Playing } else { Play::Priming(n - 1) };
                    0.0
                }
                Play::Playing => {
                    if avail > 0 {
                        let v = link.voice_at(self.read);
                        self.read += 1;
                        avail -= 1;
                        self.last = v;
                        v
                    } else {
                        if count_gaps {
                            h.underruns.fetch_add(1, Ordering::Relaxed);
                        }
                        self.last
                    }
                }
            };
            if self.mix != target {
                self.mix = if target > self.mix {
                    (self.mix + self.mix_step).min(1.0)
                } else {
                    (self.mix - self.mix_step).max(0.0)
                };
            }
            let m = self.mix;
            if m > 0.0 {
                for s in frame.iter_mut() {
                    *s = *s * (1.0 - m) + voice * m;
                }
            }
        }
    }
}

/// Peak of the microphone the current owner just wrote. Zero means that
/// client is not the one carrying the real signal.
fn recent_mic_peak(link: &Link) -> f32 {
    let end = link.mic_position();
    let n = 480.min(end as usize);
    let mut peak = 0f32;
    for i in 0..n {
        peak = peak.max(link.mic_at(end - 1 - i as u64).abs());
    }
    peak
}

fn new_apo(outer: Raw) -> Raw {
    let apo = Box::new(Apo {
        inner: &INNER_VTBL,
        apo: &APO_VTBL,
        rt: &RT_VTBL,
        cfg: &CFG_VTBL,
        sfx: &SFX_VTBL,
        refs: AtomicU32::new(1),
        outer: std::ptr::null_mut(),
        state: UnsafeCell::new(State::new()),
    });
    let raw = Box::into_raw(apo);
    unsafe {
        (*raw).outer = if outer.is_null() { raw as Raw } else { outer };
    }
    raw as Raw
}

unsafe extern "system" fn factory_qi(this: Raw, iid: *const GUID, out: *mut Raw) -> HRESULT {
    unsafe {
        if iid.is_null() || out.is_null() {
            return E_POINTER;
        }
        if *iid == IUnknown::IID || *iid == IClassFactory::IID {
            *out = this;
            S_OK
        } else {
            *out = std::ptr::null_mut();
            E_NOINTERFACE
        }
    }
}

unsafe extern "system" fn factory_ref(_: Raw) -> u32 {
    1
}

unsafe extern "system" fn factory_create(_: Raw, outer: Raw, iid: *const GUID, out: *mut Raw) -> HRESULT {
    unsafe {
        if iid.is_null() || out.is_null() {
            return E_POINTER;
        }
        *out = std::ptr::null_mut();
        // An aggregated object may only hand out its inner IUnknown.
        if !outer.is_null() && *iid != IUnknown::IID {
            return E_NOINTERFACE;
        }
        let inner = new_apo(outer);
        let hr = inner_qi(inner, iid, out);
        inner_release(inner);
        hr
    }
}

unsafe extern "system" fn factory_lock(_: Raw, _: i32) -> HRESULT {
    S_OK
}

static FACTORY_VTBL: FactoryVtbl = FactoryVtbl {
    unk: UnkVtbl { qi: factory_qi, add_ref: factory_ref, release: factory_ref },
    create: factory_create,
    lock_server: factory_lock,
};

#[repr(C)]
struct Factory(&'static FactoryVtbl);
unsafe impl Sync for Factory {}
static FACTORY: Factory = Factory(&FACTORY_VTBL);

#[no_mangle]
pub unsafe extern "system" fn DllGetClassObject(clsid: *const GUID, iid: *const GUID, out: *mut Raw) -> HRESULT {
    unsafe {
        if clsid.is_null() || out.is_null() {
            return E_POINTER;
        }
        if *clsid != CLSID {
            *out = std::ptr::null_mut();
            return CLASS_E_CLASSNOTAVAILABLE;
        }
        factory_qi(&FACTORY as *const Factory as Raw, iid, out)
    }
}

#[no_mangle]
pub extern "system" fn DllCanUnloadNow() -> HRESULT {
    S_FALSE
}
