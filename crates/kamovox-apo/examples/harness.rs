//! Loads the built APO DLL like the audio engine does and drives it with a
//! fake microphone and a fake KamoVox app.
//!
//! `cargo build -p kamovox-apo --release; cargo run -p kamovox-apo --release --example harness`

#![allow(non_snake_case)]

use std::ffi::c_void;
use std::sync::atomic::Ordering;

use kamovox_link::{now_ms, Link};
use windows::Win32::Media::Audio::Apo::*;
use windows::Win32::Media::Audio::WAVEFORMATEX;
use windows::Win32::System::Com::IClassFactory;
use windows::Win32::System::LibraryLoader::{GetProcAddress, LoadLibraryW};
use windows_core::{implement, w, Interface, BOOL, GUID, HRESULT, IUnknown, Ref, Result, s};

const CLSID: GUID = GUID::from_u128(0x7a3e5c21_4b9d_4e6f_a1c8_3d2f9b7e6a54);
const FLOAT: GUID = GUID::from_u128(0x00000003_0000_0010_8000_00aa00389b71);

#[implement(IAudioMediaType)]
struct Format(u32, f32);

impl IAudioMediaType_Impl for Format_Impl {
    fn IsCompressedFormat(&self) -> Result<BOOL> {
        Ok(false.into())
    }
    fn IsEqual(&self, _: Ref<IAudioMediaType>) -> Result<u32> {
        Ok(0)
    }
    fn GetAudioFormat(&self) -> *mut WAVEFORMATEX {
        std::ptr::null_mut()
    }
    fn GetUncompressedAudioFormat(&self, f: *mut UNCOMPRESSEDAUDIOFORMAT) -> Result<()> {
        unsafe {
            *f = UNCOMPRESSEDAUDIOFORMAT {
                guidFormatType: FLOAT,
                dwSamplesPerFrame: self.0,
                dwBytesPerSampleContainer: 4,
                dwValidBitsPerSample: 32,
                fFramesPerSecond: self.1,
                dwChannelMask: 3,
            };
        }
        Ok(())
    }
}

/// Minimal aggregating object, like the audio engine's wrapper.
#[repr(C)]
struct Outer {
    vtbl: &'static [usize; 3],
    refs: std::sync::atomic::AtomicU32,
    inner: *mut c_void,
}

unsafe extern "system" fn outer_qi(this: *mut c_void, iid: *const GUID, out: *mut *mut c_void) -> HRESULT {
    unsafe {
        let o = &*(this as *const Outer);
        if *iid == IUnknown::IID {
            outer_add(this);
            *out = this;
            return HRESULT(0);
        }
        let inner = IUnknown::from_raw_borrowed(&o.inner).unwrap();
        inner.query(&*iid, out)
    }
}
unsafe extern "system" fn outer_add(this: *mut c_void) -> u32 {
    unsafe { (*(this as *const Outer)).refs.fetch_add(1, Ordering::AcqRel) + 1 }
}
unsafe extern "system" fn outer_release(this: *mut c_void) -> u32 {
    unsafe { (*(this as *const Outer)).refs.fetch_sub(1, Ordering::AcqRel) - 1 }
}
static OUTER_VTBL: [usize; 3] = [0; 3];

fn aggregated(factory: &IClassFactory) -> Result<()> {
    let vtbl: &'static [usize; 3] = Box::leak(Box::new([outer_qi as usize, outer_add as usize, outer_release as usize]));
    let _ = &OUTER_VTBL;
    let outer = Box::leak(Box::new(Outer { vtbl, refs: 1.into(), inner: std::ptr::null_mut() }));
    let outer_raw = outer as *mut Outer as *mut c_void;
    let outer_unk = unsafe { IUnknown::from_raw_borrowed(&outer_raw).unwrap() };
    let inner: IUnknown = unsafe { factory.CreateInstance(outer_unk)? };
    outer.inner = inner.as_raw();
    let mut apo = std::ptr::null_mut();
    unsafe { outer_qi(outer as *mut Outer as _, &IAudioProcessingObject::IID, &mut apo).ok()? };
    let apo = unsafe { IAudioProcessingObject::from_raw(apo) };
    let back: IUnknown = apo.cast()?;
    let ok = back.as_raw() == outer as *mut Outer as *mut c_void;
    let rt: IAudioProcessingObjectRT = apo.cast()?;
    drop((rt, back, apo));
    println!("Agregação: interfaces voltam ao objeto externo = {ok}, referências externas = {}", outer.refs.load(Ordering::Relaxed));
    drop(inner);
    Ok(())
}

type GetClassObject = unsafe extern "system" fn(*const GUID, *const GUID, *mut *mut c_void) -> HRESULT;

fn main() -> Result<()> {
    unsafe { std::env::set_var("KAMOVOX_LINK", "Local\\KamoVoxHarness") };
    let dll = std::env::current_exe().unwrap().parent().unwrap().parent().unwrap().join("kamovox_apo.dll");
    let path: Vec<u16> = dll.to_string_lossy().encode_utf16().chain(Some(0)).collect();
    let module = unsafe { LoadLibraryW(windows_core::PCWSTR(path.as_ptr()))? };
    let get: GetClassObject = unsafe { std::mem::transmute(GetProcAddress(module, s!("DllGetClassObject")).expect("export")) };
    let _ = w!("");

    let mut factory = std::ptr::null_mut();
    unsafe { get(&CLSID, &IClassFactory::IID, &mut factory).ok()? };
    let factory = unsafe { IClassFactory::from_raw(factory) };
    let apo: IAudioProcessingObject = unsafe { factory.CreateInstance(None::<&IUnknown>)? };
    let rt: IAudioProcessingObjectRT = apo.cast()?;
    let cfg: IAudioProcessingObjectConfiguration = apo.cast()?;
    let _sfx: IAudioSystemEffects = apo.cast()?;
    println!("COM: criado e todas as interfaces respondem");

    aggregated(&factory)?;

    let props = unsafe { apo.GetRegistrationProperties()? };
    println!("Registro: flags {:#x}, clsid ok = {}", unsafe { (*props).Flags.0 }, unsafe { (*props).clsid } == CLSID);

    let channels = 2u32;
    let sr = 48_000.0;
    let fmt: IAudioMediaType = Format(channels, sr).into();
    let supported = unsafe { apo.IsInputFormatSupported(&fmt, &fmt)? };
    println!("Formato float estéreo aceito: {}", supported.as_raw() == fmt.as_raw());

    let frames = 480u32;
    let desc = APO_CONNECTION_DESCRIPTOR {
        Type: APO_CONNECTION_BUFFER_TYPE_EXTERNAL,
        pBuffer: 0,
        u32MaxFrameCount: frames,
        pFormat: std::mem::ManuallyDrop::new(Some(fmt.clone())),
        u32Signature: 0,
    };
    unsafe { cfg.LockForProcess(&[&desc], &[&desc])? };

    let link = Link::open().expect("o APO deveria ter criado o link");
    let h = link.header();
    let mut buf = vec![0f32; (frames * channels) as usize];
    let mut run = |buf: &mut Vec<f32>, value: f32| {
        buf.fill(value);
        let mut prop = APO_CONNECTION_PROPERTY {
            pBuffer: buf.as_mut_ptr() as usize,
            u32ValidFrameCount: frames,
            u32BufferFlags: APO_BUFFER_FLAGS(1),
            u32Signature: 0,
        };
        let input: *const APO_CONNECTION_PROPERTY = &prop;
        let mut output: *mut APO_CONNECTION_PROPERTY = &mut prop;
        unsafe { rt.APOProcess(1, &input, 1, &mut output) };
    };

    run(&mut buf, 0.25);
    println!("Sem o app: passa o microfone direto ({})", buf[0]);

    let mut mic = vec![0f32; 4096];
    let mut pos = 0;
    let got = link.pull_mic(&mut pos, &mut mic);
    println!("Microfone chegou no app: {got} amostras, valor {}", mic[0]);

    h.cushion.store(240, Ordering::Relaxed);
    h.app_beat.store(now_ms(), Ordering::Release);
    h.app_active.store(1, Ordering::Release);
    let mut outputs = Vec::new();
    for _ in 0..12 {
        h.app_beat.store(now_ms(), Ordering::Release);
        run(&mut buf, 0.25);
        outputs.push(*buf.last().unwrap());
        // The "app" turns every microphone sample into -0.5.
        let n = link.pull_mic(&mut pos, &mut mic);
        link.push_voice(&vec![-0.5; n]);
    }
    println!("Com o app (fim de cada período): {outputs:?}");
    println!("Falhas de buffer: {}", h.underruns.load(Ordering::Relaxed));

    h.app_active.store(0, Ordering::Release);
    let mut outputs = Vec::new();
    for _ in 0..3 {
        run(&mut buf, 0.25);
        outputs.push(*buf.last().unwrap());
    }
    println!("App desligado volta ao microfone: {outputs:?}");
    unsafe { cfg.UnlockForProcess()? };
    Ok(())
}
