//! The other push-to-talk program mutes the Windows microphone itself.
//! That mute stays on after it closes, so KamoVox keeps receiving silence.
//! KamoVox cuts its own voice in software and leaves the microphone open.

use windows_sys::core::GUID;
use windows_sys::Win32::System::Com::{CoCreateInstance, CoInitializeEx, CLSCTX_ALL, COINIT_MULTITHREADED};

const CLSID_ENUMERATOR: GUID = GUID::from_u128(0xBCDE0395_E52F_467C_8E3D_C4579291692E);
const IID_ENUMERATOR: GUID = GUID::from_u128(0xA95664D2_9614_4F35_A746_DE8DB63617E6);
const IID_VOLUME: GUID = GUID::from_u128(0x5CDF2C82_841E_4546_9722_0CF74078229A);
const CAPTURE: u32 = 1;
const ACTIVE: u32 = 1;

/// Clears the Windows mute on every active microphone.
pub fn unmute_captures() {
    unsafe {
        let _ = CoInitializeEx(std::ptr::null(), COINIT_MULTITHREADED as u32);
        let mut enumerator: *mut std::ffi::c_void = std::ptr::null_mut();
        if CoCreateInstance(&CLSID_ENUMERATOR, std::ptr::null_mut(), CLSCTX_ALL, &IID_ENUMERATOR, &mut enumerator) != 0
            || enumerator.is_null()
        {
            return;
        }
        let mut list: *mut std::ffi::c_void = std::ptr::null_mut();
        let hr = com_call3(enumerator, 3, CAPTURE, ACTIVE, &mut list);
        release(enumerator);
        if hr != 0 || list.is_null() {
            return;
        }
        let mut count = 0u32;
        let _ = com_call_out(list, 3, &mut count);
        for i in 0..count {
            let mut device: *mut std::ffi::c_void = std::ptr::null_mut();
            if com_call_index(list, 4, i, &mut device) != 0 || device.is_null() {
                continue;
            }
            let mut volume: *mut std::ffi::c_void = std::ptr::null_mut();
            let hr = com_call_activate(device, &IID_VOLUME, &mut volume);
            release(device);
            if hr != 0 || volume.is_null() {
                continue;
            }
            let _ = set_mute(volume, false);
            release(volume);
        }
        release(list);
    }
}

unsafe fn set_mute(volume: *mut std::ffi::c_void, mute: bool) -> i32 {
    unsafe {
        let vt = *(volume as *const *const usize);
        let set: unsafe extern "system" fn(*mut std::ffi::c_void, i32, *const GUID) -> i32 = std::mem::transmute(*vt.add(14));
        set(volume, i32::from(mute), std::ptr::null())
    }
}

unsafe fn com_call3(this: *mut std::ffi::c_void, slot: usize, a: u32, b: u32, out: *mut *mut std::ffi::c_void) -> i32 {
    unsafe {
        let vt = *(this as *const *const usize);
        let f: unsafe extern "system" fn(*mut std::ffi::c_void, u32, u32, *mut *mut std::ffi::c_void) -> i32 = std::mem::transmute(*vt.add(slot));
        f(this, a, b, out)
    }
}

unsafe fn com_call_out(this: *mut std::ffi::c_void, slot: usize, out: *mut u32) -> i32 {
    unsafe {
        let vt = *(this as *const *const usize);
        let f: unsafe extern "system" fn(*mut std::ffi::c_void, *mut u32) -> i32 = std::mem::transmute(*vt.add(slot));
        f(this, out)
    }
}

unsafe fn com_call_index(this: *mut std::ffi::c_void, slot: usize, index: u32, out: *mut *mut std::ffi::c_void) -> i32 {
    unsafe {
        let vt = *(this as *const *const usize);
        let f: unsafe extern "system" fn(*mut std::ffi::c_void, u32, *mut *mut std::ffi::c_void) -> i32 = std::mem::transmute(*vt.add(slot));
        f(this, index, out)
    }
}

unsafe fn com_call_activate(this: *mut std::ffi::c_void, iid: *const GUID, out: *mut *mut std::ffi::c_void) -> i32 {
    unsafe {
        let vt = *(this as *const *const usize);
        let f: unsafe extern "system" fn(*mut std::ffi::c_void, *const GUID, u32, *mut std::ffi::c_void, *mut *mut std::ffi::c_void) -> i32 =
            std::mem::transmute(*vt.add(3));
        f(this, iid, CLSCTX_ALL, std::ptr::null_mut(), out)
    }
}

unsafe fn release(this: *mut std::ffi::c_void) {
    unsafe {
        if this.is_null() {
            return;
        }
        let vt = *(this as *const *const usize);
        let f: unsafe extern "system" fn(*mut std::ffi::c_void) -> u32 = std::mem::transmute(*vt.add(2));
        f(this);
    }
}
