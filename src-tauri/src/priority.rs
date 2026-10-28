//! Keeps audio smooth while the window is minimized or hidden in the tray.
//!
//! Windows 11 puts background processes in "efficiency mode" (EcoQoS): lower
//! clocks and a coarse 15.6 ms timer. Our 1–2 ms polling loops then wake far too
//! late and the voice crackles, so the process opts out and the audio threads
//! register with the multimedia scheduler (MMCSS).

#[cfg(windows)]
pub fn keep_process_responsive() {
    use windows_sys::Wdk::Graphics::Direct3D::{D3DKMTSetProcessSchedulingPriorityClass, D3DKMT_SCHEDULINGPRIORITYCLASS_HIGH};
    use windows_sys::Win32::Media::timeBeginPeriod;
    use windows_sys::Win32::System::Threading::{
        GetCurrentProcess, ProcessPowerThrottling, SetPriorityClass, SetProcessInformation, ABOVE_NORMAL_PRIORITY_CLASS,
        HIGH_PRIORITY_CLASS,
        PROCESS_POWER_THROTTLING_CURRENT_VERSION, PROCESS_POWER_THROTTLING_EXECUTION_SPEED,
        PROCESS_POWER_THROTTLING_IGNORE_TIMER_RESOLUTION, PROCESS_POWER_THROTTLING_STATE,
    };
    unsafe {
        let state = PROCESS_POWER_THROTTLING_STATE {
            Version: PROCESS_POWER_THROTTLING_CURRENT_VERSION,
            ControlMask: PROCESS_POWER_THROTTLING_EXECUTION_SPEED | PROCESS_POWER_THROTTLING_IGNORE_TIMER_RESOLUTION,
            StateMask: 0,
        };
        SetProcessInformation(
            GetCurrentProcess(),
            ProcessPowerThrottling,
            &state as *const _ as *const core::ffi::c_void,
            size_of::<PROCESS_POWER_THROTTLING_STATE>() as u32,
        );
        // Like a game: the voice must not wait behind other programs, on the
        // CPU nor on the graphics card that runs the AI voice.
        if SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS) == 0 {
            SetPriorityClass(GetCurrentProcess(), ABOVE_NORMAL_PRIORITY_CLASS);
        }
        D3DKMTSetProcessSchedulingPriorityClass(GetCurrentProcess(), D3DKMT_SCHEDULINGPRIORITYCLASS_HIGH);
        timeBeginPeriod(1);
    }
}

#[cfg(not(windows))]
pub fn keep_process_responsive() {}

/// Registers the calling thread as a "Pro Audio" task.
#[cfg(windows)]
pub fn pro_audio_thread() {
    use windows_sys::Win32::System::Threading::AvSetMmThreadCharacteristicsW;
    let task: Vec<u16> = "Pro Audio".encode_utf16().chain(Some(0)).collect();
    let mut index = 0u32;
    unsafe { AvSetMmThreadCharacteristicsW(task.as_ptr(), &mut index) };
}

#[cfg(not(windows))]
pub fn pro_audio_thread() {}
