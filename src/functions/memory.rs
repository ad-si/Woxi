#[allow(unused_imports)]
use super::*;

#[cfg(target_os = "windows")]
use std::mem::size_of;

// Windows-specific imports
#[cfg(target_os = "windows")]
#[repr(C)]
#[allow(non_snake_case)]
struct MEMORYSTATUSEX {
  dwLength: u32,
  dwMemoryLoad: u32,
  ullTotalPhys: u64,
  ullAvailPhys: u64,
  ullTotalPageFile: u64,
  ullAvailPageFile: u64,
  ullTotalVirtual: u64,
  ullAvailVirtual: u64,
  ullAvailExtendedVirtual: u64,
}

#[cfg(target_os = "windows")]
#[link(name = "kernel32")]
unsafe extern "system" {
  fn GlobalMemoryStatusEx(lpBuffer: *mut MEMORYSTATUSEX) -> i32;
}

#[cfg(target_os = "windows")]
fn get_memory_status() -> Option<MEMORYSTATUSEX> {
  unsafe {
    let mut mem_status = MEMORYSTATUSEX {
      dwLength: size_of::<MEMORYSTATUSEX>() as u32,
      dwMemoryLoad: 0,
      ullTotalPhys: 0,
      ullAvailPhys: 0,
      ullTotalPageFile: 0,
      ullAvailPageFile: 0,
      ullTotalVirtual: 0,
      ullAvailVirtual: 0,
      ullAvailExtendedVirtual: 0,
    };

    if GlobalMemoryStatusEx(&mut mem_status as *mut MEMORYSTATUSEX) == 0 {
      return None;
    }

    Some(mem_status)
  }
}

#[cfg(target_os = "windows")]
#[repr(C)]
#[allow(non_snake_case)]
struct PROCESS_MEMORY_COUNTERS {
  cb: u32,
  PageFaultCount: u32,
  PeakWorkingSetSize: usize,
  WorkingSetSize: usize,
  QuotaPeakPagedPoolUsage: usize,
  QuotaPagedPoolUsage: usize,
  QuotaPeakNonPagedPoolUsage: usize,
  QuotaNonPagedPoolUsage: usize,
  PagefileUsage: usize,
  PeakPagefileUsage: usize,
}

#[cfg(target_os = "windows")]
#[link(name = "psapi")]
unsafe extern "system" {
  fn GetProcessMemoryInfo(
    hProcess: isize,
    lpBuffer: *mut PROCESS_MEMORY_COUNTERS,
    cb: u32,
  ) -> i32;
}

#[cfg(target_os = "windows")]
fn get_process_memory_counters() -> Option<PROCESS_MEMORY_COUNTERS> {
  unsafe {
    let cb = size_of::<PROCESS_MEMORY_COUNTERS>() as u32;
    let mut counters = PROCESS_MEMORY_COUNTERS {
      cb,
      PageFaultCount: 0,
      PeakWorkingSetSize: 0,
      WorkingSetSize: 0,
      QuotaPeakPagedPoolUsage: 0,
      QuotaPagedPoolUsage: 0,
      QuotaPeakNonPagedPoolUsage: 0,
      QuotaNonPagedPoolUsage: 0,
      PagefileUsage: 0,
      PeakPagefileUsage: 0,
    };

    if GetProcessMemoryInfo(
      -1,
      &mut counters as *mut PROCESS_MEMORY_COUNTERS,
      cb,
    ) == 0
    {
      return None;
    }

    Some(counters)
  }
}

#[cfg(target_os = "macos")]
fn macos_current_rss() -> i128 {
  use std::mem::MaybeUninit;
  unsafe {
    let mut info = MaybeUninit::<libc::mach_task_basic_info_data_t>::uninit();
    let mut count = (std::mem::size_of::<libc::mach_task_basic_info_data_t>()
      / std::mem::size_of::<libc::natural_t>())
      as libc::mach_msg_type_number_t;
    #[allow(deprecated)]
    let task = libc::mach_task_self();
    let kr = libc::task_info(
      task,
      libc::MACH_TASK_BASIC_INFO,
      info.as_mut_ptr() as libc::task_info_t,
      &mut count,
    );
    if kr == libc::KERN_SUCCESS {
      let info = info.assume_init();
      info.resident_size as i128
    } else {
      -1
    }
  }
}

pub fn memory_physical() -> i128 {
  #[cfg(target_os = "macos")]
  {
    // sysctlbyname("hw.memsize") returns total physical memory in bytes
    let mut size: u64 = 0;
    let mut len = std::mem::size_of::<u64>();
    let name = std::ffi::CString::new("hw.memsize").unwrap_or_default();
    let ret = unsafe {
      libc::sysctlbyname(
        name.as_ptr(),
        &mut size as *mut u64 as *mut libc::c_void,
        &mut len,
        std::ptr::null_mut(),
        0,
      )
    };
    if ret == 0 { size as i128 } else { -1 }
  }
  #[cfg(target_os = "linux")]
  {
    // Parse /proc/meminfo for MemTotal (kB) and convert to bytes
    let contents = std::fs::read_to_string("/proc/meminfo").unwrap_or_default();
    for line in contents.lines() {
      if let Some(rest) = line.strip_prefix("MemTotal:")
        && let Some(value) = rest.split_whitespace().next()
      {
        let kb: u64 = value.parse().unwrap_or_default();
        return (kb * 1024) as i128;
      }
    }
    -1
  }
  #[cfg(target_os = "windows")]
  {
    let result = get_memory_status();
    if let Some(status) = result {
      return status.ullTotalPhys as i128;
    }
    -1
  }

  #[cfg(not(any(
    target_os = "macos",
    target_os = "linux",
    target_os = "windows"
  )))]
  {
    -1
  }
}

/// Returns the peak memory usage of the current process in bytes.
pub fn max_memory_used() -> i128 {
  #[cfg(target_os = "linux")]
  {
    std::fs::read_to_string("/proc/self/status")
      .ok()
      .and_then(|s| {
        s.lines()
          .find(|l| l.starts_with("VmPeak:"))
          .and_then(|l| l.split_whitespace().nth(1))
          .and_then(|v| v.parse::<i128>().ok())
          .map(|kb| kb * 1024)
      })
      .unwrap_or(-1)
  }

  #[cfg(target_os = "macos")]
  {
    use std::mem::MaybeUninit;
    // Use getrusage ru_maxrss for peak memory (kernel-maintained high watermark).
    // Take the max with current RSS to ensure consistency.
    let ru_max = unsafe {
      let mut usage = MaybeUninit::<libc::rusage>::uninit();
      if libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) == 0 {
        usage.assume_init().ru_maxrss as i128
      } else {
        -1
      }
    };
    let current = macos_current_rss();
    std::cmp::max(ru_max, current)
  }

  #[cfg(target_os = "windows")]
  {
    let result = get_process_memory_counters();
    if let Some(counters) = result {
      return counters.PeakWorkingSetSize as i128;
    }
    -1
  }

  #[cfg(not(any(
    target_os = "macos",
    target_os = "linux",
    target_os = "windows"
  )))]
  {
    -1
  }
}

/// Returns an estimate of the free system memory in bytes, or -1 if
/// the platform-specific query fails.
pub fn memory_available() -> i128 {
  #[cfg(target_os = "linux")]
  {
    if let Ok(contents) = std::fs::read_to_string("/proc/meminfo") {
      for line in contents.lines() {
        if let Some(rest) = line.strip_prefix("MemAvailable:")
          && let Some(tok) = rest.split_whitespace().next()
          && let Ok(kb) = tok.parse::<i128>()
        {
          return kb * 1024;
        }
      }
    }
    -1
  }

  #[cfg(target_os = "macos")]
  {
    use std::mem::MaybeUninit;
    // vm_statistics64 via host_statistics64 returns page counts for free,
    // active, inactive, wired and speculative memory. Consider free +
    // inactive + speculative as "available".
    let mut stats = MaybeUninit::<libc::vm_statistics64>::uninit();
    let mut count: libc::mach_msg_type_number_t =
      (std::mem::size_of::<libc::vm_statistics64>()
        / std::mem::size_of::<libc::integer_t>())
        as libc::mach_msg_type_number_t;
    #[allow(deprecated)]
    let host = unsafe { libc::mach_host_self() };
    let kr = unsafe {
      libc::host_statistics64(
        host,
        libc::HOST_VM_INFO64,
        stats.as_mut_ptr() as *mut libc::integer_t,
        &mut count,
      )
    };
    if kr != libc::KERN_SUCCESS {
      return -1;
    }
    let stats = unsafe { stats.assume_init() };
    let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as i128;
    if page_size <= 0 {
      return -1;
    }
    let free = stats.free_count as i128;
    let inactive = stats.inactive_count as i128;
    let speculative = stats.speculative_count as i128;
    (free + inactive + speculative) * page_size
  }

  #[cfg(target_os = "windows")]
  {
    let result = get_memory_status();
    if let Some(status) = result {
      return status.ullAvailPhys as i128;
    }
    -1
  }

  #[cfg(not(any(
    target_os = "macos",
    target_os = "linux",
    target_os = "windows"
  )))]
  {
    -1
  }
}

/// Returns the current memory usage (RSS) of the process in bytes.
pub fn memory_in_use() -> i128 {
  #[cfg(target_os = "linux")]
  {
    std::fs::read_to_string("/proc/self/status")
      .ok()
      .and_then(|s| {
        s.lines()
          .find(|l| l.starts_with("VmRSS:"))
          .and_then(|l| l.split_whitespace().nth(1))
          .and_then(|v| v.parse::<i128>().ok())
          .map(|kb| kb * 1024)
      })
      .unwrap_or(-1)
  }

  #[cfg(target_os = "macos")]
  {
    macos_current_rss()
  }

  #[cfg(target_os = "windows")]
  {
    let result = get_process_memory_counters();
    if let Some(counters) = result {
      return counters.WorkingSetSize as i128;
    }
    -1
  }

  #[cfg(not(any(
    target_os = "macos",
    target_os = "linux",
    target_os = "windows"
  )))]
  {
    -1
  }
}

/// Returns the total CPU time used by the current process in seconds
/// (user + system). Falls back to 0.0 on platforms without `getrusage`.
pub fn cpu_time_used() -> f64 {
  #[cfg(unix)]
  {
    use std::mem::MaybeUninit;
    unsafe {
      let mut usage = MaybeUninit::<libc::rusage>::uninit();
      if libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) == 0 {
        let u = usage.assume_init();
        let user =
          u.ru_utime.tv_sec as f64 + (u.ru_utime.tv_usec as f64) * 1e-6;
        let sys = u.ru_stime.tv_sec as f64 + (u.ru_stime.tv_usec as f64) * 1e-6;
        return user + sys;
      }
    }
    0.0
  }
  #[cfg(not(unix))]
  {
    0.0
  }
}
