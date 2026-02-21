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
      0
    }
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
      .unwrap_or(0)
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
        0
      }
    };
    let current = macos_current_rss();
    std::cmp::max(ru_max, current)
  }

  #[cfg(not(any(target_os = "linux", target_os = "macos")))]
  {
    0
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
      .unwrap_or(0)
  }

  #[cfg(target_os = "macos")]
  {
    macos_current_rss()
  }

  #[cfg(not(any(target_os = "linux", target_os = "macos")))]
  {
    0
  }
}
