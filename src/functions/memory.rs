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

/// Returns an estimate of the free system memory in bytes, or -1 if
/// the platform-specific query fails.
pub fn memory_available() -> i128 {
  #[cfg(target_os = "linux")]
  {
    if let Ok(contents) = std::fs::read_to_string("/proc/meminfo") {
      for line in contents.lines() {
        if let Some(rest) = line.strip_prefix("MemAvailable:") {
          if let Some(tok) = rest.split_whitespace().next() {
            if let Ok(kb) = tok.parse::<i128>() {
              return kb * 1024;
            }
          }
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

  #[cfg(not(any(target_os = "linux", target_os = "macos")))]
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
