#[allow(unused_imports)]
use super::*;

use std::mem::size_of;

// Windows-specific imports
#[repr(C)]
#[allow(non_snake_case)]
#[allow(clippy::upper_case_acronyms)]
pub struct MEMORYSTATUSEX {
  pub dwLength: u32,
  pub dwMemoryLoad: u32,
  pub ullTotalPhys: u64,
  pub ullAvailPhys: u64,
  pub ullTotalPageFile: u64,
  pub ullAvailPageFile: u64,
  pub ullTotalVirtual: u64,
  pub ullAvailVirtual: u64,
  pub ullAvailExtendedVirtual: u64,
}

#[link(name = "kernel32")]
unsafe extern "system" {
  fn GlobalMemoryStatusEx(lpBuffer: *mut MEMORYSTATUSEX) -> i32;
}

pub fn get_memory_status() -> Option<MEMORYSTATUSEX> {
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

#[repr(C)]
#[allow(non_snake_case)]
pub struct PROCESS_MEMORY_COUNTERS {
  pub cb: u32,
  pub PageFaultCount: u32,
  pub PeakWorkingSetSize: usize,
  pub WorkingSetSize: usize,
  pub QuotaPeakPagedPoolUsage: usize,
  pub QuotaPagedPoolUsage: usize,
  pub QuotaPeakNonPagedPoolUsage: usize,
  pub QuotaNonPagedPoolUsage: usize,
  pub PagefileUsage: usize,
  pub PeakPagefileUsage: usize,
}

#[link(name = "psapi")]
unsafe extern "system" {
  fn GetProcessMemoryInfo(
    hProcess: isize,
    lpBuffer: *mut PROCESS_MEMORY_COUNTERS,
    cb: u32,
  ) -> i32;
}

pub fn get_process_memory_counters() -> Option<PROCESS_MEMORY_COUNTERS> {
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

#[repr(C)]
#[allow(non_snake_case)]
struct PROCESS_BASIC_INFORMATION {
  ExitStatus: i32,
  PebBaseAddress: isize,
  AffinityMask: isize,
  BasePriority: i32,
  UniqueProcessId: isize,
  InheritedFromUniqueProcessId: isize,
}

#[link(name = "ntdll")]
unsafe extern "system" {
  fn NtQueryInformationProcess(
    ProcessHandle: isize,
    ProcessInformationClass: u32,
    ProcessInformation: *mut PROCESS_BASIC_INFORMATION,
    ProcessInformationLength: u32,
    ReturnLength: *mut u32,
  ) -> i32;
}

pub fn getppid() -> isize {
  unsafe {
    let mut length = 0u32;
    let mut info = PROCESS_BASIC_INFORMATION {
      ExitStatus: 0,
      PebBaseAddress: 0,
      AffinityMask: 0,
      BasePriority: 0,
      UniqueProcessId: 0,
      InheritedFromUniqueProcessId: 0,
    };
    let pbi_length = size_of::<PROCESS_BASIC_INFORMATION>() as u32;

    let status = NtQueryInformationProcess(
      -1,
      0,
      &mut info as *mut PROCESS_BASIC_INFORMATION,
      pbi_length,
      &mut length as *mut u32,
    );
    if status >= 0 && length == pbi_length {
      info.InheritedFromUniqueProcessId
    } else {
      -1
    }
  }
}
