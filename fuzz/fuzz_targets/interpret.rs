//! Fuzz target for the full interpreter pipeline (parse + evaluate).
//!
//! Invariant: `woxi::interpret` must return `Ok` or a proper
//! `InterpreterError` for every input — it must never panic, abort, or
//! overflow the stack. Hangs are caught by libFuzzer's `-timeout` flag
//! (see the `fuzz-interpret` make target).

#![no_main]

use libfuzzer_sys::fuzz_target;

/// Heads that touch the filesystem, network, or environment. Inputs
/// containing them are skipped so the fuzzer neither litters the disk nor
/// mutates towards I/O instead of interpreter logic.
const SIDE_EFFECT_DENYLIST: &[&str] = &[
  "Export",
  "Import",
  "OpenWrite",
  "OpenAppend",
  "OpenRead",
  "Put",
  "Get",
  "DeleteFile",
  "DeleteDirectory",
  "CreateFile",
  "CreateDirectory",
  "RenameFile",
  "CopyFile",
  "SetDirectory",
  "Run",
  "URLFetch",
  "URLRead",
  "URLDownload",
  "Install",
  "Pause",
  "Environment",
];

fuzz_target!(|data: &[u8]| {
  if data.len() > 2048 {
    return;
  }
  let Ok(input) = std::str::from_utf8(data) else {
    return;
  };
  if SIDE_EFFECT_DENYLIST.iter().any(|head| input.contains(head)) {
    return;
  }
  // Suppress Print/echo output — libFuzzer treats stdout noise as slowdown
  // and the output is meaningless for crash detection.
  woxi::set_quiet_print(true);
  let _ = woxi::interpret(input);
});
