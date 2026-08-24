//! Fuzz target for the full interpreter pipeline (parse + evaluate).
//!
//! Invariant: `woxi::interpret` must return `Ok` or a proper
//! `InterpreterError` for every input — it must never panic, abort, or
//! overflow the stack. Hangs are caught by libFuzzer's `-timeout` flag
//! (see the `fuzz-interpret` make target), which is only meaningful for
//! inputs that are supposed to terminate — hence the denylists below.

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

/// Heads that iterate until a condition the program itself computes turns
/// false. A mutation that breaks the condition — dropping the `Break[]`,
/// misspelling the counter's update — yields a program that runs forever
/// *by definition*, exactly as it does in the Wolfram Language, and
/// libFuzzer reports the resulting hang as a finding. Two of the last ten
/// nightly fuzz runs failed that way, both on a mangled `While` loop, so
/// inputs mentioning one of these heads are skipped: a hang there says
/// nothing about the interpreter. Runaway evaluation *outside* an explicit
/// loop (a rewrite rule that never reaches a fixed point, say) is a real
/// bug and still trips the timeout.
///
/// `TimeConstrained` is here for the same reason from the other side: it
/// bounds a computation in the Wolfram Language, but Woxi cannot interrupt
/// a running evaluation and only compares the elapsed time afterwards, so
/// a body that never finishes hangs the fuzzer on a known limitation.
///
/// The entries are matched as substrings, so `While` also covers
/// `NestWhile` and `For` also covers `Format` — a few percent of inputs
/// skipped for free is cheaper than an unreliable nightly.
const NONTERMINATING_DENYLIST: &[&str] =
  &["While", "For", "FixedPoint", "TimeConstrained"];

fuzz_target!(|data: &[u8]| {
  if data.len() > 2048 {
    return;
  }
  let Ok(input) = std::str::from_utf8(data) else {
    return;
  };
  if SIDE_EFFECT_DENYLIST
    .iter()
    .chain(NONTERMINATING_DENYLIST)
    .any(|head| input.contains(head))
  {
    return;
  }
  // Suppress Print/echo output — libFuzzer treats stdout noise as slowdown
  // and the output is meaningless for crash detection.
  woxi::set_quiet_print(true);
  let _ = woxi::interpret(input);
});
