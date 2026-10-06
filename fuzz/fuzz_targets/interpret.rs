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
///
/// `Socket` (matched as a substring, so it covers `SocketOpen`,
/// `SocketListen`, `SocketConnect`, `SocketReadMessage`, …) is here
/// because a read blocks until the peer writes: a mutation that misspells
/// the socket a request is written to leaves `SocketReadMessage` waiting
/// forever, exactly as it does in the Wolfram Language.
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
  "Socket",
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

/// Heads that iterate a user function a written number of times. Such a
/// program terminates, but the fuzzer mutates the seed's *data*, and an
/// iterate that grows each step turns that count into an exponent: in
/// `tests/scripts/dragon_curve.wls` the line `{a, b}` becomes `{aLog, b}`,
/// and the 11 `Nest` steps that fold a numeric curve in 0.08 s now build a
/// symbolic one of 250 million leaves — wolframscript needs 12 s and 8 GB of
/// unshared `ByteCount` for it, which under ASan is far past both the
/// timeout and `-rss_limit_mb`. The finding says nothing about the
/// interpreter, so inputs mentioning `Nest` (and through it `NestList`,
/// `NestGraph`, `NestTree`) are skipped like the loop heads above.
const EXPONENTIAL_DENYLIST: &[&str] = &["Nest"];

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
    .chain(EXPONENTIAL_DENYLIST)
    .any(|head| input.contains(head))
  {
    return;
  }
  // Definitions, memoized values and system variables live in thread-local
  // state that outlives one `interpret` call, so without this an input's
  // behaviour depends on every input libFuzzer happened to run before it:
  // findings stop reproducing from the artifact alone, and a leftover
  // definition can make a cheap program pathological — a memoized
  // `f[n_] := f[n] = …` that still meets an older, unmemoized definition of
  // the same symbol recurses exponentially. libFuzzer requires the target
  // to be a pure function of its input; this is what makes it one.
  woxi::clear_state();
  // Suppress Print/echo output — libFuzzer treats stdout noise as slowdown
  // and the output is meaningless for crash detection.
  woxi::set_quiet_print(true);
  let _ = woxi::interpret(input);
});
