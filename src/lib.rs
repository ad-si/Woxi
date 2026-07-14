use pest::Parser;
use pest::iterators::Pair;
use pest_derive::Parser;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use thiserror::Error;

pub mod evaluator;
pub mod expr_list;
pub mod functions;
pub mod notebook;
pub mod syntax;
pub mod utils;
#[cfg(target_arch = "wasm32")]
pub mod wasm;

pub use expr_list::ExprList;

#[derive(Parser)]
#[grammar = "wolfram.pest"]
pub struct WolframParser;

#[derive(Clone)]
enum StoredValue {
  Association(Vec<(String, String)>),
  Raw(String),           // keep evaluated textual value
  ExprVal(syntax::Expr), // keep as structured AST for fast Part access
}
thread_local! {
    static ENV: RefCell<HashMap<String, StoredValue>> = RefCell::new(HashMap::new());
    //            name         Vec of (param_names, conditions, defaults, head_constraints, blank_types, body_AST) for multi-arity + condition + optional support
    //            blank_types: 1=Blank, 2=BlankSequence, 3=BlankNullSequence
    static FUNC_DEFS: RefCell<HashMap<String, Vec<(Vec<String>, Vec<Option<syntax::Expr>>, Vec<Option<syntax::Expr>>, Vec<Option<String>>, Vec<u8>, syntax::Expr)>>> = RefCell::new(HashMap::new());
    // Memoized literal-argument definitions (the `f[x_] := f[x] = …` idiom):
    // function name -> (joined argument key -> (arg Exprs, stored value)).
    // Kept separate from FUNC_DEFS so dispatch is O(1) per memoized value
    // instead of linearly scanning thousands of accumulated literal DownValues
    // (which made memoization O(n²)). The arg Exprs are retained so the
    // entries can be reconstructed as rules for DownValues introspection.
    pub static MEMO_VALUES: RefCell<HashMap<String, HashMap<String, (Vec<syntax::Expr>, syntax::Expr)>>> = RefCell::new(HashMap::new());
    // Function attributes (e.g., Listable, Flat, etc.)
    static FUNC_ATTRS: RefCell<HashMap<String, Vec<String>>> = RefCell::new(HashMap::new());
    // Builtin attributes that have been explicitly removed (via Unprotect,
    // ClearAttributes, or ClearAll). Subtracted from `Attributes[sym]` so
    // that e.g. `Unprotect[Pi]; Attributes[Pi]` no longer contains
    // `Protected`. Re-adding via SetAttributes/Protect prunes the entry.
    pub static FUNC_ATTRS_REMOVED: RefCell<HashMap<String, Vec<String>>> = RefCell::new(HashMap::new());
    // Function options (e.g., Options[f] = {a -> 1})
    pub static FUNC_OPTIONS: RefCell<HashMap<String, Vec<syntax::Expr>>> = RefCell::new(HashMap::new());
    // Track whether Options[f] was set with `:=` (SetDelayed) so Definition can
    // re-emit the matching operator. Symbols set with plain `=` are absent.
    pub static FUNC_OPTIONS_DELAYED: RefCell<std::collections::HashSet<String>> = RefCell::new(std::collections::HashSet::new());
    // UpValues: tag symbol -> Vec of (outer_func_name, params, conditions, defaults, heads, body, original_lhs, original_body)
    // Stored by tag symbol; checked during evaluation when a function's arguments contain the tag as head
    pub static UPVALUES: RefCell<HashMap<String, Vec<(String, Vec<String>, Vec<Option<syntax::Expr>>, Vec<Option<syntax::Expr>>, Vec<Option<String>>, syntax::Expr, syntax::Expr, syntax::Expr)>>> = RefCell::new(HashMap::new());
    // Track Part evaluation nesting depth for Part::partd warnings
    static PART_DEPTH: RefCell<usize> = const { RefCell::new(0) };
    // Track evaluation recursion depth for $RecursionLimit enforcement
    pub static RECURSION_DEPTH: Cell<usize> = const { Cell::new(0) };
    // Reap/Sow stack: each Reap call pushes a Vec to collect (value, tag) pairs
    pub static SOW_STACK: RefCell<Vec<Vec<(syntax::Expr, syntax::Expr)>>> = const { RefCell::new(Vec::new()) };
    // Context stack for Begin/End: stores the context strings pushed by Begin[]
    static CONTEXT_STACK: RefCell<Vec<String>> = const { RefCell::new(Vec::new()) };
    // Stack of saved $ContextPath values, pushed by BeginPackage[] and
    // popped by EndPackage[]. The top entry (when present) is the
    // currently-active path; an empty stack falls back to the System`/Global`
    // default.
    static CONTEXT_PATH_STACK: RefCell<Vec<Vec<String>>> = const { RefCell::new(Vec::new()) };
    // Extra packages registered by `BeginPackage[]` (and the explicit second
    // argument). `$Packages` returns these prepended to the canonical
    // `{"System`", "Global`"}` baseline.
    pub static PACKAGES_EXTRA: RefCell<Vec<String>> = const { RefCell::new(Vec::new()) };
    // Current option bindings for OptionValue during function evaluation
    // Stack of (function_name, Vec<(option_name, option_value)>)
    pub static OPTION_VALUE_CONTEXT: RefCell<Vec<(String, Vec<(String, syntax::Expr)>)>> = const { RefCell::new(Vec::new()) };
    // Inline OptionsPattern defaults: func_name -> Vec of Option<Vec<Expr>> per overload
    // When OptionsPattern[{a -> a0, ...}] is used, stores the inline defaults for that overload
    pub static FUNC_OPTS_INLINE: RefCell<HashMap<String, Vec<Option<Vec<syntax::Expr>>>>> = RefCell::new(HashMap::new());
    // Evaluation stack: tracks the chain of function calls currently being evaluated.
    // Used to produce stack traces when evaluation errors/messages occur.
    pub static EVAL_STACK: RefCell<Vec<String>> = const { RefCell::new(Vec::new()) };
    // Last captured stack trace: set when an EvaluationError propagates through
    // the function call evaluation, so the top-level handler can print it.
    pub static LAST_ERROR_TRACE: RefCell<Option<String>> = const { RefCell::new(None) };
}

#[derive(Error, Debug)]
pub enum InterpreterError {
  #[error("Parse error: {0}")]
  ParseError(#[from] Box<pest::error::Error<Rule>>),
  #[error("Empty input")]
  EmptyInput,
  #[error("Evaluation error: {0}")]
  EvaluationError(String),
  #[error("Return")]
  ReturnValue(Box<syntax::Expr>),
  #[error("Break")]
  BreakSignal,
  #[error("Continue")]
  ContinueSignal,
  #[error("Throw")]
  ThrowValue(Box<syntax::Expr>, Option<Box<syntax::Expr>>),
  #[error("$Aborted")]
  Abort,
  /// Internal signal for tail-call optimization (never user-visible)
  #[error("TailCall")]
  TailCall(Box<syntax::Expr>),
  /// Internal signal for Goto[tag] — caught by CompoundExpression
  #[error("Goto")]
  GotoSignal(Box<syntax::Expr>),
}

/// A playable audio output captured during evaluation. Visual hosts (the
/// Woxi Playground and Woxi Studio) render it as a graphical audio player.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AudioOutput {
  /// Base64-encoded audio data. Empty when the data is unavailable (e.g. a
  /// file-backed `Audio` whose file cannot be read, as in the browser
  /// playground) — hosts still render the player chrome, it just cannot play.
  pub base64: String,
  /// MIME type of the encoded data (e.g. "audio/wav", "audio/flac").
  pub mime: String,
  /// Display label — the file name for file-backed `Audio` objects.
  pub label: Option<String>,
}

/// Extended result type that includes both stdout and the result
#[derive(Debug, Clone)]
pub struct InterpretResult {
  pub stdout: String,
  pub result: String,
  pub graphics: Option<String>,
  pub output_svg: Option<String>,
  /// Playable audio (synthesized from Play/Sound, or an Audio object), if any.
  pub sound: Option<AudioOutput>,
  pub warnings: Vec<String>,
}

impl WolframParser {
  pub fn parse_wolfram(
    input: &str,
  ) -> Result<pest::iterators::Pairs<'_, Rule>, Box<pest::error::Error<Rule>>>
  {
    // Deeply nested invalid input (e.g. `f[f[f[...` without closing
    // brackets) makes pest backtrack exponentially, so rejecting it would
    // take hours. The call limit turns that into a "call limit reached"
    // parse error. The worst script in tests/scripts needs ~5,700 calls
    // per byte, so 20,000 per byte (plus a base allowance covering the
    // fixed overhead on tiny inputs) leaves ample headroom for legitimate
    // code of any size, while short pathological inputs are rejected in
    // well under a second even in debug builds on slow CI hardware.
    let limit =
      1_000_000_usize.saturating_add(input.len().saturating_mul(20_000));
    pest::set_call_limit(std::num::NonZeroUsize::new(limit));
    Self::parse(Rule::Program, input).map_err(Box::new)
  }
}

pub fn parse(
  input: &str,
) -> Result<pest::iterators::Pairs<'_, Rule>, Box<pest::error::Error<Rule>>> {
  WolframParser::parse_wolfram(input)
}

// Global RNG state: None = use thread_rng(), Some = use seeded ChaCha8Rng
thread_local! {
    static SEEDED_RNG: RefCell<Option<ChaCha8Rng>> = const { RefCell::new(None) };
}

/// Seed the global RNG with a specific seed value (SeedRandom[n]).
pub fn seed_rng(seed: u64) {
  SEEDED_RNG.with(|rng| {
    *rng.borrow_mut() = Some(ChaCha8Rng::seed_from_u64(seed));
  });
}

/// Reset the global RNG to non-deterministic mode (SeedRandom[]).
pub fn unseed_rng() {
  SEEDED_RNG.with(|rng| {
    *rng.borrow_mut() = None;
  });
}

/// Execute a closure with a mutable reference to the current RNG.
/// Uses the seeded RNG if set, otherwise falls back to thread_rng().
pub fn with_rng<F, R>(f: F) -> R
where
  F: FnOnce(&mut dyn rand::RngCore) -> R,
{
  SEEDED_RNG.with(|cell| {
    let mut borrow = cell.borrow_mut();
    if let Some(ref mut seeded) = *borrow {
      f(seeded)
    } else {
      f(&mut rand::thread_rng())
    }
  })
}

// Captured output from Print statements
thread_local! {
    static CAPTURED_STDOUT: RefCell<String> = const { RefCell::new(String::new()) };
}

// Quiet print mode — suppresses Print's stdout output while still
// capturing to the internal buffer. Used by conformance tests to
// separate Print side-effects from expression results.
thread_local! {
    static QUIET_PRINT: RefCell<bool> = const { RefCell::new(false) };
}

/// Enable/disable quiet print mode (suppresses Print's stdout output).
pub fn set_quiet_print(enabled: bool) {
  QUIET_PRINT.with(|q| *q.borrow_mut() = enabled);
}

/// Check if quiet print mode is enabled.
pub fn is_quiet_print() -> bool {
  QUIET_PRINT.with(|q| *q.borrow())
}

// Visual display mode flag — set by interpret_with_stdout to enable
// rendering of display wrappers like TableForm as SVG grids
thread_local! {
    static VISUAL_MODE: RefCell<bool> = const { RefCell::new(false) };
}

/// Whether the interpreter is in visual (notebook-like) display mode, where
/// front-end-only behaviors apply (e.g. `Defer[expr]` shows `expr` without
/// its wrapper). False in plain CLI/script mode, which matches wolframscript.
pub fn is_visual_mode() -> bool {
  VISUAL_MODE.with(|v| *v.borrow())
}

// REPL session flag — set by the `woxi repl` command. Unlike VISUAL_MODE it
// does not change result rendering (the terminal REPL keeps plain CLI/text
// output to match wolframscript), but it does enable persistent `%` / `Out[]`
// history caching across evaluations in the same process.
thread_local! {
    static REPL_MODE: RefCell<bool> = const { RefCell::new(false) };
}

/// Enable or disable REPL session mode (persistent `%` / `Out[]` history).
pub fn set_repl_mode(enabled: bool) {
  REPL_MODE.with(|v| *v.borrow_mut() = enabled);
}

/// Whether output history (`%` / `Out[]`) should persist across evaluations.
/// True in both visual (notebook) mode and terminal REPL mode.
fn output_history_enabled() -> bool {
  VISUAL_MODE.with(|v| *v.borrow()) || REPL_MODE.with(|v| *v.borrow())
}

// Dark mode flag — when true, SVG output uses a dark color palette
thread_local! {
    static DARK_MODE: RefCell<bool> = const { RefCell::new(false) };
}

pub fn is_dark_mode() -> bool {
  DARK_MODE.with(|d| *d.borrow())
}

pub fn set_dark_mode(enabled: bool) {
  DARK_MODE.with(|d| *d.borrow_mut() = enabled);
}

// Captured graphical output (SVG) from Plot and related functions
thread_local! {
    static CAPTURED_GRAPHICS: RefCell<Vec<String>> = const { RefCell::new(Vec::new()) };
}

// Captured GraphicsBox expression (Mathematica .nb format) from Graphics/Plot
thread_local! {
    static CAPTURED_GRAPHICSBOX: RefCell<Option<String>> = const { RefCell::new(None) };
}

// Captured SVG rendering of the text output (always generated, with superscripts etc.)
thread_local! {
    static CAPTURED_OUTPUT_SVG: RefCell<Option<String>> = const { RefCell::new(None) };
}

// Captured playable audio (from Play/Sound synthesis or an Audio object).
// The visual hosts turn this into a graphical audio player.
thread_local! {
    static CAPTURED_SOUND: RefCell<Option<AudioOutput>> = const { RefCell::new(None) };
}

// Most-recent evaluated expression for `%` / `Out[]` shortcuts. Set by
// `interpret_with_stdout` after each successful evaluation and consulted by
// the `Out[]` dispatch path. Independent of `$Line`: we only retain the
// latest value (mirroring what woxi-studio actually needs for cells like
// `N[%]` that reference the previous cell's result).
thread_local! {
    static LAST_OUTPUT_EXPR: RefCell<Option<syntax::Expr>> = const { RefCell::new(None) };
}

/// Stash an evaluated expression so subsequent `%` / `Out[]` references
/// can resolve to it. Called from `interpret_with_stdout` after success.
fn set_last_output(expr: syntax::Expr) {
  LAST_OUTPUT_EXPR.with(|c| *c.borrow_mut() = Some(expr));
}

/// Retrieve the most recent stashed expression, if any.
pub fn get_last_output() -> Option<syntax::Expr> {
  LAST_OUTPUT_EXPR.with(|c| c.borrow().clone())
}

/// Drop the cached previous output — used when the studio resets state
/// before re-evaluating the whole notebook.
pub fn clear_last_output() {
  LAST_OUTPUT_EXPR.with(|c| *c.borrow_mut() = None);
}

// Session start time for SessionTime[]
static SESSION_START: std::sync::LazyLock<web_time::Instant> =
  std::sync::LazyLock::new(web_time::Instant::now);

/// Returns the elapsed time in seconds since the session started.
pub fn session_time() -> f64 {
  SESSION_START.elapsed().as_secs_f64()
}

/// Look up the value stored in ENV for a name and return it as an Expr.
/// Used by introspection functions like OwnValues. Returns None if the name
/// has no stored value.
pub fn lookup_env_as_expr(name: &str) -> Option<syntax::Expr> {
  ENV.with(|e| {
    e.borrow().get(name).map(|v| match v {
      StoredValue::ExprVal(expr) => expr.clone(),
      StoredValue::Raw(s) => match interpret_to_expr(s) {
        Ok(expr) => expr,
        Err(_) => syntax::Expr::String(s.clone()),
      },
      StoredValue::Association(_) => syntax::Expr::Identifier(name.to_string()),
    })
  })
}

// Captured unimplemented function calls (e.g. "Quantity[13.77, \"BillionYears\"]")
thread_local! {
    static UNIMPLEMENTED_CALLS: RefCell<Vec<String>> = const { RefCell::new(Vec::new()) };
}

// Captured warnings (general-purpose, e.g. deprecation notices)
thread_local! {
    static CAPTURED_WARNINGS: RefCell<Vec<String>> = const { RefCell::new(Vec::new()) };
}

// Captured Wolfram-style messages (printed inline, tracked for Check/Quiet interaction)
thread_local! {
    static CAPTURED_MESSAGES: RefCell<Vec<String>> = const { RefCell::new(Vec::new()) };
    /// Per-calculation display counts by message name (Symbol::tag), for
    /// wolframscript's General::stop suppression of repeated messages.
    static MESSAGE_STOP_COUNTS: RefCell<std::collections::HashMap<String, usize>> =
      RefCell::new(std::collections::HashMap::new());
}

// Set of message tags suppressed via `Off[head::tag]`. Keys are formatted as
// "Head::tag" (e.g. "Power::indet"). When a message about to be emitted starts
// with one of these tags, it is silently dropped instead of being printed.
thread_local! {
    static OFF_MESSAGES: RefCell<std::collections::HashSet<String>> =
        RefCell::new(std::collections::HashSet::new());
}

/// Mark a message tag as off, so future `emit_message` calls whose text starts
/// with `"<tag>: "` are dropped. Used to implement `Off[Head::tag]`.
pub fn off_message(tag: &str) {
  OFF_MESSAGES.with(|s| {
    s.borrow_mut().insert(tag.to_string());
  });
}

/// Re-enable a previously suppressed message tag (mirror of `off_message`).
pub fn on_message(tag: &str) {
  OFF_MESSAGES.with(|s| {
    s.borrow_mut().remove(tag);
  });
}

fn message_is_off(msg: &str) -> bool {
  // Scan each line for a `Head::tag: ` prefix and check whether any matches
  // a suppressed tag. We can't just take msg[..find(": ")] because some
  // messages prepend a right-aligned context line (e.g. the exponent above
  // the base in Power::indet) before the tag line.
  OFF_MESSAGES.with(|set| {
    let set = set.borrow();
    if set.is_empty() {
      return false;
    }
    for line in msg.lines() {
      if let Some(colon_space) = line.find(": ")
        && let Some(double_colon) = line[..colon_space].find("::")
      {
        let tag = &line[..colon_space];
        // Validate `tag` looks like Head::tag (no spaces).
        if !tag.contains(' ') && double_colon > 0 && set.contains(tag) {
          return true;
        }
      }
    }
    false
  })
}

// Directory of the notebook the current evaluation is running in. Front-ends
// (e.g. woxi-studio) set this to the parent directory of the loaded `.nb`
// file so `NotebookDirectory[]` returns a real path; when unset, the function
// stays unevaluated to match wolframscript's behavior in non-FrontEnd mode.
thread_local! {
    static NOTEBOOK_DIRECTORY: RefCell<Option<String>> = const { RefCell::new(None) };
}

/// Set the directory returned by `NotebookDirectory[]` for the current
/// thread. Pass `None` to clear it.
pub fn set_notebook_directory(dir: Option<String>) {
  NOTEBOOK_DIRECTORY.with(|d| *d.borrow_mut() = dir);
}

/// Get the directory configured via `set_notebook_directory`, if any.
pub fn get_notebook_directory() -> Option<String> {
  NOTEBOOK_DIRECTORY.with(|d| d.borrow().clone())
}

// When true, emit_message prints to stdout instead of stderr (matching wolframscript behavior)
thread_local! {
    static MESSAGES_TO_STDOUT: RefCell<bool> = const { RefCell::new(false) };
}

/// Enable/disable routing messages to stdout (for eval mode, matching wolframscript).
pub fn set_messages_to_stdout(enabled: bool) {
  MESSAGES_TO_STDOUT.with(|f| *f.borrow_mut() = enabled);
}

// Quiet level: when > 0, message printing is suppressed
thread_local! {
    static QUIET_LEVEL: RefCell<usize> = const { RefCell::new(0) };
}

// Stream registry for open streams (InputStream/OutputStream)
#[derive(Clone, Debug)]
pub enum StreamKind {
  StringStream(String), // content of the string
  FileStream(String),   // file path
}

#[derive(Clone, Debug)]
pub struct OpenStream {
  pub name: String,
  pub kind: StreamKind,
  pub id: usize,
  pub position: usize,
}

thread_local! {
    static STREAM_REGISTRY: RefCell<HashMap<usize, OpenStream>> = RefCell::new(HashMap::new());
    static STREAM_COUNTER: RefCell<usize> = const { RefCell::new(1) };
}

/// Register a new open stream and return its ID
pub fn register_stream(name: String, kind: StreamKind) -> usize {
  let id = STREAM_COUNTER.with(|c| {
    let mut counter = c.borrow_mut();
    let id = *counter;
    *counter += 1;
    id
  });
  let stream = OpenStream {
    name,
    kind,
    id,
    position: 0,
  };
  STREAM_REGISTRY.with(|reg| {
    reg.borrow_mut().insert(id, stream);
  });
  id
}

/// Close a stream by ID, returning the stream name and kind if it was open
pub fn close_stream(id: usize) -> Option<(String, StreamKind)> {
  STREAM_REGISTRY
    .with(|reg| reg.borrow_mut().remove(&id).map(|s| (s.name, s.kind)))
}

/// Check if a stream is open
pub fn is_stream_open(id: usize) -> bool {
  STREAM_REGISTRY.with(|reg| reg.borrow().contains_key(&id))
}

/// Get the remaining content of a stream (for reading)
pub fn get_stream_content(id: usize) -> Option<(String, usize)> {
  STREAM_REGISTRY.with(|reg| {
    let registry = reg.borrow();
    registry.get(&id).map(|s| {
      let content = match &s.kind {
        StreamKind::StringStream(text) => text.clone(),
        StreamKind::FileStream(path) => {
          std::fs::read_to_string(path).unwrap_or_default()
        }
      };
      (content, s.position)
    })
  })
}

/// Get the current read position of a stream
pub fn get_stream_position(id: usize) -> Option<usize> {
  STREAM_REGISTRY.with(|reg| reg.borrow().get(&id).map(|s| s.position))
}

/// Set the read position of a stream to an absolute position
pub fn set_stream_position(id: usize, new_position: usize) {
  STREAM_REGISTRY.with(|reg| {
    let mut registry = reg.borrow_mut();
    if let Some(stream) = registry.get_mut(&id) {
      stream.position = new_position;
    }
  });
}

/// Advance the read position of a stream
pub fn advance_stream_position(id: usize, new_position: usize) {
  STREAM_REGISTRY.with(|reg| {
    let mut registry = reg.borrow_mut();
    if let Some(stream) = registry.get_mut(&id) {
      stream.position = new_position;
    }
  });
}

/// Clears the captured stdout buffer
fn clear_captured_stdout() {
  CAPTURED_STDOUT.with(|buffer| {
    buffer.borrow_mut().clear();
  });
}

/// Appends to the captured stdout buffer
fn capture_stdout(text: &str) {
  CAPTURED_STDOUT.with(|buffer| {
    buffer.borrow_mut().push_str(text);
    buffer.borrow_mut().push('\n');
  });
}

/// Appends to the captured stdout buffer without a trailing newline.
pub fn capture_stdout_raw(text: &str) {
  CAPTURED_STDOUT.with(|buffer| {
    buffer.borrow_mut().push_str(text);
  });
}

/// Gets the captured stdout content
fn get_captured_stdout() -> String {
  CAPTURED_STDOUT.with(|buffer| buffer.borrow().clone())
}

/// Clears the captured warnings, messages, and unimplemented calls buffers
fn clear_captured_warnings() {
  UNIMPLEMENTED_CALLS.with(|buffer| {
    buffer.borrow_mut().clear();
  });
  CAPTURED_WARNINGS.with(|buffer| {
    buffer.borrow_mut().clear();
  });
  CAPTURED_MESSAGES.with(|buffer| {
    buffer.borrow_mut().clear();
  });
}

/// Records a call to an unimplemented built-in function (e.g. "Quantity[13.77, \"BillionYears\"]")
pub fn capture_unimplemented_call(call_str: &str) {
  UNIMPLEMENTED_CALLS.with(|buffer| {
    buffer.borrow_mut().push(call_str.to_string());
  });
}

/// Appends a warning message
pub fn capture_warning(text: &str) {
  CAPTURED_WARNINGS.with(|buffer| {
    buffer.borrow_mut().push(text.to_string());
  });
}

/// Gets the captured warnings, consolidating unimplemented function calls into a single message.
/// Includes both general warnings and Wolfram-style messages.
pub fn get_captured_warnings() -> Vec<String> {
  let mut warnings = Vec::new();

  let calls = UNIMPLEMENTED_CALLS.with(|buffer| buffer.borrow().clone());
  if !calls.is_empty() {
    let joined = calls.join(", ");
    let verb = if calls.len() == 1 {
      "is a built-in Wolfram Language function"
    } else {
      "are built-in Wolfram Language functions"
    };
    warnings.push(format!("{} {} not yet implemented in Woxi.", joined, verb));
  }

  CAPTURED_WARNINGS.with(|buffer| {
    warnings.extend(buffer.borrow().clone());
  });

  // Include messages (used by Check[] to detect message generation)
  CAPTURED_MESSAGES.with(|buffer| {
    warnings.extend(buffer.borrow().clone());
  });

  warnings
}

/// Gets warnings suitable for end-of-interpretation display.
/// Excludes CAPTURED_MESSAGES since those are already printed inline by emit_message.
fn get_warnings_for_display() -> Vec<String> {
  let mut warnings = Vec::new();

  let calls = UNIMPLEMENTED_CALLS.with(|buffer| buffer.borrow().clone());
  if !calls.is_empty() {
    let joined = calls.join(", ");
    let verb = if calls.len() == 1 {
      "is a built-in Wolfram Language function"
    } else {
      "are built-in Wolfram Language functions"
    };
    warnings.push(format!("{} {} not yet implemented in Woxi.", joined, verb));
  }

  CAPTURED_WARNINGS.with(|buffer| {
    warnings.extend(buffer.borrow().clone());
  });

  warnings
}

/// Gets the raw CAPTURED_MESSAGES buffer
pub fn get_captured_messages_raw() -> Vec<String> {
  CAPTURED_MESSAGES.with(|buffer| buffer.borrow().clone())
}

/// Returns true if currently inside a Quiet[] evaluation
fn is_quiet() -> bool {
  QUIET_LEVEL.with(|level| *level.borrow() > 0)
}

/// Increment the quiet level (enter a Quiet[] block)
pub fn push_quiet() {
  QUIET_LEVEL.with(|level| *level.borrow_mut() += 1);
}

/// Decrement the quiet level (leave a Quiet[] block)
pub fn pop_quiet() {
  QUIET_LEVEL.with(|level| {
    let mut l = level.borrow_mut();
    if *l > 0 {
      *l -= 1;
    }
  });
}

/// Snapshot the current state of all warning/message buffers (for Quiet save/restore)
pub fn snapshot_warnings() -> (Vec<String>, Vec<String>, Vec<String>) {
  let unimpl = UNIMPLEMENTED_CALLS.with(|b| b.borrow().clone());
  let warns = CAPTURED_WARNINGS.with(|b| b.borrow().clone());
  let msgs = CAPTURED_MESSAGES.with(|b| b.borrow().clone());
  (unimpl, warns, msgs)
}

/// Restore all warning/message buffers to a previous snapshot
pub fn restore_warnings(snapshot: (Vec<String>, Vec<String>, Vec<String>)) {
  UNIMPLEMENTED_CALLS.with(|b| *b.borrow_mut() = snapshot.0);
  CAPTURED_WARNINGS.with(|b| *b.borrow_mut() = snapshot.1);
  CAPTURED_MESSAGES.with(|b| *b.borrow_mut() = snapshot.2);
}

/// Capture the current evaluation stack trace for the most recent error.
/// Called when an error propagates through function call evaluation.
/// Only captures if no trace has been captured yet (preserves the deepest trace).
pub fn capture_error_trace() {
  LAST_ERROR_TRACE.with(|t| {
    if t.borrow().is_none()
      && let Some(trace) = format_stack_trace()
    {
      *t.borrow_mut() = Some(trace);
    }
  });
}

/// Take the last captured error stack trace (clears it).
pub fn take_error_trace() -> Option<String> {
  LAST_ERROR_TRACE.with(|t| t.borrow_mut().take())
}

/// Push a function name onto the evaluation stack.
pub fn push_eval_stack(name: &str) {
  EVAL_STACK.with(|s| s.borrow_mut().push(name.to_string()));
}

/// Pop the top entry from the evaluation stack.
pub fn pop_eval_stack() {
  EVAL_STACK.with(|s| {
    s.borrow_mut().pop();
  });
}

/// Get a snapshot of the current evaluation stack (bottom to top).
pub fn get_eval_stack() -> Vec<String> {
  EVAL_STACK.with(|s| s.borrow().clone())
}

/// Format the evaluation stack as a human-readable stack trace string.
/// Always returns None now — the trace was Woxi-specific debug info that
/// diverged from wolframscript's output. Keeping the helper so callers
/// don't need to change.
fn format_stack_trace() -> Option<String> {
  None
}

/// Emit a Wolfram-style message (e.g. "Power::infy: Infinite expression 1/0 encountered.").
/// Suppressed when inside Quiet[]. Tracked in CAPTURED_MESSAGES for Check[] interaction.
/// When messages_to_stdout is enabled, prints to stdout (matching wolframscript).
/// Includes a stack trace showing the chain of function calls.
///
/// This writes to the process stream only (stdout when `messages_to_stdout`
/// is enabled — the CLI `eval`/`run`/REPL paths — stderr otherwise). It does
/// NOT touch the `CAPTURED_STDOUT` buffer that `interpret_with_stdout`
/// returns: most messages are diagnostics, and Woxi still emits some that
/// wolframscript does not, so mirroring them all into captured stdout would
/// pollute snapshots with Woxi-only output. Use [`emit_message_to_stdout`]
/// only at the specific sites whose message provably matches wolframscript's
/// stdout (e.g. Import/MedianFilter argument errors).
pub fn emit_message(msg: &str) {
  let _ = emit_message_core(msg);
}

/// Public wrapper for [`message_name`], used by Check's tag filtering.
pub fn message_name_of(msg: &str) -> Option<String> {
  message_name(msg)
}

/// Extract the `Symbol::tag` message name from a message string. Some
/// messages start with right-aligned context lines (e.g. Power::indet
/// prints the exponent above the base), so scan every line for the tag.
fn message_name(msg: &str) -> Option<String> {
  for line in msg.lines() {
    if let Some(dc) = line.find("::")
      && let Some(rest) = line[dc + 2..].find(':')
    {
      let name = &line[..dc + 2 + rest];
      if !name.contains(' ') {
        return Some(name.to_string());
      }
    }
  }
  None
}

/// Shared emission path: captures the message, applies wolframscript's
/// General::stop suppression (the same Symbol::tag prints at most three
/// times per calculation; the third is followed by a General::stop notice
/// and later ones are silent), and prints to the configured stream.
/// Returns whether the message was displayed and the General::stop line
/// when one was announced, so `emit_message_to_stdout` can mirror both.
fn emit_message_core(msg: &str) -> (bool, Option<String>) {
  if message_is_off(msg) {
    return (false, None);
  }
  CAPTURED_MESSAGES.with(|buffer| {
    buffer.borrow_mut().push(msg.to_string());
  });
  if is_quiet() {
    return (false, None);
  }
  let mut stop_line: Option<String> = None;
  if let Some(name) = message_name(msg)
    && name != "General::stop"
  {
    let count = MESSAGE_STOP_COUNTS.with(|m| {
      let mut m = m.borrow_mut();
      let c = m.entry(name.clone()).or_insert(0);
      *c += 1;
      *c
    });
    if count > 3 {
      return (false, None);
    }
    if count == 3 {
      let stop = format!(
        "General::stop: Further output of {name} will be suppressed during this calculation."
      );
      CAPTURED_MESSAGES.with(|buffer| {
        buffer.borrow_mut().push(stop.clone());
      });
      stop_line = Some(stop);
    }
  }
  let to_stdout = MESSAGES_TO_STDOUT.with(|f| *f.borrow());
  let trace = format_stack_trace();
  if to_stdout {
    println!();
    println!("{}", msg);
    if let Some(trace) = trace {
      println!("{}", trace);
    }
    if let Some(stop) = &stop_line {
      println!();
      println!("{}", stop);
    }
  } else {
    eprintln!();
    eprintln!("{}", msg);
    if let Some(trace) = trace {
      eprintln!("{}", trace);
    }
    if let Some(stop) = &stop_line {
      eprintln!();
      eprintln!("{}", stop);
    }
  }
  (true, stop_line)
}

/// Reset the per-calculation General::stop counters (a new top-level
/// evaluation is a new "calculation").
fn reset_message_stop_counts() {
  MESSAGE_STOP_COUNTS.with(|m| m.borrow_mut().clear());
}

/// Like [`emit_message`], but also mirrors the message — with wolframscript's
/// leading-blank-line format — into `CAPTURED_STDOUT`, the buffer that
/// `interpret_with_stdout` returns to the snapshot tests, playground, and
/// Jupyter kernel. wolframscript writes messages to stdout, so this keeps
/// those library consumers byte-for-byte consistent with it.
///
/// Use this ONLY where Woxi's message provably matches wolframscript's
/// stdout for the same input; for the general diagnostic case use
/// [`emit_message`] (see its note). Respects `Quiet[]`/`Off[]`.
pub fn emit_message_to_stdout(msg: &str) {
  let (shown, stop_line) = emit_message_core(msg);
  if !shown {
    return;
  }
  capture_stdout_raw("\n");
  capture_stdout_raw(msg);
  capture_stdout_raw("\n");
  if let Some(stop) = stop_line {
    capture_stdout_raw("\n");
    capture_stdout_raw(&stop);
    capture_stdout_raw("\n");
  }
}

/// Clears the captured graphics buffer
pub fn clear_captured_graphics() {
  CAPTURED_GRAPHICS.with(|buffer| {
    buffer.borrow_mut().clear();
  });
}

/// Stores SVG graphics for capture by the Jupyter kernel
pub fn capture_graphics(svg: &str) {
  CAPTURED_GRAPHICS.with(|buffer| {
    buffer.borrow_mut().push(svg.to_string());
  });
}

/// Capture SVG and return an Expr::Graphics carrying the SVG data.
pub fn graphics_result(svg: String) -> syntax::Expr {
  capture_graphics(&svg);
  syntax::Expr::Graphics {
    svg,
    is_3d: false,
    source: None,
    head: None,
  }
}

/// Like `graphics_result` but reports `head` (e.g. `GeoGraphics`) from `Head`
/// instead of the default `Graphics`. The SVG renders identically.
fn graphics_result_with_head(svg: String, head: &str) -> syntax::Expr {
  capture_graphics(&svg);
  syntax::Expr::Graphics {
    svg,
    is_3d: false,
    source: None,
    head: Some(head.to_string()),
  }
}

/// Like `graphics_result` but also stores the source plot data
/// so that `Show` can later merge pre-rendered plots by re-rendering
/// via plotters.
pub fn graphics_result_with_source(
  svg: String,
  source: syntax::PlotSource,
) -> syntax::Expr {
  capture_graphics(&svg);
  syntax::Expr::Graphics {
    svg,
    is_3d: false,
    source: Some(Box::new(source)),
    head: None,
  }
}

/// Capture SVG and return an Expr::Graphics for 3D graphics.
pub fn graphics3d_result(svg: String) -> syntax::Expr {
  capture_graphics(&svg);
  syntax::Expr::Graphics {
    svg,
    is_3d: true,
    source: None,
    head: None,
  }
}

/// Gets the last captured graphics content (backward compatible)
pub fn get_captured_graphics() -> Option<String> {
  CAPTURED_GRAPHICS.with(|buffer| buffer.borrow().last().cloned())
}

/// Gets all captured graphics SVGs
fn get_all_captured_graphics() -> Vec<String> {
  CAPTURED_GRAPHICS.with(|buffer| buffer.borrow().clone())
}

/// Clears the captured GraphicsBox buffer
fn clear_captured_graphicsbox() {
  CAPTURED_GRAPHICSBOX.with(|buffer| {
    *buffer.borrow_mut() = None;
  });
}

/// Stores a GraphicsBox expression string for .nb export
pub fn capture_graphicsbox(expr: &str) {
  CAPTURED_GRAPHICSBOX.with(|buffer| {
    *buffer.borrow_mut() = Some(expr.to_string());
  });
}

/// Gets the captured GraphicsBox expression
pub fn get_captured_graphicsbox() -> Option<String> {
  CAPTURED_GRAPHICSBOX.with(|buffer| buffer.borrow().clone())
}

/// Clears the captured output SVG buffer
fn clear_captured_output_svg() {
  CAPTURED_OUTPUT_SVG.with(|buffer| {
    *buffer.borrow_mut() = None;
  });
}

/// Stores an SVG rendering of the text output
fn capture_output_svg(svg: &str) {
  CAPTURED_OUTPUT_SVG.with(|buffer| {
    *buffer.borrow_mut() = Some(svg.to_string());
  });
}

/// Gets the captured output SVG
pub fn get_captured_output_svg() -> Option<String> {
  CAPTURED_OUTPUT_SVG.with(|buffer| buffer.borrow().clone())
}

/// Clears the captured audio buffer
fn clear_captured_sound() {
  CAPTURED_SOUND.with(|buffer| {
    *buffer.borrow_mut() = None;
  });
}

/// Stores synthesized audio as a base64-encoded WAV (from Play/Sound).
fn capture_sound(wav_base64: &str) {
  capture_audio(AudioOutput {
    base64: wav_base64.to_string(),
    mime: "audio/wav".to_string(),
    label: None,
  });
}

/// Stores playable audio (e.g. from a file-backed Audio object).
fn capture_audio(audio: AudioOutput) {
  CAPTURED_SOUND.with(|buffer| {
    *buffer.borrow_mut() = Some(audio);
  });
}

/// Gets the captured playable audio, if any.
pub fn get_captured_sound() -> Option<AudioOutput> {
  CAPTURED_SOUND.with(|buffer| buffer.borrow().clone())
}

/// Set a system variable (like $ScriptCommandLine) in the environment
pub fn set_system_variable(name: &str, value: &str) {
  ENV.with(|e| {
    e.borrow_mut()
      .insert(name.to_string(), StoredValue::Raw(value.to_string()));
  });
}

/// Remove a first line that starts with "#!" (shebang);
/// returns the remainder as a new `String`.
pub fn without_shebang(src: &str) -> String {
  if src.starts_with("#!") {
    src.lines().skip(1).collect::<Vec<_>>().join("\n")
  } else {
    src.to_owned()
  }
}

/// Get all defined symbol names (variables and user functions).
/// Used by the Names[] function.
pub fn get_defined_names() -> Vec<String> {
  let mut names = Vec::new();
  ENV.with(|e| {
    for key in e.borrow().keys() {
      names.push(key.clone());
    }
  });
  FUNC_DEFS.with(|m| {
    for key in m.borrow().keys() {
      if !names.contains(key) {
        names.push(key.clone());
      }
    }
  });
  names.sort();
  names
}

/// Push a context onto the Begin/End context stack.
pub fn push_context(ctx: String) {
  CONTEXT_STACK.with(|s| s.borrow_mut().push(ctx));
}

/// Pop a context from the Begin/End context stack.
/// Returns the popped context, or None if the stack is empty.
pub fn pop_context() -> Option<String> {
  CONTEXT_STACK.with(|s| s.borrow_mut().pop())
}

/// The currently active context — the top of the Begin/BeginPackage
/// stack, or `"Global`"` when nothing has been pushed.
pub fn current_context() -> String {
  CONTEXT_STACK.with(|s| {
    s.borrow()
      .last()
      .cloned()
      .unwrap_or_else(|| "Global`".to_string())
  })
}

/// The currently active `$ContextPath` — the top of the
/// `CONTEXT_PATH_STACK`, or the `["System`", "Global`"]` default
/// when nothing has been pushed by `BeginPackage[]`.
pub fn current_context_path() -> Vec<String> {
  CONTEXT_PATH_STACK.with(|s| {
    s.borrow()
      .last()
      .cloned()
      .unwrap_or_else(|| vec!["System`".to_string(), "Global`".to_string()])
  })
}

/// Push a new `$ContextPath` value (used by `BeginPackage[]`).
pub fn push_context_path(path: Vec<String>) {
  CONTEXT_PATH_STACK.with(|s| s.borrow_mut().push(path));
}

/// Pop the topmost `$ContextPath` value (used by `EndPackage[]`).
pub fn pop_context_path() -> Option<Vec<String>> {
  CONTEXT_PATH_STACK.with(|s| s.borrow_mut().pop())
}

/// Whether any `BeginPackage[]` is currently active (i.e. the
/// `$ContextPath` stack is non-empty). Used by `EndPackage[]` to decide
/// between popping and emitting `EndPackage::noctx`.
pub fn has_package_context() -> bool {
  CONTEXT_PATH_STACK.with(|s| !s.borrow().is_empty())
}

/// Register an additional package context to be reported by `$Packages`.
/// Pushed by `BeginPackage[]` (the first argument and any extras in the
/// second argument). Duplicate entries are ignored.
pub fn register_package(name: String) {
  PACKAGES_EXTRA.with(|s| {
    let mut v = s.borrow_mut();
    if !v.contains(&name) {
      v.push(name);
    }
  });
}

/// Return the dynamic `$Packages` list — extras pushed by `BeginPackage[]`
/// followed by the canonical `{"System`", "Global`"}` baseline.
pub fn packages_list() -> Vec<String> {
  let mut out = PACKAGES_EXTRA.with(|s| s.borrow().clone());
  out.push("System`".to_string());
  out.push("Global`".to_string());
  out
}

/// DownValues for `sym` as stored 6-tuples, merging the literal-argument
/// memoizations (kept in MEMO_VALUES, not FUNC_DEFS) back in as if they were
/// ordinary literal DownValues — synthesizing the same shape `set_ast` would
/// have produced. Memoized entries come first (literal definitions precede
/// pattern ones, matching specificity ordering) and are sorted by key for a
/// deterministic order. Returns None only when the symbol has neither.
/// Used by Definition / Information so they still report `f[0] = 42`-style
/// literal definitions after the memoization fast-path routed them aside.
#[allow(clippy::type_complexity)]
pub fn down_values_with_memo(
  sym: &str,
) -> Option<
  Vec<(
    Vec<String>,
    Vec<Option<syntax::Expr>>,
    Vec<Option<syntax::Expr>>,
    Vec<Option<String>>,
    Vec<u8>,
    syntax::Expr,
  )>,
> {
  let memo_entries = MEMO_VALUES.with(|m| {
    m.borrow().get(sym).map(|cache| {
      let mut entries: Vec<_> = cache.iter().collect();
      entries.sort_by(|a, b| a.0.cmp(b.0));
      entries
        .into_iter()
        .map(|(_key, (arg_exprs, value))| {
          let n = arg_exprs.len();
          let params: Vec<String> =
            (0..n).map(|i| format!("_dv{}", i)).collect();
          let conditions: Vec<Option<syntax::Expr>> = arg_exprs
            .iter()
            .enumerate()
            .map(|(i, a)| {
              Some(syntax::Expr::Comparison {
                operands: vec![
                  syntax::Expr::Identifier(format!("_dv{}", i)),
                  a.clone(),
                ],
                operators: vec![syntax::ComparisonOp::SameQ],
              })
            })
            .collect();
          (
            params,
            conditions,
            vec![None; n],
            vec![None; n],
            vec![1u8; n],
            value.clone(),
          )
        })
        .collect::<Vec<_>>()
    })
  });
  let func_defs = FUNC_DEFS.with(|m| m.borrow().get(sym).cloned());
  match (memo_entries, func_defs) {
    (None, None) => None,
    (Some(m), None) => Some(m),
    (None, Some(f)) => Some(f),
    (Some(mut m), Some(f)) => {
      m.extend(f);
      Some(m)
    }
  }
}

/// Restores previously-saved global bindings when dropped, so `f` in
/// [`with_scoped_globals`] cannot leak or lose bindings even on early return.
struct ScopedGlobals {
  saved: Vec<(String, Option<StoredValue>)>,
}

impl Drop for ScopedGlobals {
  fn drop(&mut self) {
    // Restore in reverse installation order.
    for (name, prev) in self.saved.drain(..).rev() {
      ENV.with(|e| {
        let mut env = e.borrow_mut();
        match prev {
          Some(v) => {
            env.insert(name, v);
          }
          None => {
            env.remove(&name);
          }
        }
      });
    }
  }
}

/// Run `f` with `(name, value)` bindings installed as global symbol values,
/// restoring the previous globals afterward.
///
/// Each `value` string is parsed and evaluated *once* here and stored
/// structurally (`ExprVal`), so a large literal (e.g. a Manipulate `data`
/// matrix) is parsed a single time rather than re-embedded and re-parsed on
/// every body/display/probe evaluation inside `f`. Parsing large list
/// literals dominates the cost of an interactive re-render, so this is the
/// difference between a snappy and a sluggish Manipulate widget.
pub fn with_scoped_globals<R>(
  bindings: &[(String, String)],
  f: impl FnOnce() -> R,
) -> R {
  let mut saved: Vec<(String, Option<StoredValue>)> =
    Vec::with_capacity(bindings.len());
  for (name, value) in bindings {
    let evaluated = interpret_to_expr(value)
      .unwrap_or_else(|_| syntax::Expr::Identifier(name.clone()));
    let prev = ENV.with(|e| {
      e.borrow_mut()
        .insert(name.clone(), StoredValue::ExprVal(evaluated))
    });
    saved.push((name.clone(), prev));
  }
  let _guard = ScopedGlobals { saved };
  f()
}

/// Clear all thread-local interpreter state (environment variables
/// and user-defined functions).  Useful for isolating test runs.
pub fn clear_state() {
  ENV.with(|e| e.borrow_mut().clear());
  FUNC_DEFS.with(|m| m.borrow_mut().clear());
  MEMO_VALUES.with(|m| m.borrow_mut().clear());
  FUNC_ATTRS.with(|m| m.borrow_mut().clear());
  FUNC_OPTIONS.with(|m| m.borrow_mut().clear());
  FUNC_OPTIONS_DELAYED.with(|m| m.borrow_mut().clear());
  FUNC_OPTS_INLINE.with(|m| m.borrow_mut().clear());
  UPVALUES.with(|m| m.borrow_mut().clear());
  SOW_STACK.with(|s| s.borrow_mut().clear());
  CONTEXT_STACK.with(|s| s.borrow_mut().clear());
  CONTEXT_PATH_STACK.with(|s| s.borrow_mut().clear());
  PACKAGES_EXTRA.with(|s| s.borrow_mut().clear());
  RECURSION_DEPTH.with(|d| d.set(0));
  EVAL_STACK.with(|s| s.borrow_mut().clear());
  LAST_ERROR_TRACE.with(|t| *t.borrow_mut() = None);
  evaluator::assignment::USER_PRINT_FORMS.with(|v| v.borrow_mut().clear());
  evaluator::assignment::FORMAT_VALUES.with(|m| m.borrow_mut().clear());
  evaluator::assignment::SUB_VALUES.with(|m| m.borrow_mut().clear());
  evaluator::assignment::N_VALUES.with(|m| m.borrow_mut().clear());
  functions::entity_ast::clear_entity_stores();
  unseed_rng();
  clear_captured_stdout();
  reset_message_stop_counts();
  clear_captured_graphics();
  clear_captured_graphicsbox();
}

/// Set the $ScriptCommandLine variable from command-line arguments
pub fn set_script_command_line(args: &[String]) {
  // Format as a Wolfram list: {"script.wls", "arg1", "arg2", ...}
  let list_str = format!(
    "{{{}}}",
    args
      .iter()
      .map(|s| format!("\"{}\"", s))
      .collect::<Vec<_>>()
      .join(", ")
  );
  set_system_variable("$ScriptCommandLine", &list_str);
}

// Track recursion depth to avoid clearing stdout in nested calls
thread_local! {
    static INTERPRET_DEPTH: std::cell::RefCell<usize> = const { std::cell::RefCell::new(0) };
}

pub fn interpret(input: &str) -> Result<String, InterpreterError> {
  // Normalize CRLF to LF so line continuation and newline handling work
  // consistently regardless of line ending style.
  let input = if input.contains('\r') {
    std::borrow::Cow::Owned(input.replace("\r\n", "\n").replace('\r', "\n"))
  } else {
    std::borrow::Cow::Borrowed(input)
  };
  // Expand Wolfram character escapes (\.HH, \:HHHH, \OOO) to their UTF-8
  // characters BEFORE any fast paths so quoted-string and other shortcuts
  // see the post-expansion form (matching wolframscript).
  let input = if input.contains('\\') {
    std::borrow::Cow::Owned(expand_char_escapes(&input))
  } else {
    input
  };
  // Treat the modifier-letter circumflex `ˆ` (U+02C6) as the Power operator.
  let input = if input.contains('\u{02C6}') {
    std::borrow::Cow::Owned(normalize_circumflex_operator(&input))
  } else {
    input
  };
  let trimmed = input.trim();

  // Fast path for simple literals that don't need parsing
  // Check for integer
  if let Ok(n) = trimmed.parse::<i64>() {
    // Visual hosts (Playground/Studio) still want the typeset SVG so a bare
    // literal gets the same digit grouping as a computed result
    // (`10000` → `10 000`). The CLI skips this (not visual mode).
    if is_visual_mode() {
      generate_output_svg(&syntax::Expr::Integer(n.into()));
    }
    return Ok(n.to_string());
  }
  // Check for float (must contain '.' to distinguish from integer)
  if trimmed.contains('.')
    && let Ok(n) = trimmed.parse::<f64>()
  {
    // Skip the fast path when the literal would be parsed as an
    // accuracy/precision-tagged BigFloat by Wolfram:
    // - `0.000…0` with 18+ trailing zeros → accuracy form `0``N.`
    // - non-zero literal with 18+ total digits → precision form
    if let Some(dot) = trimmed.find('.') {
      let int_part = &trimmed[..dot];
      let frac_part = &trimmed[dot + 1..];
      let int_signless = int_part.trim_start_matches(['+', '-']);
      let int_zero =
        int_signless.is_empty() || int_signless.chars().all(|c| c == '0');
      let total_digits = int_signless.len() + frac_part.len();
      let zero_accuracy = n == 0.0
        && int_zero
        && frac_part.len() >= 18
        && frac_part.chars().all(|c| c == '0');
      // Use significant-digit count: leading zeros in 0.xxx don't count.
      let significant_digits = if int_zero {
        let leading_zeros = frac_part.chars().take_while(|c| *c == '0').count();
        frac_part.len().saturating_sub(leading_zeros)
      } else {
        total_digits
      };
      let nonzero_precision = n != 0.0 && significant_digits >= 18;
      if zero_accuracy || nonzero_precision {
        // Fall through to the full parser.
      } else {
        if is_visual_mode() {
          generate_output_svg(&syntax::Expr::Real(n));
        }
        return Ok(syntax::format_real(n));
      }
    } else {
      if is_visual_mode() {
        generate_output_svg(&syntax::Expr::Real(n));
      }
      return Ok(syntax::format_real(n));
    }
  }
  // Check for quoted string - return content without quotes (like wolframscript)
  if trimmed.starts_with('"') && trimmed.ends_with('"') && trimmed.len() >= 2 {
    // Make sure there are no unescaped quotes inside
    let inner = &trimmed[1..trimmed.len() - 1];
    // Skip fast path if string contains escape sequences that the parser
    // would expand (named characters, box-syntax markers, backtick escapes).
    if !inner.contains('"')
      && !inner.contains("\\[")
      && !inner.contains("\\(")
      && !inner.contains("\\)")
      && !inner.contains("\\!")
      && !inner.contains("\\*")
      && !inner.contains("\\`")
    {
      return Ok(inner.to_string());
    }
  }

  // Fast path for simple list literals like {a, b, c}
  // This handles many cases where we're just passing data around
  if trimmed.starts_with('{') && trimmed.ends_with('}') {
    // Check if it's a simple list (no operators that need evaluation)
    if !trimmed.contains("->")
      && !trimmed.contains(":>")
      && !trimmed.contains("/.")
      && !trimmed.contains("//")
      && !trimmed.contains("/@")
      && !trimmed.contains("@@")
      && !trimmed.contains('!')
      && !trimmed.contains('+')
      && !trimmed.contains('-')
      && !trimmed.contains('*')
      && !trimmed.contains('/')
      && !trimmed.contains('[')
      && !trimmed.contains('"')
      && !trimmed.contains('#')
      && !trimmed.contains("Nothing")
      && !trimmed.contains(" . ")
      && !trimmed.contains(".{")
      && !trimmed.contains('^')
      && !trimmed.contains('.')
      && !trimmed.contains('=')
      && !trimmed.contains('~')
      && !trimmed.contains('<')
      && !trimmed.contains('>')
      && !trimmed.contains('&')
      && !trimmed.contains('|')
      && !trimmed.contains('?')
      && !trimmed.contains(';')
      && !trimmed.contains('@')
      && !trimmed.contains('\u{2A2F}') // ⨯ Cross product
      && !trimmed.contains('\u{F3C4}') // \[Cross] PUA form
      && !trimmed.contains("\\[")
    // any named character operator
    // Reals may need scientific notation formatting
    {
      // Check if any element needs evaluation (named colors, date symbols, etc.)
      let needs_eval = trimmed[1..trimmed.len() - 1].split(',').any(|item| {
        let item = item.trim();
        evaluator::named_color_expr(item).is_some()
          || matches!(
            item,
            "Now"
              | "Today"
              | "Tomorrow"
              | "Yesterday"
              | "Thick"
              | "Thin"
              | "Dashed"
              | "Dotted"
              | "DotDashed"
          )
      });
      // In visual mode fall through to the full parser so a list of literals
      // gets a typeset SVG with digit-grouped numbers (`{10000}` → `{10 000}`);
      // the CLI keeps the fast return.
      if !needs_eval && !is_visual_mode() {
        // Simple list with no function calls or operators - return as-is
        return Ok(trimmed.to_string());
      }
    }
  }

  // Fast path for simple identifiers (variable lookup)
  if trimmed
    .chars()
    .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '$')
    && !trimmed.is_empty()
    && trimmed.chars().next().unwrap().is_ascii_alphabetic()
  {
    // This is a simple identifier
    if let Some(stored) = ENV.with(|e| e.borrow().get(trimmed).cloned()) {
      return Ok(match stored {
        StoredValue::ExprVal(e) => syntax::top_level_output(&e),
        StoredValue::Raw(val) => val,
        StoredValue::Association(items) => {
          let items_expr: Vec<(syntax::Expr, syntax::Expr)> = items
            .iter()
            .map(|(k, v)| {
              let key_expr = syntax::string_to_expr(k)
                .unwrap_or(syntax::Expr::Identifier(k.clone()));
              let val_expr = syntax::string_to_expr(v)
                .unwrap_or(syntax::Expr::Raw(v.clone()));
              (key_expr, val_expr)
            })
            .collect();
          syntax::expr_to_output(&syntax::Expr::Association(items_expr))
        }
      });
    }
    // Handle built-in symbols that evaluate to values
    #[cfg(not(target_arch = "wasm32"))]
    if trimmed == "Now" {
      use chrono::Local;
      let now = Local::now();
      let seconds = now
        .format("%S%.f")
        .to_string()
        .parse::<f64>()
        .unwrap_or(0.0);
      let tz_offset_hours = now.offset().local_minus_utc() as f64 / 3600.0;
      let expr = syntax::Expr::FunctionCall {
        name: "DateObject".to_string(),
        args: vec![
          syntax::Expr::List(
            vec![
              syntax::Expr::Integer(
                now.format("%Y").to_string().parse::<i128>().unwrap(),
              ),
              syntax::Expr::Integer(
                now.format("%m").to_string().parse::<i128>().unwrap(),
              ),
              syntax::Expr::Integer(
                now.format("%d").to_string().parse::<i128>().unwrap(),
              ),
              syntax::Expr::Integer(
                now.format("%H").to_string().parse::<i128>().unwrap(),
              ),
              syntax::Expr::Integer(
                now.format("%M").to_string().parse::<i128>().unwrap(),
              ),
              syntax::Expr::Real(seconds),
            ]
            .into(),
          ),
          syntax::Expr::String("Instant".to_string()),
          syntax::Expr::String("Gregorian".to_string()),
          syntax::Expr::Real(tz_offset_hours),
        ]
        .into(),
      };
      return Ok(syntax::expr_to_output(&expr));
    }
    #[cfg(target_arch = "wasm32")]
    if trimmed == "Now" {
      let now = js_sys::Date::new_0();
      let seconds =
        now.get_seconds() as f64 + now.get_milliseconds() as f64 / 1000.0;
      let tz_offset_hours = -(now.get_timezone_offset() / 60.0);
      let expr = syntax::Expr::FunctionCall {
        name: "DateObject".to_string(),
        args: vec![
          syntax::Expr::List(
            vec![
              syntax::Expr::Integer(now.get_full_year() as i128),
              syntax::Expr::Integer((now.get_month() + 1) as i128),
              syntax::Expr::Integer(now.get_date() as i128),
              syntax::Expr::Integer(now.get_hours() as i128),
              syntax::Expr::Integer(now.get_minutes() as i128),
              syntax::Expr::Real(seconds),
            ]
            .into(),
          ),
          syntax::Expr::String("Instant".to_string()),
          syntax::Expr::String("Gregorian".to_string()),
          syntax::Expr::Real(tz_offset_hours),
        ]
        .into(),
      };
      return Ok(syntax::expr_to_output(&expr));
    }
    // Handle Today/Tomorrow/Yesterday → DateObject[{y, m, d}, Day]
    #[cfg(not(target_arch = "wasm32"))]
    if trimmed == "Today" || trimmed == "Tomorrow" || trimmed == "Yesterday" {
      use chrono::{Duration, Local};
      let now = Local::now();
      let date = match trimmed {
        "Tomorrow" => now + Duration::days(1),
        "Yesterday" => now - Duration::days(1),
        _ => now,
      };
      let expr = syntax::Expr::FunctionCall {
        name: "DateObject".to_string(),
        args: vec![
          syntax::Expr::List(
            vec![
              syntax::Expr::Integer(
                date.format("%Y").to_string().parse::<i128>().unwrap(),
              ),
              syntax::Expr::Integer(
                date.format("%m").to_string().parse::<i128>().unwrap(),
              ),
              syntax::Expr::Integer(
                date.format("%d").to_string().parse::<i128>().unwrap(),
              ),
            ]
            .into(),
          ),
          syntax::Expr::String("Day".to_string()),
        ]
        .into(),
      };
      return Ok(syntax::expr_to_output(&expr));
    }
    #[cfg(target_arch = "wasm32")]
    if trimmed == "Today" || trimmed == "Tomorrow" || trimmed == "Yesterday" {
      let now = js_sys::Date::new_0();
      let offset_days: i32 = match trimmed {
        "Tomorrow" => 1,
        "Yesterday" => -1,
        _ => 0,
      };
      let ms = now.get_time() + (offset_days as f64) * 86_400_000.0;
      let d = js_sys::Date::new(&wasm_bindgen::JsValue::from_f64(ms));
      let expr = syntax::Expr::FunctionCall {
        name: "DateObject".to_string(),
        args: vec![
          syntax::Expr::List(
            vec![
              syntax::Expr::Integer(d.get_full_year() as i128),
              syntax::Expr::Integer((d.get_month() + 1) as i128),
              syntax::Expr::Integer(d.get_date() as i128),
            ]
            .into(),
          ),
          syntax::Expr::String("Day".to_string()),
        ]
        .into(),
      };
      return Ok(syntax::expr_to_output(&expr));
    }
    // Handle named colors (Red → RGBColor[1, 0, 0], etc.)
    if let Some(color_expr) = evaluator::named_color_expr(trimmed) {
      return Ok(syntax::expr_to_output(&color_expr));
    }
    // Thick → Thickness[Large]
    if trimmed == "Thick" {
      return Ok("Thickness[Large]".to_string());
    }
    // Thin → Thickness[Tiny]
    if trimmed == "Thin" {
      return Ok("Thickness[Tiny]".to_string());
    }
    // Dashed → Dashing[{Small, Small}]
    if trimmed == "Dashed" {
      return Ok("Dashing[{Small, Small}]".to_string());
    }
    // Dotted → Dashing[{0, Small}]
    if trimmed == "Dotted" {
      return Ok("Dashing[{0, Small}]".to_string());
    }
    // DotDashed → Dashing[{0, Small, Small, Small}]
    if trimmed == "DotDashed" {
      return Ok("Dashing[{0, Small, Small, Small}]".to_string());
    }
    // Return identifier as-is if not found
    return Ok(trimmed.to_string());
  }

  // Fast path for simple function calls like MemberQ[{a, b}, x]
  // This handles the common case in Select predicates
  if let Some(result) = try_fast_function_call(trimmed) {
    return result;
  }

  let depth = INTERPRET_DEPTH.with(|d| {
    let mut depth = d.borrow_mut();
    let current = *depth;
    *depth += 1;
    current
  });

  // Only clear buffers at top level; a top-level interpret call is one
  // "calculation" for General::stop message suppression.
  if depth == 0 {
    clear_captured_stdout();
    clear_captured_warnings();
    reset_message_stop_counts();
  }

  // Decrement depth on scope exit
  struct DepthGuard;
  impl Drop for DepthGuard {
    fn drop(&mut self) {
      INTERPRET_DEPTH.with(|d| *d.borrow_mut() -= 1);
    }
  }
  let _guard = DepthGuard;

  // Insert semicolons at top-level newline boundaries so the PEG grammar
  // correctly separates statements like "fib[0] = 0\nfib[1] = 1" instead
  // of treating them as implicit multiplication.
  let preprocessed = insert_statement_separators(trimmed);

  // Regular interpretation - use AST-based evaluation
  let pairs = parse(&preprocessed)?;
  let mut pairs = pairs.into_iter();
  let program = pairs.next().ok_or(InterpreterError::EmptyInput)?;

  if program.as_rule() != Rule::Program {
    return Err(InterpreterError::EvaluationError(format!(
      "Expected Program, got {:?}",
      program.as_rule()
    )));
  }

  // Collect all program statements upfront so we can support Goto/Label
  // across top-level semicolon-separated expressions.
  enum ProgramStmt<'a> {
    Expr(syntax::Expr),
    FunctionDefinition(Pair<'a, Rule>),
    TagSetDelayed(Pair<'a, Rule>),
    TagSet(Pair<'a, Rule>),
    TagUnset(Pair<'a, Rule>),
    TrailingSemicolon,
  }
  let mut stmts: Vec<ProgramStmt> = Vec::new();
  // Also collect the Expr ASTs separately (with indices) for Label lookup
  let mut expr_asts: Vec<syntax::Expr> = Vec::new();
  for node in program.into_inner() {
    match node.as_rule() {
      Rule::Expression | Rule::TopLevelSpan => {
        let expr = syntax::pair_to_expr(node);
        expr_asts.push(expr.clone());
        stmts.push(ProgramStmt::Expr(expr));
      }
      Rule::FunctionDefinition => {
        stmts.push(ProgramStmt::FunctionDefinition(node))
      }
      Rule::TagSetDelayed => stmts.push(ProgramStmt::TagSetDelayed(node)),
      Rule::TagSet => stmts.push(ProgramStmt::TagSet(node)),
      Rule::TagUnset => stmts.push(ProgramStmt::TagUnset(node)),
      Rule::TrailingSemicolon => stmts.push(ProgramStmt::TrailingSemicolon),
      _ => {} // ignore EOI, etc.
    }
  }

  // The outcome of the most recently executed statement. Expression results
  // are kept as bare `Expr`s during the loop; the display pipeline (render
  // passes, SVG typesetting, text formatting) runs once after the loop on
  // whichever value actually becomes the program's output. Running it per
  // statement typeset multi-million-cell intermediates just to throw the
  // text away — `data = Import["big.csv"]; Length[data]` spent nearly all
  // of its time formatting the discarded list.
  enum StmtOutcome {
    Display(syntax::Expr),
    Text(String),
  }
  let mut last_result: Option<StmtOutcome> = None;
  let mut any_nonempty = false;
  let mut trailing_semicolon = false;
  let mut stmt_idx = 0;
  'goto_loop: loop {
    if stmt_idx >= stmts.len() {
      break;
    }
    match &stmts[stmt_idx] {
      ProgramStmt::Expr(expr) => {
        // Evaluate using AST-based evaluation
        // At top level, uncaught Return[] just yields its argument value
        // (matching wolframscript: `Return[5]` outputs `5`, not `Return[5]`).
        //
        // Pre-pass (visual mode only): when the top-level expression is a
        // list (1-D or 2-D) of graphics-producing function calls, inject
        // `ImageSize -> per_cell_w` into each child so plots are rendered
        // at the right per-cell size instead of being scaled down after
        // the fact. Skipped in plain `interpret()` mode so symbolic text
        // output (e.g. `{TreeForm[f[x]], TreeForm[g[y]]}`) isn't polluted
        // with an injected ImageSize option that was never written.
        let rewritten_expr = if VISUAL_MODE.with(|v| *v.borrow()) {
          functions::graphics::inject_image_size_for_list_of_graphics(expr)
        } else {
          None
        };
        let expr_to_eval: &syntax::Expr =
          rewritten_expr.as_ref().unwrap_or(expr);
        // Track Return propagation so multi-statement programs short-
        // circuit at the first Return — wolframscript treats top-level
        // `;`-separated statements as a CompoundExpression, where Return
        // bubbles past the remaining steps.
        let mut return_short_circuit = false;
        let mut result_expr =
          match evaluator::evaluate_expr_to_expr(expr_to_eval) {
            Err(InterpreterError::ReturnValue(val)) => {
              return_short_circuit = true;
              *val
            }
            Ok(syntax::Expr::FunctionCall { ref name, ref args })
              if name == "Return" && args.len() == 1 =>
            {
              return_short_circuit = true;
              args[0].clone()
            }
            Err(InterpreterError::Abort) => {
              return Ok("$Aborted".to_string());
            }
            Err(InterpreterError::GotoSignal(tag)) => {
              // Search for matching Label in the top-level expression list
              if let Some(label_idx) =
                evaluator::find_label_index(&expr_asts, &tag)
              {
                // Find the stmt index corresponding to this expr_ast index
                let mut expr_count = 0;
                for (si, s) in stmts.iter().enumerate() {
                  if matches!(s, ProgramStmt::Expr(_)) {
                    if expr_count == label_idx {
                      stmt_idx = si + 1; // resume after the Label
                      continue 'goto_loop;
                    }
                    expr_count += 1;
                  }
                }
              }
              let tag_str = syntax::expr_to_string(&tag);
              emit_message(&format!(
                "Goto::nolabel: Label {} not found.",
                tag_str
              ));
              syntax::Expr::Identifier("Null".to_string())
            }
            Err(InterpreterError::ThrowValue(val, tag)) => {
              // An uncaught Throw aborts evaluation at top level.
              // wolframscript emits Throw::nocatch and produces no result;
              // surface the same message instead of leaking the internal
              // error. The throw form shows its value (and tag, if any).
              let throw_args = match tag {
                Some(t) => vec![*val, *t],
                None => vec![*val],
              };
              let throw_expr = syntax::Expr::FunctionCall {
                name: "Throw".to_string(),
                args: throw_args.into(),
              };
              emit_message(&format!(
                "Throw::nocatch: Uncaught {} returned to top level.",
                syntax::format_expr(&throw_expr, syntax::ExprForm::Output)
              ));
              // wolframscript produces no result (the message is the only
              // output), so return an empty string rather than a value.
              return Ok(String::new());
            }
            other => other?,
          };
        // Multi-statement input behaves like CompoundExpression: a
        // trailing `Sequence[…]` splices, so we keep just its last
        // element. A lone `Sequence[1, 2]` (single-statement program)
        // still prints as `12`.
        let multi_statement = stmts
          .iter()
          .filter(|s| matches!(s, ProgramStmt::Expr(_)))
          .count()
          > 1;
        if multi_statement
          && let syntax::Expr::FunctionCall {
            name: n,
            args: seq_args,
          } = &result_expr
          && n == "Sequence"
        {
          result_expr = seq_args
            .last()
            .cloned()
            .unwrap_or_else(|| syntax::Expr::Identifier("Null".to_string()));
        }
        // Keep `%` / `Out[]` history per statement (notebook semantics:
        // `2+2` then `% + 1` within one cell sees the intermediate 4). The
        // final statement's entry is overwritten with the post-render value
        // by the deferred display pipeline below.
        if output_history_enabled() {
          set_last_output(result_expr.clone());
        }
        last_result = Some(StmtOutcome::Display(result_expr));
        any_nonempty = true;
        // A Return inside a multi-statement program propagates past any
        // remaining statements — drop them and emit the Return value.
        if return_short_circuit {
          break;
        }
      }
      ProgramStmt::FunctionDefinition(node) => {
        match store_function_definition(node.clone())? {
          Some(s) => last_result = Some(StmtOutcome::Text(s)),
          None => last_result = Some(StmtOutcome::Text("\0".to_string())),
        }
        any_nonempty = true;
      }
      ProgramStmt::TagSetDelayed(node) => {
        store_tag_set_delayed(node.clone(), false)?;
        last_result = Some(StmtOutcome::Text("\0".to_string()));
        any_nonempty = true;
      }
      ProgramStmt::TagSet(node) => {
        if let Some(rhs_str) = store_tag_set_delayed(node.clone(), true)? {
          last_result = Some(StmtOutcome::Text(rhs_str));
        } else {
          last_result = Some(StmtOutcome::Text("\0".to_string()));
        }
        any_nonempty = true;
      }
      ProgramStmt::TagUnset(node) => {
        execute_tag_unset(node.clone())?;
        last_result = Some(StmtOutcome::Text("\0".to_string()));
        any_nonempty = true;
      }
      ProgramStmt::TrailingSemicolon => {
        trailing_semicolon = true;
      }
    }
    stmt_idx += 1;
  }

  // Deferred display pipeline: render/format only the value that becomes
  // the program's output (see the StmtOutcome comment above). A trailing
  // semicolon suppresses display entirely, so `Import["big.csv"];` never
  // pays for formatting the discarded table.
  let last_result: Option<String> = match last_result {
    None => None,
    Some(StmtOutcome::Text(s)) => Some(s),
    Some(StmtOutcome::Display(_)) if trailing_semicolon => {
      Some("\0".to_string())
    }
    Some(StmtOutcome::Display(result_expr)) => {
      Some(format_top_level_result(result_expr))
    }
  };

  // Print consolidated unimplemented-function warning to stderr (top-level only)
  // Uses get_warnings_for_display() to avoid re-printing messages already shown by emit_message.
  if depth == 0 {
    for w in get_warnings_for_display() {
      eprintln!("{}", w);
    }
  }

  if any_nonempty {
    if trailing_semicolon {
      Ok("\0".to_string())
    } else {
      last_result.ok_or(InterpreterError::EmptyInput)
    }
  } else {
    Err(InterpreterError::EmptyInput)
  }
}

/// Run the display pipeline on a top-level statement's evaluated value:
/// render passes (Image/Graphics/Dataset/... wrappers), SVG typesetting for
/// visual hosts, `%` history, and the final text formatting. Returns the
/// output string ("\0" for Null, i.e. suppressed display).
fn format_top_level_result(result_expr: syntax::Expr) -> String {
  // If the result is an Image, render it as a PNG <img> tag
  let result_expr = render_image_if_needed(result_expr);
  // Render unevaluated Graphics[{...}] FunctionCalls to SVG (e.g.
  // from VoronoiMesh, or Graphics that stayed symbolic for Show merging).
  // Must run before other render passes that expect Expr::Graphics.
  let result_expr = render_graphics_fc_if_needed(result_expr);
  // Curve objects (PolarCurve, FilledPolarCurve) display as rendered
  // graphics in visual hosts (playground, studio), like in Wolfram
  // notebooks. The CLI keeps the symbolic echo to match wolframscript.
  let result_expr = if VISUAL_MODE.with(|v| *v.borrow()) {
    render_polar_curve_if_needed(result_expr)
  } else {
    result_expr
  };
  // If the result is a Sound built from Play[...] segments, synthesize a
  // playable WAV and embed it as an <audio> element (visual hosts only —
  // CLI mode keeps the Sound[...] expression, which renders as -Sound-).
  let result_expr = if VISUAL_MODE.with(|v| *v.borrow()) {
    render_sound_if_needed(result_expr)
  } else {
    result_expr
  };
  // If the result is an Audio object (file-backed or from sample data),
  // capture it as playable audio so the visual hosts render a graphical
  // audio player (CLI mode keeps the symbolic Audio[...] expression).
  let result_expr = if VISUAL_MODE.with(|v| *v.borrow()) {
    render_audio_if_needed(result_expr)
  } else {
    result_expr
  };
  // Render ComputationalMusic objects (MusicNote, MusicChord, …) as
  // musical-staff SVGs (visual mode only — CLI keeps them symbolic).
  let result_expr = if VISUAL_MODE.with(|v| *v.borrow()) {
    render_music_if_needed(result_expr)
  } else {
    result_expr
  };
  // Render Molecule results as a 2-D structure diagram (visual mode
  // only — the CLI keeps the symbolic Molecule[…] echo to match
  // wolframscript).
  let result_expr = if VISUAL_MODE.with(|v| *v.borrow()) {
    render_molecule_if_needed(result_expr)
  } else {
    result_expr
  };
  // Render DateObject results (e.g. from RandomDate, Now) as the
  // framed date panel Wolfram notebooks show (visual mode only —
  // CLI keeps the symbolic form to match wolframscript).
  let result_expr = if VISUAL_MODE.with(|v| *v.borrow()) {
    render_date_object_if_needed(result_expr)
  } else {
    result_expr
  };
  // If the result is a Grid expression, render it as SVG (visual mode
  // only — CLI mode keeps Grid[...] symbolic to match wolframscript).
  let result_expr = if VISUAL_MODE.with(|v| *v.borrow()) {
    render_grid_if_needed(result_expr)
  } else {
    result_expr
  };
  // If the result is a Dataset expression, render it as an SVG table
  let result_expr = render_dataset_if_needed(result_expr);
  // If the result is a Tabular expression, render it as an SVG table
  let result_expr = render_tabular_if_needed(result_expr);
  // In visual mode, render TableForm[list], MatrixForm[list], and Column[list] as SVGs
  let result_expr = if VISUAL_MODE.with(|v| *v.borrow()) {
    let result_expr = render_color_if_needed(result_expr);
    let result_expr = render_tableform_if_needed(result_expr);
    let result_expr = render_matrixform_if_needed(result_expr);
    let result_expr = render_traditionalform_list_if_needed(result_expr);
    let result_expr = render_column_if_needed(result_expr);
    let result_expr = render_row_if_needed(result_expr);
    let result_expr = render_treeform_if_needed(result_expr);
    let result_expr = render_framed_if_needed(result_expr);
    render_highlighted_if_needed(result_expr)
  } else {
    result_expr
  };
  // If the result is a list of Graphics objects, combine their SVGs
  // (visual contexts only — plain `interpret` keeps the list shape so
  // tests like Length[Table[Graphics[...], ...]] stay accurate).
  let result_expr = if VISUAL_MODE.with(|v| *v.borrow()) {
    render_graphics_list_if_needed(result_expr)
  } else {
    result_expr
  };
  // Top-level `Return[val]` (from Block/Module/While/For catching
  // an internal Return) displays as the bare value `val`, matching
  // wolframscript's REPL. The symbolic form is preserved when the
  // Return wrapper is held (e.g. inside ToString[…, InputForm]).
  let result_expr = match &result_expr {
    syntax::Expr::FunctionCall { name, args }
      if name == "Return" && args.len() == 1 =>
    {
      args[0].clone()
    }
    _ => result_expr,
  };
  // Generate SVG rendering of the result for playground display
  generate_output_svg(&result_expr);
  // Stash the top-level Expr so `%` / `Out[]` in a subsequent
  // evaluation resolves to this cell's result. We only do this when
  // output history is enabled — visual mode (e.g. woxi-studio) or the
  // terminal REPL (`woxi repl`). Plain command-line semantics (a fresh
  // process per evaluation, where `%` collapses to `Out[0]`) are
  // preserved — matching wolframscript.
  if output_history_enabled() {
    set_last_output(result_expr.clone());
  }
  // In visual mode (playground), unwrap StandardForm/InputForm wrappers
  // so they display like in a Wolfram notebook.
  // CLI mode preserves wrappers to match wolframscript behavior.
  let is_visual = VISUAL_MODE.with(|v| *v.borrow());
  let output_text = if is_visual {
    match &result_expr {
      syntax::Expr::FunctionCall { name, args }
        if name == "StandardForm" && args.len() == 1 =>
      {
        syntax::expr_to_output(&args[0])
      }
      syntax::Expr::FunctionCall { name, args }
        if name == "InputForm" && args.len() == 1 =>
      {
        syntax::expr_to_input_form(&args[0])
      }
      syntax::Expr::FunctionCall { name, args }
        if name == "Quantity" && args.len() == 2 =>
      {
        syntax::quantity_to_visual_string(&args[0], &args[1])
      }
      _ => syntax::top_level_output(&result_expr),
    }
  } else {
    syntax::top_level_output(&result_expr)
  };
  // Convert to output string (strips quotes from strings for display).
  // Use "\0" sentinel for the Null symbol so consumers can suppress it
  // without confusing it with the string "Null".
  if matches!(&result_expr, syntax::Expr::Identifier(s) if s == "Null") {
    "\0".to_string()
  } else {
    output_text
  }
}

/// If `expr` is a Grid[…] or TextGrid[…] call (possibly nested in a list),
/// render it as SVG and return `-Graphics-`. Grid/TextGrid stay symbolic
/// during evaluation so that part-assignment works; rendering only happens
/// at the output stage.  Also unwraps TraditionalForm[Grid[…]] wrappers.
fn render_grid_if_needed(expr: syntax::Expr) -> syntax::Expr {
  match &expr {
    syntax::Expr::FunctionCall { name, args }
      if (name == "Grid" || name == "TextGrid") && !args.is_empty() =>
    {
      match functions::graphics::grid_ast(args) {
        Ok(result) => result,
        Err(_) => expr,
      }
    }
    // TraditionalForm[Grid[...]] or TraditionalForm[TextGrid[...]] — unwrap and render
    syntax::Expr::FunctionCall { name, args }
      if name == "TraditionalForm"
        && args.len() == 1
        && matches!(
          &args[0],
          syntax::Expr::FunctionCall { name: inner, args: inner_args }
          if (inner == "Grid" || inner == "TextGrid") && !inner_args.is_empty()
        ) =>
    {
      if let syntax::Expr::FunctionCall {
        args: grid_args, ..
      } = &args[0]
      {
        match functions::graphics::grid_ast(grid_args) {
          Ok(result) => result,
          Err(_) => expr,
        }
      } else {
        expr
      }
    }
    // Style[Grid[...], directives...] — propagate style into grid
    syntax::Expr::FunctionCall { name, args }
      if name == "Style"
        && args.len() >= 2
        && matches!(
          &args[0],
          syntax::Expr::FunctionCall { name: inner, args: inner_args }
          if (inner == "Grid" || inner == "TextGrid") && !inner_args.is_empty()
        ) =>
    {
      if let syntax::Expr::FunctionCall {
        args: grid_args, ..
      } = &args[0]
      {
        let style = functions::graphics::parse_grid_style(&args[1..]);
        match functions::graphics::grid_ast_styled(grid_args, &style) {
          Ok(result) => result,
          Err(_) => expr,
        }
      } else {
        expr
      }
    }
    syntax::Expr::List(items) => {
      let new_items: Vec<syntax::Expr> =
        items.iter().cloned().map(render_grid_if_needed).collect();
      syntax::Expr::List(new_items.into())
    }
    _ => expr,
  }
}

/// If `expr` is a Dataset[data, …] call, render it as an SVG table
/// and return `-Graphics-`.
fn render_dataset_if_needed(expr: syntax::Expr) -> syntax::Expr {
  match &expr {
    syntax::Expr::FunctionCall { name, args }
      if name == "Dataset" && !args.is_empty() =>
    {
      let data = &args[0];
      if let Some(svg) = functions::graphics::dataset_to_svg(data) {
        graphics_result(svg)
      } else {
        expr
      }
    }
    _ => expr,
  }
}

/// If `expr` is a Tabular[data, schema] call, render it as an SVG table
/// and return `-Graphics-`.
fn render_tabular_if_needed(expr: syntax::Expr) -> syntax::Expr {
  match &expr {
    syntax::Expr::FunctionCall { name, args }
      if name == "Tabular" && args.len() >= 2 =>
    {
      let data = &args[0];
      let schema = &args[1];
      if let Some(svg) = functions::graphics::tabular_to_svg(data, schema) {
        graphics_result(svg)
      } else {
        expr
      }
    }
    _ => expr,
  }
}

/// Check if an expression is an explicit color function call (not a named color
/// identifier like `Red`). Returns `true` for `RGBColor[…]`, `Hue[…]`,
/// `GrayLevel[…]`, `Darker[…]`, `Lighter[…]`, and theme-resolved colors
/// (`LightDarkSwitched[…]`, `ThemeColor[…]`, `SystemColor[…]`).
fn is_color_function_call(expr: &syntax::Expr) -> bool {
  matches!(
    expr,
    syntax::Expr::FunctionCall { name, .. }
      if matches!(
        name.as_str(),
        "RGBColor" | "Hue" | "GrayLevel" | "Darker" | "Lighter"
          | "LightDarkSwitched" | "ThemeColor" | "SystemColor"
      )
  )
}

/// If `expr` is an explicit color specification (RGBColor, Hue, GrayLevel,
/// Darker, Lighter), render it as a 16×16 colored-square SVG swatch.
/// If it's a list, recursively convert color elements so they become
/// individual swatch graphics (the list itself is left intact for
/// `render_graphics_list_if_needed` to wrap with `{…, …}`).
/// Bare named colors (e.g. `Red`) are left as text.
fn render_color_if_needed(mut expr: syntax::Expr) -> syntax::Expr {
  if is_color_function_call(&expr)
    && let Some(color) = functions::graphics::parse_color(&expr)
  {
    let svg = functions::graphics::color_swatch_svg(&color);
    return graphics_result(svg);
  }
  if let syntax::Expr::List(ref mut items) = expr {
    let new_items: Vec<syntax::Expr> = std::mem::take(items)
      .into_iter()
      .map(render_color_if_needed)
      .collect();
    return syntax::Expr::List(new_items.into());
  }
  expr
}

/// If `expr` is an `Audio[…]` object — file-backed (`Audio[File["path"]]` /
/// `Audio["path.flac"]`) or built from raw sample data — capture it as
/// playable audio so visual hosts (the Woxi Playground and Woxi Studio)
/// render a graphical audio player, and return the `-Audio-` placeholder.
/// Audio objects that cannot be turned into a player (e.g. symbolic data)
/// are left unchanged. Visual hosts only — the CLI keeps the symbolic form.
fn render_audio_if_needed(expr: syntax::Expr) -> syntax::Expr {
  if let syntax::Expr::FunctionCall { ref name, .. } = expr
    && name == "Audio"
    && let Some(audio) = functions::sound::audio_to_output(&expr)
  {
    capture_audio(audio);
    return syntax::Expr::Identifier("-Audio-".to_string());
  }
  expr
}

/// If `expr` is a `Sound` containing one or more `Play[f, {t, …}]` segments,
/// synthesize a WAV, capture it as an `<audio>` element, and return the
/// `-Sound-` identifier. Sounds with no samplable `Play` segment are left
/// unchanged (they still render textually as `-Sound-`).
fn render_sound_if_needed(expr: syntax::Expr) -> syntax::Expr {
  if let syntax::Expr::FunctionCall { ref name, .. } = expr
    && name == "Sound"
    && let Some(wav_base64) = functions::sound::sound_to_wav_base64(&expr)
  {
    capture_sound(&wav_base64);
    return syntax::Expr::Identifier("-Sound-".to_string());
  }
  expr
}

/// If `expr` is a ComputationalMusic object (MusicNote, MusicChord,
/// MusicScale, MusicScore, …), render it as a musical-staff SVG the way
/// Mathematica displays it, capture it as graphics, and return `-Graphics-`.
/// Music objects that carry no notation (a bare `MusicDuration`, `MusicTempo`,
/// …) are left symbolic. Visual hosts only — the CLI keeps the symbolic form.
fn render_music_if_needed(expr: syntax::Expr) -> syntax::Expr {
  // A single music object draws one staff.
  if let syntax::Expr::FunctionCall { ref name, .. } = expr
    && functions::music_ast::MUSIC_OBJECT_HEADS.contains(&name.as_str())
    && let Some(svg) = functions::music_render::music_to_svg(&expr)
  {
    return graphics_result(svg);
  }
  // A plain list of music events (e.g. {MusicNote[…], MusicNote[…]}) keeps its
  // list structure: `{ <staff>, <staff>, … }`, one staff per element.
  if functions::music_ast::is_music_object_list(&expr)
    && let Some(svg) = functions::music_render::music_list_to_svg(&expr)
  {
    return graphics_result(svg);
  }
  expr
}

/// If `expr` is a `Molecule[…]`, render it as the compact information tile
/// (structure thumbnail + formula + atom/bond counts) Wolfram notebooks show,
/// and capture it as graphics. The full 2-D structure diagram is instead
/// produced by `MoleculePlot[…]`. Visual hosts only — the CLI keeps the
/// symbolic `Molecule[…]` echo to match wolframscript.
fn render_molecule_if_needed(expr: syntax::Expr) -> syntax::Expr {
  if let syntax::Expr::FunctionCall { ref name, .. } = expr
    && name == "Molecule"
    && let Some(svg) = functions::molecule_render::molecule_tile_svg(&expr)
  {
    return graphics_result_with_head(svg, "Molecule");
  }
  expr
}

/// If `expr` is a `DateObject[…]` (e.g. from RandomDate or Now), render it
/// as the framed date panel Wolfram notebooks show. Visual hosts only —
/// the CLI keeps the symbolic form. Dates with symbolic components stay
/// symbolic.
fn render_date_object_if_needed(expr: syntax::Expr) -> syntax::Expr {
  if let syntax::Expr::FunctionCall { ref name, .. } = expr
    && name == "DateObject"
    && let Some(svg) = functions::datetime_ast::date_object_panel_svg(&expr)
  {
    return graphics_result_with_head(svg, "DateObject");
  }
  expr
}

/// If `expr` is an Image, encode it as a base64 PNG `<img>` tag,
/// capture it as graphics, and return `-Image-` identifier.
fn render_image_if_needed(expr: syntax::Expr) -> syntax::Expr {
  if let syntax::Expr::Image {
    width,
    height,
    channels,
    ref data,
    ..
  } = expr
  {
    let html =
      functions::image_ast::image_to_html_img(width, height, channels, data);
    capture_graphics(&html);
    syntax::Expr::Identifier("-Image-".to_string())
  } else {
    expr
  }
}

/// Check if an expression represents a Graphics placeholder
/// (either `-Graphics-` directly or `Style[-Graphics-, ...]`)
fn is_graphics_placeholder(expr: &syntax::Expr) -> bool {
  match expr {
    syntax::Expr::Identifier(s) if s == "-Graphics-" || s == "-Image-" => true,
    syntax::Expr::Graphics { .. } => true,
    syntax::Expr::FunctionCall { name, args }
      if name == "Style" && !args.is_empty() =>
    {
      is_graphics_placeholder(&args[0])
    }
    _ => false,
  }
}

/// Check if an expression tree contains any Graphics placeholder
fn contains_graphics_placeholder(expr: &syntax::Expr) -> bool {
  if is_graphics_placeholder(expr) {
    return true;
  }
  match expr {
    syntax::Expr::List(items) => {
      items.iter().any(contains_graphics_placeholder)
    }
    syntax::Expr::FunctionCall { args, .. } => {
      args.iter().any(contains_graphics_placeholder)
    }
    _ => false,
  }
}

/// Check if a list's items form a 3D structure (list of lists of lists)
fn is_3d_list(items: &[syntax::Expr]) -> bool {
  !items.is_empty()
    && items.iter().all(|item| {
      if let syntax::Expr::List(sub) = item {
        !sub.is_empty()
          && sub.iter().all(|s| matches!(s, syntax::Expr::List(_)))
      } else {
        false
      }
    })
}

/// If `expr` is a TableForm[list] with non-graphics data, render as a Grid SVG.
/// This is only called from `interpret_with_stdout` (visual contexts),
/// not from plain `interpret` (where TableForm stays symbolic).
fn render_tableform_if_needed(expr: syntax::Expr) -> syntax::Expr {
  match &expr {
    syntax::Expr::FunctionCall { name, args }
      if name == "TableForm" && !args.is_empty() =>
    {
      let data = &args[0];
      // Skip if content contains Graphics placeholders (handled by render_graphics_list_if_needed)
      if contains_graphics_placeholder(data) {
        return expr;
      }
      // Build grid data and optional group gap indices
      let (grid_data, group_gaps) = match data {
        syntax::Expr::List(items) if is_3d_list(items) => {
          // 3D list M[dim1][dim2][dim3]:
          // Each block M[i] is transposed (sub-lists become columns),
          // then blocks are stacked vertically.
          let mut rows: Vec<syntax::Expr> = Vec::new();
          let mut gaps: Vec<usize> = Vec::new();
          for (bi, block) in items.iter().enumerate() {
            if let syntax::Expr::List(sub_lists) = block {
              if bi > 0 {
                gaps.push(rows.len());
              }
              let dim3 = sub_lists
                .iter()
                .map(|sl| {
                  if let syntax::Expr::List(v) = sl {
                    v.len()
                  } else {
                    0
                  }
                })
                .max()
                .unwrap_or(0);
              for k in 0..dim3 {
                let row: Vec<syntax::Expr> = sub_lists
                  .iter()
                  .map(|sl| {
                    if let syntax::Expr::List(v) = sl {
                      v.get(k)
                        .cloned()
                        .unwrap_or(syntax::Expr::Identifier(String::new()))
                    } else {
                      sl.clone()
                    }
                  })
                  .collect();
                rows.push(syntax::Expr::List(row.into()));
              }
            }
          }
          (syntax::Expr::List(rows.into()), gaps)
        }
        syntax::Expr::List(items)
          if items
            .iter()
            .all(|item| matches!(item, syntax::Expr::List(_))) =>
        {
          (data.clone(), vec![])
        }
        syntax::Expr::List(items) if !items.is_empty() => {
          // 1D list — wrap each element in a single-element list (column)
          (
            syntax::Expr::List(
              items
                .iter()
                .map(|e| syntax::Expr::List(vec![e.clone()].into()))
                .collect(),
            ),
            vec![],
          )
        }
        _ => return expr,
      };
      // Forward extra args (options like TableHeadings) to grid rendering.
      // TableForm defaults to left alignment (unlike Grid which centers).
      let mut grid_args = vec![grid_data];
      let has_alignment = args[1..].iter().any(|a| {
        matches!(a, syntax::Expr::Rule { pattern, .. }
          if matches!(pattern.as_ref(), syntax::Expr::Identifier(n) if n == "Alignment"))
      });
      if !has_alignment {
        grid_args.push(syntax::Expr::Rule {
          pattern: Box::new(syntax::Expr::Identifier("Alignment".into())),
          replacement: Box::new(syntax::Expr::Identifier("Left".into())),
        });
      }
      grid_args.extend(args[1..].iter().cloned());
      let result = if group_gaps.is_empty() {
        functions::graphics::grid_ast(&grid_args)
      } else {
        functions::graphics::grid_ast_with_gaps(&grid_args, &group_gaps)
      };
      match result {
        Ok(result) => result,
        Err(_) => expr,
      }
    }
    _ => expr,
  }
}

/// If `expr` is a MatrixForm[list] with non-graphics data, render as a Grid SVG.
/// For 2D lists, render as a parenthesized matrix.
/// For 3D lists, render each sub-matrix as a parenthesized matrix, stacked vertically.
/// For 1D lists, render as a column vector with parentheses.
fn render_matrixform_if_needed(expr: syntax::Expr) -> syntax::Expr {
  match &expr {
    syntax::Expr::FunctionCall { name, args }
      if name == "MatrixForm" && args.len() == 1 =>
    {
      let data = &args[0];
      if contains_graphics_placeholder(data) {
        return expr;
      }
      match data {
        // 3D list: 2D grid of column vectors, each parenthesized, with outer parens
        syntax::Expr::List(items) if is_3d_list(items) => {
          // Build outer_rows: Vec<Vec<Expr>> where each inner Expr is a sub-list
          let outer_rows: Vec<Vec<syntax::Expr>> = items
            .iter()
            .map(|row| {
              if let syntax::Expr::List(cells) = row {
                cells.to_vec()
              } else {
                vec![row.clone()]
              }
            })
            .collect();
          match functions::graphics::matrixform_3d_ast(&outer_rows) {
            Ok(result) => result,
            Err(_) => expr,
          }
        }
        // 2D list: single matrix
        syntax::Expr::List(items)
          if items
            .iter()
            .all(|item| matches!(item, syntax::Expr::List(_))) =>
        {
          match functions::graphics::grid_ast_with_parens(&[data.clone()]) {
            Ok(result) => result,
            Err(_) => expr,
          }
        }
        // 1D list: column vector
        syntax::Expr::List(items) if !items.is_empty() => {
          let grid_data = syntax::Expr::List(
            items
              .iter()
              .map(|e| syntax::Expr::List(vec![e.clone()].into()))
              .collect(),
          );
          match functions::graphics::grid_ast_with_parens(&[grid_data]) {
            Ok(result) => result,
            Err(_) => expr,
          }
        }
        _ => expr,
      }
    }
    _ => expr,
  }
}

/// If `expr` is `TraditionalForm[{…}]`, render the list as a parenthesized matrix
/// (same visual treatment as MatrixForm).
fn render_traditionalform_list_if_needed(expr: syntax::Expr) -> syntax::Expr {
  match &expr {
    syntax::Expr::FunctionCall { name, args }
      if name == "TraditionalForm"
        && args.len() == 1
        && matches!(&args[0], syntax::Expr::List(_)) =>
    {
      let data = &args[0];
      if contains_graphics_placeholder(data) {
        return expr;
      }
      match data {
        // 2D list: matrix with parentheses
        syntax::Expr::List(items)
          if items
            .iter()
            .all(|item| matches!(item, syntax::Expr::List(_))) =>
        {
          match functions::graphics::grid_ast_with_parens(&[data.clone()]) {
            Ok(result) => result,
            Err(_) => expr,
          }
        }
        // 1D list: column vector with parentheses
        syntax::Expr::List(items) if !items.is_empty() => {
          let grid_data = syntax::Expr::List(
            items
              .iter()
              .map(|e| syntax::Expr::List(vec![e.clone()].into()))
              .collect(),
          );
          match functions::graphics::grid_ast_with_parens(&[grid_data]) {
            Ok(result) => result,
            Err(_) => expr,
          }
        }
        _ => expr,
      }
    }
    _ => expr,
  }
}

/// Pre-render display wrappers (TableForm, MatrixForm, Grid, Framed, Row,
/// nested Column, Dataset, …) inside an arbitrary expression so they appear
/// as real `Expr::Graphics` sub-SVGs when embedded into a parent layout
/// (e.g. items of a `Column[…]`). Plain text / numeric items are returned
/// unchanged.
fn render_inline_display_wrapper(expr: syntax::Expr) -> syntax::Expr {
  // The top-level pipeline transformations don't recurse into nested
  // wrapper arguments, so we apply the relevant ones here for a single
  // sub-expression.
  let expr = render_grid_if_needed(expr);
  let expr = render_dataset_if_needed(expr);
  let expr = render_tabular_if_needed(expr);
  let expr = render_tableform_if_needed(expr);
  let expr = render_matrixform_if_needed(expr);
  let expr = render_traditionalform_list_if_needed(expr);
  let expr = render_column_if_needed(expr);
  let expr = render_row_if_needed(expr);
  let expr = render_treeform_if_needed(expr);
  let expr = render_framed_if_needed(expr);
  let expr = render_highlighted_if_needed(expr);
  // Raw `Graphics[…]` / `Graphics3D[…]` items (e.g. from Plot or
  // PolyhedronData) are rendered to embedded SVG so a `Column[{plot, …}]`
  // shows the actual graphic instead of a `-Graphics-` text placeholder.
  render_graphics_fc_if_needed(expr)
}

/// If `expr` is `Column[{…}]`, render it as an SVG column and return `-Graphics-`.
fn render_column_if_needed(expr: syntax::Expr) -> syntax::Expr {
  match &expr {
    syntax::Expr::FunctionCall { name, args }
      if name == "Column" && !args.is_empty() =>
    {
      // Pre-render display wrappers inside the column's items so e.g.
      // `Column[{"hi", TableForm[{{1,2},{3,4}}]}]` shows an actual grid
      // rather than the raw `TableForm[…]` text.
      let mut new_args: Vec<syntax::Expr> = args.to_vec();
      if let syntax::Expr::List(items) = &args[0] {
        let new_items: Vec<syntax::Expr> = items
          .iter()
          .cloned()
          .map(render_inline_display_wrapper)
          .collect();
        new_args[0] = syntax::Expr::List(new_items.into());
      }
      if let Some(svg) = functions::graphics::column_to_svg(&new_args) {
        graphics_result(svg)
      } else {
        expr
      }
    }
    _ => expr,
  }
}

/// If `expr` is `Row[{…}]` or `Row[{…}, sep]`, render as a horizontal SVG row.
fn render_row_if_needed(expr: syntax::Expr) -> syntax::Expr {
  match &expr {
    syntax::Expr::FunctionCall { name, args }
      if name == "Row" && !args.is_empty() =>
    {
      if let Some(svg) = row_svg_with_rendered_items(args) {
        graphics_result(svg)
      } else {
        expr
      }
    }
    _ => expr,
  }
}

/// Pre-render display wrappers inside a `Row[…]`'s items (so e.g.
/// `Row[{Graphics[…], Framed[x]}]` embeds actual graphics rather than
/// their textual echoes) and render the whole row as a horizontal SVG.
/// `None` when the arguments don't form a renderable row.
pub(crate) fn row_svg_with_rendered_items(
  args: &[syntax::Expr],
) -> Option<String> {
  let mut new_args: Vec<syntax::Expr> = args.to_vec();
  if let syntax::Expr::List(items) = &args[0] {
    let new_items: Vec<syntax::Expr> = items
      .iter()
      .cloned()
      .map(render_inline_display_wrapper)
      .collect();
    new_args[0] = syntax::Expr::List(new_items.into());
  }
  functions::graphics::row_to_svg(&new_args)
}

/// If `expr` is a `Framed[expr]` call, render it as an SVG box with a border
/// and return `-Graphics-`. If the expression is a list containing any
/// `Framed` elements, render the whole list as a Row-like Grid so all
/// elements appear together in a single graphical output.
fn render_framed_if_needed(expr: syntax::Expr) -> syntax::Expr {
  match &expr {
    syntax::Expr::FunctionCall { name, args }
      if name == "Framed" && !args.is_empty() =>
    {
      if let Some(svg) = functions::graphics::framed_to_svg(args) {
        graphics_result(svg)
      } else {
        expr
      }
    }
    syntax::Expr::List(items) if items.iter().any(contains_framed) => {
      // Render the whole list as a Row-style SVG so all items
      // (text and Framed) appear together in one graphic.
      if let Some(svg) = functions::graphics::row_with_framed_to_svg(items) {
        graphics_result(svg)
      } else {
        expr
      }
    }
    _ => expr,
  }
}

/// Check if an expression is or contains a Framed call.
fn contains_framed(expr: &syntax::Expr) -> bool {
  match expr {
    syntax::Expr::FunctionCall { name, .. } if name == "Framed" => true,
    syntax::Expr::List(items) => items.iter().any(contains_framed),
    _ => false,
  }
}

/// If `expr` is a `Highlighted[expr]` (or `Highlighted[expr, color]`) call,
/// render it as an SVG box with a colored background and return `-Graphics-`.
/// If the expression is a list containing any `Highlighted` elements, render
/// the whole list as a Row-like layout so all elements appear together.
fn render_highlighted_if_needed(expr: syntax::Expr) -> syntax::Expr {
  match &expr {
    syntax::Expr::FunctionCall { name, args }
      if name == "Highlighted" && !args.is_empty() =>
    {
      if let Some(svg) = functions::graphics::highlighted_to_svg(args) {
        graphics_result(svg)
      } else {
        expr
      }
    }
    syntax::Expr::List(items) if items.iter().any(contains_highlighted) => {
      if let Some(svg) = functions::graphics::row_with_framed_to_svg(items) {
        graphics_result(svg)
      } else {
        expr
      }
    }
    _ => expr,
  }
}

/// Check if an expression is or contains a Highlighted call.
fn contains_highlighted(expr: &syntax::Expr) -> bool {
  match expr {
    syntax::Expr::FunctionCall { name, .. } if name == "Highlighted" => true,
    syntax::Expr::List(items) => items.iter().any(contains_highlighted),
    _ => false,
  }
}

/// If `expr` is `TreeForm[...]`, render it as a tree diagram graphic.
fn render_treeform_if_needed(expr: syntax::Expr) -> syntax::Expr {
  match &expr {
    syntax::Expr::FunctionCall { name, args }
      if name == "TreeForm" && !args.is_empty() =>
    {
      match functions::tree_form::tree_form_ast(args) {
        Ok(result) => result,
        Err(_) => expr,
      }
    }
    _ => expr,
  }
}

/// If `expr` is a top-level `PolarCurve[…]` or `FilledPolarCurve[…]` call,
/// render it as a Graphics SVG (visual mode only — the CLI keeps the
/// symbolic form to match wolframscript). Invalid curve arguments stay
/// symbolic.
fn render_polar_curve_if_needed(expr: syntax::Expr) -> syntax::Expr {
  match &expr {
    syntax::Expr::FunctionCall { name, args }
      if (name == "PolarCurve" || name == "FilledPolarCurve")
        && !args.is_empty() =>
    {
      functions::graphics::polar_curve_to_graphics(name, args).unwrap_or(expr)
    }
    _ => expr,
  }
}

/// If `expr` is an unevaluated `Graphics[{...}]` or `Graphics3D[{...}]`
/// FunctionCall (e.g. returned by VoronoiMesh), render it to SVG.
/// Also recurses into lists and wrapper forms so that e.g. `{Graphics[...], Graphics[...]}`
/// and `TableForm[{Graphics[...], ...}]` are rendered correctly.
fn render_graphics_fc_if_needed(expr: syntax::Expr) -> syntax::Expr {
  match &expr {
    syntax::Expr::FunctionCall { name, args }
      if name == "Graphics" && !args.is_empty() =>
    {
      if let Ok(rendered) = functions::graphics::graphics_ast(args) {
        rendered
      } else {
        expr
      }
    }
    syntax::Expr::FunctionCall { name, args }
      if name == "Graphics3D" && !args.is_empty() =>
    {
      if let Ok(rendered) = functions::plot3d::graphics3d_ast(args) {
        rendered
      } else {
        expr
      }
    }
    syntax::Expr::FunctionCall { name, args }
      if name == "Graph" && args.len() >= 2 =>
    {
      // wolframscript summarises Graph as `Graph[<n>, <m>]` instead of
      // the `-Graphics-` placeholder. Render the SVG (so jupyter/HTML
      // surfaces still get it via capture_graphics) but keep the Expr
      // as the original FunctionCall so expr_to_string can print the
      // summary. ExportString[Graph[…], "SVG"] re-invokes graph_ast
      // directly, so it does not depend on this auto-render step.
      if let Ok(rendered) = functions::graph::graph_ast(args)
        && let syntax::Expr::Graphics { ref svg, .. } = rendered
      {
        capture_graphics(svg);
      }
      expr
    }
    syntax::Expr::FunctionCall { name, args }
      if name == "Region" && !args.is_empty() =>
    {
      // Region[reg, opts…] displays as a plot of the region (embedding
      // dimension 2 or 3), like in Wolfram notebooks. Unsupported or
      // symbolic regions keep the textual Region[…] echo.
      if let Some(rendered) = functions::region::region_to_graphics(args) {
        rendered
      } else {
        expr
      }
    }
    syntax::Expr::FunctionCall { name, args }
      if name == "MeshRegion" && args.len() == 2 =>
    {
      // Render MeshRegion as SVG (e.g. from VoronoiMesh)
      if let Some(svg) =
        functions::voronoi::mesh_region_to_svg(&args[0], &args[1])
      {
        capture_graphics(&svg);
        syntax::Expr::Graphics {
          svg,
          is_3d: false,
          source: None,
          head: None,
        }
      } else {
        expr
      }
    }
    syntax::Expr::List(items) => {
      let new_items: Vec<syntax::Expr> = items
        .iter()
        .cloned()
        .map(render_graphics_fc_if_needed)
        .collect();
      syntax::Expr::List(new_items.into())
    }
    syntax::Expr::FunctionCall { name, args }
      if matches!(
        name.as_str(),
        "TableForm"
          | "MatrixForm"
          | "Column"
          | "Row"
          | "Style"
          | "MathMLForm"
          | "StandardForm"
          | "InputForm"
          | "OutputForm"
          | "Plus"
          | "Times"
          | "Power"
      ) =>
    {
      let new_args: Vec<syntax::Expr> = args
        .iter()
        .cloned()
        .map(render_graphics_fc_if_needed)
        .collect();
      syntax::Expr::FunctionCall {
        name: name.clone(),
        args: new_args.into(),
      }
    }
    // Arithmetic BinaryOps (Plus, Times, Power, etc.) can carry a
    // Graphics[...] FunctionCall as a child — wolframscript still
    // summarizes that inner graphic as `-Graphics-`, so recurse.
    syntax::Expr::BinaryOp { op, left, right } => syntax::Expr::BinaryOp {
      op: *op,
      left: Box::new(render_graphics_fc_if_needed(*left.clone())),
      right: Box::new(render_graphics_fc_if_needed(*right.clone())),
    },
    syntax::Expr::UnaryOp { op, operand } => syntax::Expr::UnaryOp {
      op: *op,
      operand: Box::new(render_graphics_fc_if_needed(*operand.clone())),
    },
    _ => expr,
  }
}

/// If the result is a list (1D, 2D, or 3D) of `-Graphics-` items,
/// or a `TableForm` wrapping such a list, combine captured SVGs into a grid.
fn render_graphics_list_if_needed(expr: syntax::Expr) -> syntax::Expr {
  // Unwrap TableForm[list] or MathMLForm[TableForm[list]] etc.
  let has_tableform = has_form_wrapper(&expr, "TableForm");
  let inner = unwrap_form_wrappers(&expr);

  let all_svgs = get_all_captured_graphics();
  if all_svgs.is_empty() {
    return expr;
  }

  // 1D list of Graphics
  if let syntax::Expr::List(items) = inner {
    if items.iter().all(is_graphics_placeholder)
      && items.len() > 1
      && items.len() <= all_svgs.len()
    {
      // Take the last N SVGs (they correspond to the list items)
      let start = all_svgs.len() - items.len();
      let row: Vec<String> = all_svgs[start..].to_vec();
      if let Some(combined) = functions::graphics::graphics_list_svg(&row) {
        // Clear and re-capture with the combined SVG
        clear_captured_graphics();
        return graphics_result(combined);
      }
    }

    // 2D list: list of lists of Graphics
    if items.iter().all(|e| {
      if let syntax::Expr::List(inner) = e {
        inner.iter().all(is_graphics_placeholder) && !inner.is_empty()
      } else {
        false
      }
    }) && !items.is_empty()
    {
      let total_cells: usize = items
        .iter()
        .map(|e| {
          if let syntax::Expr::List(inner) = e {
            inner.len()
          } else {
            0
          }
        })
        .sum();
      if total_cells <= all_svgs.len() {
        let start = all_svgs.len() - total_cells;
        let mut offset = start;
        let mut rows: Vec<Vec<String>> = Vec::new();
        for item in items {
          if let syntax::Expr::List(inner) = item {
            let row: Vec<String> =
              all_svgs[offset..offset + inner.len()].to_vec();
            offset += inner.len();
            rows.push(row);
          }
        }
        if let Some(combined) =
          functions::graphics::combine_graphics_svgs(&rows)
        {
          clear_captured_graphics();
          return graphics_result(combined);
        }
      }
    }

    // 3D list: list of lists of lists of Graphics
    // Structure: items[dim1][dim2][dim3]
    if items.iter().all(|e| {
      if let syntax::Expr::List(rows) = e {
        rows.iter().all(|r| {
          if let syntax::Expr::List(cols) = r {
            cols.iter().all(is_graphics_placeholder) && !cols.is_empty()
          } else {
            false
          }
        }) && !rows.is_empty()
      } else {
        false
      }
    }) && !items.is_empty()
    {
      let total_cells: usize = items
        .iter()
        .map(|e| {
          if let syntax::Expr::List(rows) = e {
            rows
              .iter()
              .map(|r| {
                if let syntax::Expr::List(cols) = r {
                  cols.len()
                } else {
                  0
                }
              })
              .sum()
          } else {
            0
          }
        })
        .sum();
      if total_cells <= all_svgs.len() {
        let start = all_svgs.len() - total_cells;

        // Collect SVGs into 3D structure [dim1][dim2][dim3]
        let mut offset = start;
        let mut svg_3d: Vec<Vec<Vec<String>>> = Vec::new();
        for item in items {
          if let syntax::Expr::List(inner_rows) = item {
            let mut block: Vec<Vec<String>> = Vec::new();
            for r in inner_rows {
              if let syntax::Expr::List(cols) = r {
                let mut row_svgs: Vec<String> = Vec::new();
                for _ in cols {
                  if offset < all_svgs.len() {
                    row_svgs.push(all_svgs[offset].clone());
                    offset += 1;
                  }
                }
                block.push(row_svgs);
              }
            }
            svg_3d.push(block);
          }
        }

        let rows: Vec<Vec<String>> = if has_tableform {
          // TableForm: transpose each block (dim3→rows, dim2→cols),
          // stack blocks vertically
          let mut rows = Vec::new();
          for block in &svg_3d {
            let dim3 = block.iter().map(|r| r.len()).max().unwrap_or(0);
            for k in 0..dim3 {
              let row: Vec<String> = block
                .iter()
                .map(|sub| sub.get(k).cloned().unwrap_or_default())
                .collect();
              rows.push(row);
            }
          }
          rows
        } else {
          // No TableForm: one row per dim1, flatten dim2×dim3 as columns
          svg_3d
            .into_iter()
            .map(|block| block.into_iter().flatten().collect())
            .collect()
        };

        if let Some(combined) =
          functions::graphics::combine_graphics_svgs(&rows)
        {
          clear_captured_graphics();
          return graphics_result(combined);
        }
      }
    }
  }

  expr
}

/// Check if an expression has a specific form wrapper (e.g. "TableForm")
fn has_form_wrapper(expr: &syntax::Expr, target: &str) -> bool {
  match expr {
    syntax::Expr::FunctionCall { name, args }
      if args.len() == 1
        && matches!(
          name.as_str(),
          "TableForm"
            | "MathMLForm"
            | "StandardForm"
            | "InputForm"
            | "OutputForm"
            | "TraditionalForm"
        ) =>
    {
      name == target || has_form_wrapper(&args[0], target)
    }
    _ => false,
  }
}

/// Unwrap form wrappers like TableForm, MathMLForm, StandardForm, etc.
fn unwrap_form_wrappers(expr: &syntax::Expr) -> &syntax::Expr {
  match expr {
    syntax::Expr::FunctionCall { name, args }
      if args.len() == 1
        && matches!(
          name.as_str(),
          "TableForm"
            | "MathMLForm"
            | "StandardForm"
            | "InputForm"
            | "OutputForm"
            | "TraditionalForm"
        ) =>
    {
      unwrap_form_wrappers(&args[0])
    }
    _ => expr,
  }
}

/// Generate an SVG rendering of the result expression and capture it.
/// This is used by the playground to display all results with proper formatting.
/// Upper bound on the number of atomic tokens the output-SVG typesetter
/// will process. A larger result (e.g. a full `Import` of a big CSV) would
/// produce a multi-hundred-MB SVG that no host can usefully display; the
/// visual hosts fall back to the plain-text rendering when no output SVG is
/// captured.
const OUTPUT_SVG_MAX_TOKENS: usize = 20_000;

/// Count atomic tokens in `expr`, stopping early once `budget` is
/// exhausted. Returns true when the expression is larger than the budget.
fn expr_exceeds_token_budget(expr: &syntax::Expr, budget: &mut usize) -> bool {
  expr_exceeds_token_budget_depth(expr, budget, 0)
}

/// Deeply-nested results (e.g. `Nest[f, x, 500]`) are treated as exceeding the
/// budget once they pass `MAX_SVG_DEPTH`. This both bounds this counter's own
/// recursion and makes `generate_output_svg` skip typesetting expressions too
/// deep for the (unbounded-recursion) box renderer — which would otherwise
/// overflow the stack.
fn expr_exceeds_token_budget_depth(
  expr: &syntax::Expr,
  budget: &mut usize,
  depth: usize,
) -> bool {
  use syntax::Expr;
  const MAX_SVG_DEPTH: usize = 256;
  if *budget == 0 || depth > MAX_SVG_DEPTH {
    return true;
  }
  let d = depth + 1;
  match expr {
    Expr::List(items) => items
      .iter()
      .any(|e| expr_exceeds_token_budget_depth(e, budget, d)),
    Expr::FunctionCall { args, .. } => {
      *budget = budget.saturating_sub(1);
      args
        .iter()
        .any(|e| expr_exceeds_token_budget_depth(e, budget, d))
    }
    Expr::Association(pairs) => pairs.iter().any(|(k, v)| {
      expr_exceeds_token_budget_depth(k, budget, d)
        || expr_exceeds_token_budget_depth(v, budget, d)
    }),
    Expr::BinaryOp { left, right, .. } => {
      *budget = budget.saturating_sub(1);
      expr_exceeds_token_budget_depth(left, budget, d)
        || expr_exceeds_token_budget_depth(right, budget, d)
    }
    Expr::UnaryOp { operand, .. } => {
      *budget = budget.saturating_sub(1);
      expr_exceeds_token_budget_depth(operand, budget, d)
    }
    _ => {
      *budget = budget.saturating_sub(1);
      false
    }
  }
}

fn generate_output_svg(expr: &syntax::Expr) {
  // Very large results are not typeset: the box conversion + glyph layout +
  // SVG string would each be orders of magnitude bigger than the data
  // itself (this made large CSV `Import`s appear to hang in the browser).
  let mut budget = OUTPUT_SVG_MAX_TOKENS;
  if expr_exceeds_token_budget(expr, &mut budget) {
    return;
  }
  // Skip for Graphics/Image results (they already have captured SVG/HTML)
  if matches!(expr, syntax::Expr::Identifier(s) if s == "-Graphics-" || s == "-Graphics3D-" || s == "-Image-")
    || matches!(expr, syntax::Expr::Graphics { .. })
    || matches!(expr, syntax::Expr::Image { .. })
  {
    return;
  }
  // Skip for FullForm results — display as plain text in the playground
  if matches!(expr, syntax::Expr::FunctionCall { name, args } if name == "FullForm" && args.len() == 1)
  {
    return;
  }
  // QuestionObject renders as a question panel (prompt, answer choices,
  // Submit button) rather than as typeset expression text.
  if let Some(svg) = functions::assessment_render::question_object_to_svg(expr)
  {
    capture_output_svg(&svg);
    return;
  }
  // Skip SVG for InputForm results — display as plain text
  if matches!(expr, syntax::Expr::FunctionCall { name, args } if name == "InputForm" && args.len() == 1)
  {
    return;
  }
  // Skip SVG for CForm/TeXForm/FortranForm — display converted text as plain text
  if matches!(expr, syntax::Expr::FunctionCall { name, args }
    if (name == "CForm" || name == "TeXForm" || name == "FortranForm") && args.len() == 1)
  {
    return;
  }
  // Unwrap StandardForm/TraditionalForm — render SVG for the inner expression.
  // TraditionalForm routes through the dedicated traditional typesetter so
  // math renders in conventional notation (∑/∫ operators, invisible HoldForm,
  // π/∞ glyphs, `sin(x)` instead of `Sin[x]`, …) rather than the literal
  // StandardForm box tree.
  let mut traditional = false;
  let expr = if let syntax::Expr::FunctionCall { name, args } = expr {
    if (name == "StandardForm" || name == "TraditionalForm") && args.len() == 1
    {
      traditional = name == "TraditionalForm";
      &args[0]
    } else {
      expr
    }
  } else {
    expr
  };
  // Convert expression to box form, then render boxes to SVG.
  // RawBoxes[...] and DisplayForm[...] pass their contents directly as boxes.
  let boxes = if let syntax::Expr::FunctionCall { name, args } = expr {
    if (name == "RawBoxes" || name == "DisplayForm") && args.len() == 1 {
      args[0].clone()
    } else if traditional {
      evaluator::dispatch::complex_and_special::expr_to_box_form_traditional(
        expr,
      )
    } else {
      evaluator::dispatch::complex_and_special::expr_to_box_form(expr)
    }
  } else if traditional {
    evaluator::dispatch::complex_and_special::expr_to_box_form_traditional(expr)
  } else {
    evaluator::dispatch::complex_and_special::expr_to_box_form(expr)
  };
  // The box form of a machine/arbitrary-precision Real carries a trailing
  // backtick precision marker (e.g. `2.``), matching wolframscript's textual
  // `MakeBoxes` output. In a typeset display — which the Playground/Studio
  // SVG emulates — that marker is suppressed, so strip it before layout.
  let boxes = strip_number_precision_markers(&boxes);
  let layout = functions::graphics::layout_box(&boxes, 14.0);
  let text_fill = functions::graphics::theme().text_primary;
  let svg = functions::graphics::layout_to_svg(&layout, text_fill);
  capture_output_svg(&svg);
}

/// Rewrite numeric box-String leaves to their typeset notebook form: the
/// trailing precision marker is dropped (so a machine real shows `2.` instead
/// of `` 2.` ``) and an arbitrary-precision real is truncated to its precision
/// in significant digits (so `` N[Pi, 3] `` shows `3.14`, not all 20 digits).
/// Recurses through the box tree (RowBox/SuperscriptBox/… → List args).
fn strip_number_precision_markers(expr: &syntax::Expr) -> syntax::Expr {
  use syntax::Expr;
  match expr {
    Expr::String(s) => {
      let display = precision_number_display(s).unwrap_or_else(|| s.clone());
      // A machine/real Real in `mantissa*^exp` scientific notation is stored as
      // a literal InputForm string (`1.*^10`). Typeset it as `1. × 10^exp` so
      // the Playground/Studio SVG shows a superscript exponent instead of the
      // raw `*^`, matching the notebook front-end.
      match scientific_string_to_box(&display) {
        Some(boxed) => boxed,
        // A plain (non-scientific) number gets thin-space digit grouping for
        // the notebook-style display (`10000000000` → `10 000 000 000`). The
        // scientific mantissa above is intentionally left ungrouped, matching
        // the Wolfram notebook.
        None => Expr::String(functions::graphics::group_digits_str(&display)),
      }
    }
    Expr::List(items) => Expr::List(
      items
        .iter()
        .map(strip_number_precision_markers)
        .collect::<Vec<_>>()
        .into(),
    ),
    Expr::FunctionCall { name, args } => Expr::FunctionCall {
      name: name.clone(),
      args: args
        .iter()
        .map(strip_number_precision_markers)
        .collect::<Vec<_>>()
        .into(),
    },
    other => other.clone(),
  }
}

/// If `s` is a real number in `mantissa*^exp` scientific notation, build the
/// typeset box `RowBox[{mantissa, " × ", SuperscriptBox["10", exp]}]` so the
/// Playground/Studio SVG renders `1. × 10^10` (with a superscript exponent)
/// instead of the literal InputForm `1.*^10`. The exponent is an optionally
/// signed integer; anything else (or no `*^` factor) returns `None`.
fn scientific_string_to_box(s: &str) -> Option<syntax::Expr> {
  use syntax::Expr;
  let idx = s.find("*^")?;
  let mantissa = &s[..idx];
  let exp = &s[idx + 2..];
  if mantissa.is_empty() || exp.is_empty() {
    return None;
  }
  let exp_digits = exp.strip_prefix('-').unwrap_or(exp);
  if exp_digits.is_empty() || !exp_digits.bytes().all(|b| b.is_ascii_digit()) {
    return None;
  }
  Some(Expr::FunctionCall {
    name: "RowBox".to_string(),
    args: vec![Expr::List(
      vec![
        Expr::String(mantissa.to_string()),
        Expr::String(" \u{00d7} ".to_string()),
        Expr::FunctionCall {
          name: "SuperscriptBox".to_string(),
          args: vec![
            Expr::String("10".to_string()),
            Expr::String(exp.to_string()),
          ]
          .into(),
        },
      ]
      .into(),
    )]
    .into(),
  })
}

/// Convert a single numeric token in Wolfram backtick notation to its typeset
/// notebook display form. Returns `None` when `token` is not a number with a
/// precision marker, so symbol context names like `` Global` `` (which start
/// with a letter) are left untouched.
///
/// - A machine real (bare backtick, `` 2.` ``) just loses the marker → `2.`.
/// - An arbitrary-precision real (`` 3.1415…`3. ``) is truncated to its
///   precision in significant figures → `3.14`.
/// - A `*^exp` scientific suffix is preserved (`` 1.23`5.*^6 `` → `1.23*^6`).
fn precision_number_display(token: &str) -> Option<String> {
  // Numeric leaves start with a digit (negatives are wrapped in a RowBox
  // with a separate "-" token, so the magnitude leaf is unsigned).
  if !token.as_bytes().first().is_some_and(u8::is_ascii_digit) {
    return None;
  }
  let tick = token.find('`')?;
  // The backtick must immediately follow a digit or decimal point.
  let prev = token[..tick].chars().next_back()?;
  if !(prev.is_ascii_digit() || prev == '.') {
    return None;
  }
  let mantissa = &token[..tick];
  // Skip the second backtick of an accuracy form (`` 0``5. ``).
  let after = &token[tick + 1..];
  let after = after.strip_prefix('`').unwrap_or(after);
  // The precision spec is the run of digits/dots after the backtick(s);
  // anything else (e.g. a `*^exp` scientific suffix) is preserved.
  let spec_end = after
    .find(|c: char| !(c.is_ascii_digit() || c == '.'))
    .unwrap_or(after.len());
  let prec_spec = &after[..spec_end];
  let suffix = &after[spec_end..];
  // A parseable precision → round to that many significant digits, keeping
  // trailing zeros (an arbitrary-precision real shows every requested figure).
  // A bare backtick marks a machine-precision real: the notebook front end
  // shows those at 6 significant figures with trailing zeros dropped
  // (`N[Pi]` → `3.14159`, `0.1 + 0.2` → `0.3`), so round to 6 and trim.
  let body = match prec_spec.parse::<f64>() {
    Ok(prec) => {
      let digits = (prec.round() as i64).max(1) as usize;
      round_significant(mantissa, digits)
    }
    Err(_) => drop_trailing_frac_zeros(&round_significant(mantissa, 6)),
  };
  Some(format!("{}{}", body, suffix))
}

/// Round the unsigned decimal `mantissa` (e.g. `"3.1415926"`, `"0.00123"`, no
/// sign or precision marker) to `prec` significant figures for notebook
/// display, matching Wolfram's `N[Pi, 4]` → `3.142`. Magnitude is preserved
/// with placeholder zeros in the integer part (`314.159` → 2 sig figs →
/// `310.`), rounding carries propagate (`9.99` → 2 sig figs → `10.`), and a
/// trailing decimal point is always kept so an approximate real still reads as
/// `3.` rather than `3`. When `prec` exceeds the digits available, every
/// stored digit is shown unrounded.
pub(crate) fn round_significant(mantissa: &str, prec: usize) -> String {
  let prec = prec.max(1);
  let (int_s, frac_s) = match mantissa.find('.') {
    Some(i) => (&mantissa[..i], &mantissa[i + 1..]),
    None => (mantissa, ""),
  };
  // Flatten to a digit stream and record the decimal point position (the
  // number of integer digits), so each digit's place value is known.
  let digits: Vec<u8> = int_s
    .chars()
    .chain(frac_s.chars())
    .filter_map(|c| c.to_digit(10).map(|d| d as u8))
    .collect();
  let point_pos = int_s.chars().filter(char::is_ascii_digit).count() as isize;

  // Index of the first significant (non-zero) digit; a zero value shows "0.".
  let Some(fs) = digits.iter().position(|&d| d != 0) else {
    return "0.".to_string();
  };

  let avail = digits.len() - fs;
  let keep = prec.min(avail);
  let mut kept = digits[fs..fs + keep].to_vec();
  // Place-value exponent of the first kept digit (10^lead_exp).
  let mut lead_exp = point_pos - 1 - fs as isize;

  // Round half-up using the first dropped digit, propagating any carry. A
  // carry out of the front (all nines) grows the magnitude by one place.
  if keep < avail && digits[fs + keep] >= 5 {
    let mut i = kept.len();
    loop {
      if i == 0 {
        kept.insert(0, 1);
        kept.truncate(prec); // drop the now-redundant trailing zero
        lead_exp += 1;
        break;
      }
      i -= 1;
      if kept[i] == 9 {
        kept[i] = 0;
      } else {
        kept[i] += 1;
        break;
      }
    }
  }

  let m = kept.len() as isize;
  let tail_exp = lead_exp - (m - 1);
  // Digit at a given place-value exponent (kept digits, else placeholder 0).
  let digit_at = |exp: isize| -> char {
    if (tail_exp..=lead_exp).contains(&exp) {
      (b'0' + kept[(lead_exp - exp) as usize]) as char
    } else {
      '0'
    }
  };

  let mut int_str = String::new();
  for exp in (0..=lead_exp.max(0)).rev() {
    int_str.push(digit_at(exp));
  }
  let mut frac_str = String::new();
  if tail_exp < 0 {
    for exp in (tail_exp..=-1).rev() {
      frac_str.push(digit_at(exp));
    }
  }
  format!("{}.{}", int_str, frac_str)
}

/// Drop trailing zeros from the fractional part of an `int.frac` decimal string
/// while always keeping the decimal point (`0.300000` → `0.3`, `2.` → `2.`,
/// `100.` → `100.`). Only fractional zeros are trimmed; integer-part zeros carry
/// magnitude and stay. Used for machine-precision real display, which shows no
/// trailing zeros (unlike a fixed arbitrary-precision real).
fn drop_trailing_frac_zeros(s: &str) -> String {
  match s.split_once('.') {
    Some((int_s, frac_s)) => {
      format!("{}.{}", int_s, frac_s.trim_end_matches('0'))
    }
    None => s.to_string(),
  }
}

/// Rewrite every arbitrary-precision real in a flat output string to its
/// notebook display form: the backtick precision marker is removed and the
/// mantissa truncated to its precision in significant figures
/// (`` {3.1415…`1., 3.1415…`3.} `` → `{3., 3.14}`). Machine-real text carries
/// no marker, so it passes through unchanged.
///
/// The CLI / `eval` result string keeps the backtick InputForm (it must match
/// `wolframscript -code`), so notebook front-ends (Woxi Studio) apply this at
/// the display layer only.
pub fn truncate_precision_reals(text: &str) -> String {
  let bytes = text.as_bytes();
  let n = bytes.len();
  let mut out = String::with_capacity(n);
  let mut i = 0;
  // Whether the previous char continues an identifier/number, so a digit that
  // follows it starts no fresh number token (e.g. the `2` in the symbol `x2`).
  let mut in_word = false;
  while i < n {
    let c = bytes[i];
    if c.is_ascii_digit() && !in_word {
      let start = i;
      let mut j = i;
      while j < n && (bytes[j].is_ascii_digit() || bytes[j] == b'.') {
        j += 1;
      }
      if j < n && bytes[j] == b'`' {
        // Consume the marker: an optional second backtick, the precision
        // digits/dots, then an optional `*^exp` scientific suffix.
        let mut k = j + 1;
        if k < n && bytes[k] == b'`' {
          k += 1;
        }
        while k < n && (bytes[k].is_ascii_digit() || bytes[k] == b'.') {
          k += 1;
        }
        if k + 1 < n && bytes[k] == b'*' && bytes[k + 1] == b'^' {
          k += 2;
          if k < n && bytes[k] == b'-' {
            k += 1;
          }
          while k < n && bytes[k].is_ascii_digit() {
            k += 1;
          }
        }
        if let Some(display) = precision_number_display(&text[start..k]) {
          out.push_str(&display);
          i = k;
          in_word = true;
          continue;
        }
      }
      // Not a precision token — copy the scanned digits verbatim.
      out.push_str(&text[start..j]);
      i = j;
      in_word = true;
      continue;
    }
    let ch = text[i..].chars().next().unwrap();
    let len = ch.len_utf8();
    out.push_str(&text[i..i + len]);
    // A following digit continues a symbol iff this char is alphanumeric, an
    // underscore, or a context backtick.
    in_word = ch.is_ascii_alphanumeric() || ch == '_' || ch == '`';
    i += len;
  }
  scientific_to_caret(&out)
}

/// Rewrite the InputForm scientific-notation operator `*^` (as in `1.5*^-8`) to
/// the readable `×10^` form for plain-text notebook display (Woxi Studio):
/// `1.*^10` → `1.×10^10`, `1.5*^-8` → `1.5×10^-8`. `*^` only ever occurs as a
/// real literal's exponent marker, always immediately after a digit or `.`, so
/// the guard on the preceding character leaves any other `*` / `^` untouched.
/// The Playground/Studio SVG path renders the exponent as a true superscript
/// via [`scientific_string_to_box`]; the plain-text widget cannot, so the base
/// stays inline with a caret.
fn scientific_to_caret(text: &str) -> String {
  let mut out = String::with_capacity(text.len() + 8);
  let mut prev: Option<char> = None;
  let mut chars = text.chars().peekable();
  while let Some(c) = chars.next() {
    if c == '*'
      && chars.peek() == Some(&'^')
      && prev.is_some_and(|p| p.is_ascii_digit() || p == '.')
    {
      chars.next(); // consume the '^'
      out.push_str("\u{00d7}10^");
      prev = Some('^');
    } else {
      out.push(c);
      prev = Some(c);
    }
  }
  out
}

/// Expand Wolfram character escape sequences to UTF-8 characters:
///   `\.HH`     → 2-digit hex code point (ASCII range)
///   `\:HHHH`   → 4-digit hex code point (BMP)
///   `\OOO`     → 3-digit octal code point
/// Inside string literals (`"..."`), the leading `\\` (literal backslash) is
/// preserved unchanged so the string parser still sees its own escapes.
/// Outside strings, escapes are expanded directly. Other escape sequences
/// like `\n`, `\[Name]`, `\"` are left untouched for the string parser /
/// named-character handler to deal with.
/// Normalize the modifier-letter circumflex `ˆ` (U+02C6) to the ASCII caret
/// `^` so it acts as the Power operator. Some keyboards (notably macOS, where
/// the `^` dead key emits a lone modifier circumflex) produce U+02C6 instead of
/// U+005E. Because U+02C6 is a Unicode letter (category Lm), the grammar would
/// otherwise swallow it into an identifier (e.g. `xˆ2` → symbol `xˆ`), so we
/// fix it at the source. Characters inside string literals are left untouched.
fn normalize_circumflex_operator(input: &str) -> String {
  if !input.contains('\u{02C6}') {
    return input.to_string();
  }
  let mut result = String::with_capacity(input.len());
  let mut in_string = false;
  let mut chars = input.chars().peekable();
  while let Some(ch) = chars.next() {
    if ch == '"' {
      in_string = !in_string;
      result.push(ch);
      continue;
    }
    if in_string {
      result.push(ch);
      // Preserve escapes (e.g. `\"`) verbatim so a quote inside the escape
      // doesn't flip the string state.
      if ch == '\\'
        && let Some(next) = chars.next()
      {
        result.push(next);
      }
      continue;
    }
    result.push(if ch == '\u{02C6}' { '^' } else { ch });
  }
  result
}

fn expand_char_escapes(input: &str) -> String {
  // Fast path: no backslash means nothing to do.
  if !input.contains('\\') {
    return input.to_string();
  }
  let chars: Vec<char> = input.chars().collect();
  let len = chars.len();
  let mut result = String::with_capacity(input.len());
  let mut i = 0;
  let mut in_string = false;
  while i < len {
    let ch = chars[i];
    if ch == '"' {
      in_string = !in_string;
      result.push(ch);
      i += 1;
      continue;
    }
    if ch != '\\' {
      result.push(ch);
      i += 1;
      continue;
    }
    // `ch == '\\'` here. Decide what kind of escape follows.
    let next = chars.get(i + 1).copied();
    // Inside strings, `\\` is a literal-backslash escape — preserve it so the
    // string parser still produces a single backslash.
    if in_string && next == Some('\\') {
      result.push('\\');
      result.push('\\');
      i += 2;
      continue;
    }
    match next {
      Some('.')
        // `\.HH` — 2 hex digits
        if i + 3 < len
          && chars[i + 2].is_ascii_hexdigit()
          && chars[i + 3].is_ascii_hexdigit()
        => {
          let hex: String = [chars[i + 2], chars[i + 3]].iter().collect();
          if let Ok(code) = u32::from_str_radix(&hex, 16)
            && let Some(c) = char::from_u32(code)
          {
            result.push(c);
            i += 4;
            continue;
          }
        }
      Some(':')
        // `\:HHHH` — 4 hex digits
        if i + 5 < len
          && chars[i + 2].is_ascii_hexdigit()
          && chars[i + 3].is_ascii_hexdigit()
          && chars[i + 4].is_ascii_hexdigit()
          && chars[i + 5].is_ascii_hexdigit()
        => {
          let hex: String =
            [chars[i + 2], chars[i + 3], chars[i + 4], chars[i + 5]]
              .iter()
              .collect();
          if let Ok(code) = u32::from_str_radix(&hex, 16)
            && let Some(c) = char::from_u32(code)
          {
            result.push(c);
            i += 6;
            continue;
          }
        }
      Some(d) if ('0'..='7').contains(&d)
        // `\OOO` — 3 octal digits (must all be 0-7).
        && i + 3 < len
          && ('0'..='7').contains(&chars[i + 2])
          && ('0'..='7').contains(&chars[i + 3])
        => {
          let oct: String =
            [chars[i + 1], chars[i + 2], chars[i + 3]].iter().collect();
          if let Ok(code) = u32::from_str_radix(&oct, 8)
            && let Some(c) = char::from_u32(code)
          {
            result.push(c);
            i += 4;
            continue;
          }
        }
      _ => {}
    }
    // Not a recognized escape — emit the backslash and continue.
    result.push(ch);
    i += 1;
  }
  result
}

/// Insert semicolons at top-level newline boundaries so the parser treats
/// each logical line as a separate statement.  Newlines inside brackets,
/// parentheses or braces are left alone (they're part of a multiline expression).
/// Lines that already end with `;` or `:=` are also left alone.
pub fn insert_statement_separators(input: &str) -> String {
  // Fast path: no newlines means nothing to do
  if !input.contains('\n') {
    return input.to_string();
  }

  let mut result = String::with_capacity(input.len() + 32);
  let mut depth: i32 = 0; // nesting depth of [], (), {}
  let mut in_string = false;
  let mut comment_depth: i32 = 0;
  let mut line_has_code = false; // whether the current line has non-whitespace, non-comment content
  let mut last_code_char: Option<char> = None; // last meaningful (non-comment) character
  let mut prev_code_char: Option<char> = None; // second-to-last meaningful character
  // Deferred semicolon: instead of inserting `;` immediately at a newline,
  // we record the position where it should go. We only actually insert it
  // when we later encounter actual code on a subsequent line. This avoids
  // adding a spurious trailing `;` when only comments/whitespace follow.
  let mut pending_semi_pos: Option<usize> = None;
  let chars: Vec<char> = input.chars().collect();
  let len = chars.len();
  let mut i = 0;

  while i < len {
    let ch = chars[i];

    // Track comment state: (* ... *) with nesting support
    if !in_string && i + 1 < len && ch == '(' && chars[i + 1] == '*' {
      comment_depth += 1;
      result.push(ch);
      i += 1;
      continue;
    }
    if comment_depth > 0 && i + 1 < len && ch == '*' && chars[i + 1] == ')' {
      comment_depth -= 1;
      result.push(ch);
      result.push(chars[i + 1]);
      i += 2;
      continue;
    }
    if comment_depth > 0 {
      result.push(ch);
      i += 1;
      continue;
    }

    // Track string state
    if ch == '"' {
      in_string = !in_string;
      // A string is actual code — flush any pending semicolon
      if let Some(pos) = pending_semi_pos.take() {
        result.insert(pos, ';');
      }
      result.push(ch);
      line_has_code = true;
      prev_code_char = last_code_char;
      last_code_char = Some(ch);
      i += 1;
      continue;
    }
    if in_string {
      result.push(ch);
      prev_code_char = last_code_char;
      last_code_char = Some(ch);
      i += 1;
      continue;
    }

    // Handle line continuation: backslash followed by newline
    if ch == '\\' && i + 1 < len && chars[i + 1] == '\n' {
      // Skip both the backslash and the newline — the next line continues this one
      i += 2;
      continue;
    }

    // Track nesting depth (including <| |> for associations)
    if ch == '<' && i + 1 < len && chars[i + 1] == '|' {
      depth += 1;
      // Push both characters and advance past them
      if let Some(pos) = pending_semi_pos.take() {
        result.insert(pos, ';');
      }
      line_has_code = true;
      prev_code_char = Some('<');
      last_code_char = Some('|');
      result.push('<');
      result.push('|');
      i += 2;
      continue;
    } else if ch == '|' && i + 1 < len && chars[i + 1] == '>' {
      depth -= 1;
      if let Some(pos) = pending_semi_pos.take() {
        result.insert(pos, ';');
      }
      line_has_code = true;
      prev_code_char = Some('|');
      last_code_char = Some('>');
      result.push('|');
      result.push('>');
      i += 2;
      continue;
    }
    match ch {
      '[' | '(' | '{' => depth += 1,
      ']' | ')' | '}' => depth -= 1,
      _ => {}
    }

    if ch == '\n' && depth == 0 {
      // Only add `;` if the current line had actual code (not just comments/whitespace)
      // and doesn't already end with `;` or `:=` or `/:` (TagSet continuation)
      // or an operator character (indicating the expression continues on the next line)
      let ends_with_set_delayed =
        last_code_char == Some('=') && prev_code_char == Some(':');
      let ends_with_tag_set =
        last_code_char == Some(':') && prev_code_char == Some('/');
      // Line ending with an operator means the expression continues on the next line.
      // `>` alone is the Greater operator; `<>` is StringJoin; `->`, `:>`, `>>`, `>>>`
      // all end in `>`. Treating any trailing `>` as a continuation covers all forms.
      let ends_with_operator = matches!(
        last_code_char,
        Some('+' | '-' | '*' | '/' | '^' | '@' | '~' | ',' | '=' | '<' | '>' | '|')
      ) || (last_code_char == Some('&')
        && prev_code_char == Some('&')) // &&
        || (last_code_char == Some('.')
        && prev_code_char == Some('/')); // /. or //.
      // `!` at end of line is ambiguous: postfix Factorial (`5!`) or prefix
      // Not (`:= ! palindromeQ@...`). Treat it as a continuation when the
      // preceding code char is one that can't be a value (operators or
      // openers), since only then is the `!` necessarily prefix Not.
      let ends_with_prefix_not = last_code_char == Some('!')
        && matches!(
          prev_code_char,
          Some(
            '='
              | '&'
              | '|'
              | ','
              | ';'
              | '{'
              | '('
              | '['
              | '+'
              | '-'
              | '*'
              | '/'
              | '^'
              | '@'
              | '~'
              | '<'
              | '>'
          )
        );
      let needs_semi = line_has_code
        && last_code_char != Some(';')
        && !ends_with_set_delayed
        && !ends_with_tag_set
        && !ends_with_operator
        && !ends_with_prefix_not;

      if needs_semi {
        // Defer the semicolon — record position before the newline
        pending_semi_pos = Some(result.len());
      }
      result.push('\n');

      // Reset line tracking
      line_has_code = false;
      last_code_char = None;
      prev_code_char = None;
    } else if ch == '\n' {
      // Newline inside nesting — just pass through
      result.push(ch);
    } else {
      if !ch.is_whitespace() {
        // Actual code encountered — flush any pending semicolon
        if let Some(pos) = pending_semi_pos.take() {
          result.insert(pos, ';');
        }
        line_has_code = true;
        prev_code_char = last_code_char;
        last_code_char = Some(ch);
      }
      result.push(ch);
    }

    i += 1;
  }

  result
}

/// Split input into top-level statements at newline boundaries.
/// Respects bracket nesting (newlines inside `[]`, `()`, `{}` are kept),
/// strings, comments, and `:=` continuations.
pub fn split_into_statements(input: &str) -> Vec<String> {
  // Normalize CRLF to LF so line continuation and newline handling work
  // consistently regardless of line ending style.
  let input = if input.contains('\r') {
    std::borrow::Cow::Owned(input.replace("\r\n", "\n").replace('\r', "\n"))
  } else {
    std::borrow::Cow::Borrowed(input)
  };
  let trimmed = input.trim();
  if trimmed.is_empty() {
    return vec![String::new()];
  }
  if !trimmed.contains('\n') {
    return vec![trimmed.to_string()];
  }

  let mut statements = Vec::new();
  let mut current = String::with_capacity(trimmed.len());
  let mut depth: i32 = 0;
  let mut in_string = false;
  let mut comment_depth: i32 = 0;
  let mut line_has_code = false;
  let mut current_has_code = false; // tracks whether the current buffer has actual code (not just comments/whitespace)
  let mut last_code_char: Option<char> = None;
  let mut prev_code_char: Option<char> = None;
  let chars: Vec<char> = trimmed.chars().collect();
  let len = chars.len();
  let mut i = 0;

  while i < len {
    let ch = chars[i];

    // Handle line continuation: backslash followed by newline
    if !in_string
      && comment_depth == 0
      && ch == '\\'
      && i + 1 < len
      && chars[i + 1] == '\n'
    {
      // Skip both the backslash and the newline — the next line continues this one
      i += 2;
      continue;
    }

    // Track comment state: (* ... *) with nesting support
    if !in_string && i + 1 < len && ch == '(' && chars[i + 1] == '*' {
      comment_depth += 1;
      current.push(ch);
      i += 1;
      continue;
    }
    if comment_depth > 0 && i + 1 < len && ch == '*' && chars[i + 1] == ')' {
      comment_depth -= 1;
      current.push(ch);
      current.push(chars[i + 1]);
      i += 2;
      continue;
    }
    if comment_depth > 0 {
      current.push(ch);
      i += 1;
      continue;
    }

    // Track string state
    if ch == '"' {
      in_string = !in_string;
      current.push(ch);
      line_has_code = true;
      prev_code_char = last_code_char;
      last_code_char = Some(ch);
      i += 1;
      continue;
    }
    if in_string {
      current.push(ch);
      prev_code_char = last_code_char;
      last_code_char = Some(ch);
      i += 1;
      continue;
    }

    // Track nesting depth (including <| |> for associations)
    if ch == '<' && i + 1 < len && chars[i + 1] == '|' {
      depth += 1;
      line_has_code = true;
      current_has_code = true;
      prev_code_char = Some('<');
      last_code_char = Some('|');
      current.push('<');
      current.push('|');
      i += 2;
      continue;
    } else if ch == '|' && i + 1 < len && chars[i + 1] == '>' {
      depth -= 1;
      line_has_code = true;
      current_has_code = true;
      prev_code_char = Some('|');
      last_code_char = Some('>');
      current.push('|');
      current.push('>');
      i += 2;
      continue;
    }
    match ch {
      '[' | '(' | '{' => depth += 1,
      ']' | ')' | '}' => depth -= 1,
      _ => {}
    }

    if ch == '\n' && depth == 0 {
      let ends_with_set_delayed =
        last_code_char == Some('=') && prev_code_char == Some(':');
      let ends_with_tag_set =
        last_code_char == Some(':') && prev_code_char == Some('/');
      // /; (Condition) at end of line means the expression continues
      let ends_with_condition =
        last_code_char == Some(';') && prev_code_char == Some('/');
      // Line ending with an operator means the expression continues
      let ends_with_operator = matches!(
        last_code_char,
        Some('+' | '-' | '*' | '/' | '^' | '@' | '~' | ',' | '=' | '<' | '|')
      ) || (last_code_char == Some('>')
        && matches!(prev_code_char, Some('-' | ':' | '>')))
        || (last_code_char == Some('&') && prev_code_char == Some('&'));
      // Trailing `!` is prefix Not (and thus a continuation) when the
      // previous code char is an operator/opener rather than a value.
      let ends_with_prefix_not = last_code_char == Some('!')
        && matches!(
          prev_code_char,
          Some(
            '='
              | '&'
              | '|'
              | ','
              | ';'
              | '{'
              | '('
              | '['
              | '+'
              | '-'
              | '*'
              | '/'
              | '^'
              | '@'
              | '~'
              | '<'
              | '>'
          )
        );

      let should_split = line_has_code
        && !ends_with_set_delayed
        && !ends_with_tag_set
        && !ends_with_condition
        && !ends_with_operator
        && !ends_with_prefix_not;

      if should_split {
        let stmt = current.trim().to_string();
        if !stmt.is_empty() {
          statements.push(stmt);
        }
        current.clear();
        current_has_code = false;
      } else {
        current.push(ch);
      }

      line_has_code = false;
      last_code_char = None;
      prev_code_char = None;
    } else if ch == '\n' {
      // Newline inside nesting — just pass through
      current.push(ch);
    } else {
      if !ch.is_whitespace() {
        line_has_code = true;
        current_has_code = true;
        prev_code_char = last_code_char;
        last_code_char = Some(ch);
      }
      current.push(ch);
    }

    i += 1;
  }

  let stmt = current.trim().to_string();
  if !stmt.is_empty() && current_has_code {
    statements.push(stmt);
  }

  if statements.is_empty() {
    statements.push(String::new());
  }

  statements
}

/// Try to evaluate a simple function call without full parsing.
/// Returns Some(result) if successfully handled, None if needs full parsing.
fn try_fast_function_call(
  input: &str,
) -> Option<Result<String, InterpreterError>> {
  // Pattern: FunctionName[arg1, arg2, ...]
  // Must start with a letter and have exactly one balanced [...] pair at the end

  let open_bracket = input.find('[')?;
  if !input.ends_with(']') {
    return None;
  }

  let func_name = &input[..open_bracket];
  // Validate function name is a simple identifier
  if func_name.is_empty()
    || !func_name
      .chars()
      .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '$')
    || !func_name.chars().next().unwrap().is_ascii_alphabetic()
  {
    return None;
  }

  let args_str = &input[open_bracket + 1..input.len() - 1];

  // Check for nested brackets (indicates complex expression)
  let mut depth = 0;
  for c in args_str.chars() {
    match c {
      '[' => depth += 1,
      ']' => depth -= 1,
      _ => {}
    }
    if depth < 0 {
      return None; // Unbalanced
    }
  }
  // After processing, we should be at depth 0 for well-formed args

  // Split arguments by comma (respecting nested structures)
  let args = split_args(args_str);

  // Handle specific functions that are commonly called
  match func_name {
    "MemberQ" => {
      if args.len() != 2 {
        return None;
      }
      let elem_expr = args[1].trim();

      // If the second argument looks like a pattern, skip the fast path
      // and let the full evaluator handle it with proper pattern matching
      if elem_expr.contains('_')
        || elem_expr.starts_with("Repeated")
        || elem_expr.starts_with("Alternatives")
        || elem_expr.contains("Pattern[")
        || elem_expr.contains("Blank[")
      {
        return None;
      }

      // First arg should be a list, second is the element to find
      let list_str = args[0].trim();

      // Evaluate the element expression
      let target = match interpret(elem_expr) {
        Ok(v) => v,
        Err(_) => return None,
      };

      // Parse the list
      if !list_str.starts_with('{') || !list_str.ends_with('}') {
        return None; // Not a literal list, need full parsing
      }

      let inner = &list_str[1..list_str.len() - 1];
      let list_elems = split_args(inner);

      // Check if target is in the list
      for elem in list_elems {
        let elem = elem.trim();
        // Try to evaluate the list element if needed
        let elem_val = if elem.contains('[') {
          match interpret(elem) {
            Ok(v) => v,
            Err(_) => elem.to_string(),
          }
        } else {
          elem.to_string()
        };
        if elem_val == target {
          return Some(Ok("True".to_string()));
        }
      }
      Some(Ok("False".to_string()))
    }
    // First/Rest intentionally have no fast path: the AST
    // implementations carry the conformant ::nofirst/::norest/::normal
    // message behavior.
    _ => None,
  }
}

/// Split a comma-separated argument list, respecting nested structures
fn split_args(s: &str) -> Vec<String> {
  let mut args = Vec::new();
  let mut current = String::new();
  let mut depth = 0;

  for c in s.chars() {
    match c {
      '{' | '[' | '(' => {
        depth += 1;
        current.push(c);
      }
      '}' | ']' | ')' => {
        depth -= 1;
        current.push(c);
      }
      ',' if depth == 0 => {
        args.push(current.trim().to_string());
        current.clear();
      }
      _ => {
        current.push(c);
      }
    }
  }
  if !current.is_empty() {
    args.push(current.trim().to_string());
  }
  args
}

/// Parse and evaluate a single expression, returning the raw `Expr` so
/// callers can inspect the evaluated AST (e.g. to detect a held
/// `Manipulate[…]` call). Only the first top-level expression in
/// `input` is evaluated.
///
/// Unlike `interpret` this does not format the result as a string, so
/// held function calls like `Manipulate[…]` are preserved as
/// `Expr::FunctionCall` values.
pub fn interpret_to_expr(
  input: &str,
) -> Result<syntax::Expr, InterpreterError> {
  let normalized = if input.contains('\r') {
    input.replace("\r\n", "\n").replace('\r', "\n")
  } else {
    input.to_string()
  };
  // Treat the modifier-letter circumflex `ˆ` (U+02C6) as the Power operator.
  let normalized = normalize_circumflex_operator(&normalized);

  let mut pairs = parse(&normalized).map_err(|e| {
    InterpreterError::EvaluationError(format!("Parse error: {}", e))
  })?;
  let program = pairs.next().ok_or(InterpreterError::EmptyInput)?;
  if program.as_rule() != Rule::Program {
    return Err(InterpreterError::EvaluationError(format!(
      "Expected Program, got {:?}",
      program.as_rule()
    )));
  }

  for node in program.into_inner() {
    if matches!(node.as_rule(), Rule::Expression | Rule::TopLevelSpan) {
      let expr = syntax::pair_to_expr(node);
      return evaluator::evaluate_expr_to_expr(&expr);
    }
  }
  Err(InterpreterError::EmptyInput)
}

/// New interpret function that returns both stdout and the result
pub fn interpret_with_stdout(
  input: &str,
) -> Result<InterpretResult, InterpreterError> {
  // Clear the capture buffers
  clear_captured_stdout();
  clear_captured_graphics();
  clear_captured_graphicsbox();
  clear_captured_warnings();
  clear_captured_output_svg();
  clear_captured_sound();

  // Enable visual mode for display wrapper rendering (e.g. TableForm → Grid SVG)
  VISUAL_MODE.with(|v| *v.borrow_mut() = true);

  // Capture the top-level Expr so `%` resolves to this cell's result in
  // subsequent evaluations. The cache is set at the moment evaluation
  // completes (inside the dispatch loop) via `set_last_output`, so we
  // do nothing here besides keep the value put there.
  // Perform the standard interpretation
  let result = interpret(input);

  // Reset visual mode
  VISUAL_MODE.with(|v| *v.borrow_mut() = false);

  let result = result?;

  // Get the captured output
  let stdout = get_captured_stdout();
  let graphics = get_captured_graphics();
  let output_svg = get_captured_output_svg();
  let sound = get_captured_sound();
  let warnings = get_captured_warnings();

  // When `Information[…]` (or the `?sym` / `?Plot*` shortcuts) captured a
  // styled SVG card, suppress the textual `InformationData[…]` echo so the
  // visual host (playground, woxi-studio) shows only the graphical card.
  // The CLI path is unaffected because it calls `interpret()` directly.
  let result = if graphics.is_some()
    && (result.starts_with("InformationData[")
      || result.starts_with("InformationDataGrid["))
    && (result.ends_with("|>]")
      || result.ends_with("False]")
      || result.ends_with("True]"))
  {
    String::new()
  } else {
    result
  };

  // Return stdout, result, and any graphical output
  Ok(InterpretResult {
    stdout,
    result,
    graphics,
    output_svg,
    sound,
    warnings,
  })
}

fn store_function_definition(
  pair: Pair<Rule>,
) -> Result<Option<String>, InterpreterError> {
  // FunctionDefinition  :=  Identifier "[" (Pattern ("," Pattern)*)? "]" ":=" Expression
  let raw_lhs = {
    // pest pairs are not Clone, so capture the source slice up to ":=".
    let span = pair.as_span().as_str();
    span
      .split_once(":=")
      .map(|(l, _)| l.trim().to_string())
      .unwrap_or_else(|| span.trim().to_string())
  };
  let mut inner = pair.into_inner();
  let func_name = inner.next().unwrap().as_str().to_owned(); // Identifier

  // Reject assignments to built-in Protected heads (e.g. `Sin`,
  // `Plus`, …). wolframscript emits `SetDelayed::write: Tag <h>
  // in <lhs> is Protected.` and returns `$Failed`. Special case:
  // `N[sym, …] := body` is stored as an NValue on `sym` instead
  // of as a DownValue on `N`, so wolframscript allows it even
  // though `N` is Protected. Treat that pattern as not-rejected.
  // Heads where wolframscript allows DownValue-shaped assignment
  // (the rule is redirected to a per-symbol mechanism — NValues
  // for `N`, Messages for `MessageName`, Format/Default/Options
  // for the formatting heads, etc.).
  let allows_redirected_rule = matches!(
    func_name.as_str(),
    "N" | "MessageName" | "Format" | "Default" | "Options"
  );
  let is_n_value_assignment = allows_redirected_rule;
  let was_unprotected = FUNC_ATTRS_REMOVED.with(|m| {
    m.borrow()
      .get(func_name.as_str())
      .is_some_and(|attrs| attrs.contains(&"Protected".to_string()))
  });
  let is_builtin_protected = !is_n_value_assignment
    && !was_unprotected
    && evaluator::get_builtin_attributes(&func_name).contains(&"Protected");
  let is_user_protected = FUNC_ATTRS.with(|m| {
    m.borrow()
      .get(func_name.as_str())
      .is_some_and(|attrs| attrs.contains(&"Protected".to_string()))
  });
  if is_builtin_protected || is_user_protected {
    let (tag, ret) = if is_builtin_protected && !is_user_protected {
      ("SetDelayed::write", Some("$Failed".to_string()))
    } else {
      ("SetDelayed::wrsym", None)
    };
    emit_message(&format!(
      "{}: Tag {} in {} is Protected.",
      tag, func_name, raw_lhs
    ));
    return Ok(ret);
  }

  // Collect all pattern parameters with their optional conditions, defaults, and head constraints
  let mut params = Vec::new();
  let mut conditions: Vec<Option<syntax::Expr>> = Vec::new();
  let mut defaults: Vec<Option<syntax::Expr>> = Vec::new();
  let mut heads: Vec<Option<String>> = Vec::new();
  let mut blank_types: Vec<u8> = Vec::new();
  let mut body_pair = None;
  let mut has_any_condition = false;

  for item in inner {
    match item.as_rule() {
      Rule::PatternCondition => {
        // PatternCondition = { PatternName ~ "_" ~ "/;" ~ ConditionExpr }
        let mut pat_inner = item.into_inner();
        let param_name = pat_inner.next().unwrap().as_str().to_owned();
        // The second child is the ConditionExpr
        let cond_pair = pat_inner.next().unwrap();
        let cond_expr = syntax::pair_to_expr(cond_pair);
        params.push(param_name);
        conditions.push(Some(cond_expr));
        defaults.push(None);
        heads.push(None);
        blank_types.push(1);
        has_any_condition = true;
      }
      Rule::PatternOptionalSimple => {
        // PatternOptionalSimple = { PatternName ~ "_" ~ ":" ~ Term }
        let mut pat_inner = item.into_inner();
        let param_name = pat_inner.next().unwrap().as_str().to_owned();
        let default_pair = pat_inner.next().unwrap();
        let default_expr = syntax::pair_to_expr(default_pair);
        params.push(param_name);
        conditions.push(None);
        defaults.push(Some(default_expr));
        heads.push(None);
        blank_types.push(1);
      }
      Rule::PatternOptionalWithHead => {
        // PatternOptionalWithHead = { PatternName ~ "_" ~ Identifier ~ ":" ~ Term }
        let mut pat_inner = item.into_inner();
        let param_name = pat_inner.next().unwrap().as_str().to_owned();
        let head_name = pat_inner.next().unwrap().as_str().to_owned();
        let default_pair = pat_inner.next().unwrap();
        let default_expr = syntax::pair_to_expr(default_pair);
        params.push(param_name);
        conditions.push(None);
        defaults.push(Some(default_expr));
        heads.push(Some(head_name));
        blank_types.push(1);
      }
      Rule::PatternOptionalDefaultSimple => {
        // PatternOptionalDefaultSimple = { PatternName? ~ "_" ~ "." }
        let mut pat_inner = item.into_inner();
        let param_name = pat_inner
          .next()
          .map(|p| p.as_str().to_owned())
          .unwrap_or_default();
        let position = params.len() + 1;
        params.push(param_name);
        conditions.push(None);
        // `x_.` resolves its default from `Default[func, position]` at
        // dispatch time. Store that as the placeholder so the runtime
        // lookup chain (which falls through to `Default[func]`) supplies
        // the user-set value (matching Wolfram's `f[x_.] := ...` +
        // `Default[f] = c` semantics).
        defaults.push(Some(syntax::Expr::FunctionCall {
          name: "Default".to_string(),
          args: vec![
            syntax::Expr::Identifier(func_name.clone()),
            syntax::Expr::Integer(position as i128),
          ]
          .into(),
        }));
        heads.push(None);
        blank_types.push(1);
      }
      Rule::PatternWithHead => {
        // Extract parameter name and head (e.g., "x_List" -> name="x", head="List")
        let full = item.as_str();
        let mut pat_inner = item.into_inner();
        let pat_name_str = pat_inner.next().unwrap().as_str();
        let blank_count = full[pat_name_str.len()..]
          .chars()
          .take_while(|&c| c == '_')
          .count();
        let param_name = pat_name_str.to_owned();
        let head_name = pat_inner.next().unwrap().as_str().to_owned();
        params.push(param_name);
        conditions.push(None);
        defaults.push(None);
        heads.push(Some(head_name));
        blank_types.push(blank_count.min(3) as u8);
      }
      Rule::PatternTest => {
        // PatternTest: x_?test or _?test or x_Head?test or _Head?test etc.
        let full_str = item.as_str();
        let mut pat_inner = item.into_inner();
        let first = pat_inner.next().unwrap();
        let (param_name, remaining) = if first.as_rule() == Rule::PatternName {
          (first.as_str().to_owned(), pat_inner.next().unwrap())
        } else {
          // Anonymous blank _?test — generate a placeholder param name
          (format!("__pt{}", params.len()), first)
        };
        // Check for optional head (PatternTestHead)
        let (head, test_pair) = if remaining.as_rule() == Rule::PatternTestHead
        {
          (
            Some(remaining.as_str().to_owned()),
            pat_inner.next().unwrap(),
          )
        } else {
          (None, remaining)
        };
        // Extract blank_type from underscores between name and ?
        // Skip auto-generated names when computing offset
        let name_offset = if param_name.starts_with("__pt") {
          0
        } else {
          param_name.len()
        };
        let bt = full_str[name_offset..]
          .chars()
          .take_while(|&c| c == '_')
          .count()
          .min(3) as u8;
        let test_expr = syntax::pair_to_expr(test_pair);
        // Build condition: testFunc[paramName]
        let cond_expr = syntax::Expr::FunctionCall {
          name: syntax::expr_to_string(&test_expr),
          args: vec![syntax::Expr::Identifier(param_name.clone())].into(),
        };
        params.push(param_name);
        conditions.push(Some(cond_expr));
        defaults.push(None);
        heads.push(head);
        blank_types.push(bt);
        has_any_condition = true;
      }
      Rule::PatternSimple => {
        // Extract parameter name and blank type from pattern
        // "x_" -> (name="x", blank_type=1), "u__" -> (name="u", blank_type=2), "v___" -> (name="v", blank_type=3)
        let s = item.as_str();
        let name = s.trim_end_matches('_');
        let blank_count = s.len() - name.len();
        params.push(name.to_owned());
        conditions.push(None);
        defaults.push(None);
        heads.push(None);
        blank_types.push(blank_count.min(3) as u8);
      }
      Rule::Expression
      | Rule::ExpressionNoImplicit
      | Rule::CompoundExpression => {
        body_pair = Some(item);
      }
      _ => {}
    }
  }

  // Convert body to AST instead of storing as string
  let raw_body_expr = syntax::pair_to_expr(body_pair.ok_or_else(|| {
    InterpreterError::EvaluationError("Missing function body".into())
  })?);

  // f[x_] := body /; test parses as `body = Condition[actual_body, test]`.
  // Keep the Condition wrapper on the stored body so DownValues can show
  // the original `body /; test` form. Dispatch treats `Condition[expr,
  // False]` as "skip this rule", which preserves the runtime guard
  // behavior even though we no longer mirror the test in the parameter
  // condition slots.
  let body_expr = if let syntax::Expr::FunctionCall { ref name, args: _ } =
    raw_body_expr
    && name == "Condition"
  {
    has_any_condition = true;
    raw_body_expr
  } else {
    raw_body_expr
  };

  FUNC_DEFS.with(|m| {
    let mut defs = m.borrow_mut();
    let entry = defs.entry(func_name).or_insert_with(Vec::new);
    let arity = params.len();
    if has_any_condition {
      // Conditional definition: only remove existing definitions with same arity
      // that have the exact same conditions (re-definition of same pattern)
      // For simplicity, just append - Wolfram keeps all conditional defs
    } else {
      // Unconditional definition: remove only other unconditional defs with same arity,
      // same blank_types, AND same head constraints (keep conditional definitions and defs
      // with different blank patterns or different heads,
      // e.g. f[u_] and f[u__] are distinct overloads,
      // and f[x_Integer] and f[x_String] are distinct overloads).
      //
      // A `/;` guard (`f[x_] := body /; test`) is stored as a `Condition`
      // wrapper on the BODY, not in the parameter `conds` slots, so it must
      // be detected via the body too — otherwise a later same-pattern
      // unconditional rule would wrongly delete the guarded rule. In Wolfram,
      // `f[a_,b_] := f[b,a] /; a>b` and `f[a_,b_] := …` are distinct
      // DownValues and both are retained.
      entry.retain(|(p, conds, d, h, bt, body)| {
        p.len() != arity
          || bt != &blank_types
          || h != &heads
          // Optional (defaulted) positions distinguish overloads: `f[x_, y_]`
          // and `f[x_, y_:0]` are separate DownValues, so keep an existing rule
          // whose optional-arg pattern differs from the new one's.
          || d.iter()
            .map(|x| x.is_some())
            .ne(defaults.iter().map(|x| x.is_some()))
          || conds.iter().any(|c| c.is_some())
          || matches!(
            body,
            syntax::Expr::FunctionCall { name, .. } if name == "Condition"
          )
      });
    }
    // Add the new definition with parsed AST, conditions, defaults, and head constraints.
    // Insert by the rule partial order: place the new rule before the first
    // existing rule it strictly dominates (is more specific than). Rules it does
    // not dominate — including incomparable ones, such as a guarded but
    // structurally looser rule vs an unguarded tighter one — keep definition
    // order, matching Wolfram.
    let pos = entry
      .iter()
      .position(|(p, c, d, h, bt, b)| {
        crate::evaluator::assignment::rule_dominates(
          &params,
          &heads,
          &blank_types,
          &conditions,
          &defaults,
          &body_expr,
          p,
          h,
          bt,
          c,
          d,
          b,
        )
      })
      .unwrap_or(entry.len());
    entry.insert(
      pos,
      (params, conditions, defaults, heads, blank_types, body_expr),
    );
  });
  Ok(None)
}

/// Handle TagSetDelayed: tag /: f[args...] := body  (evaluate_rhs=false)
/// Handle TagSet:        tag /: f[args...] = body   (evaluate_rhs=true)
/// Stores an upvalue definition for `tag` that fires when `f` is called
/// with arguments containing `tag` as a head.
fn store_tag_set_delayed(
  pair: Pair<Rule>,
  evaluate_rhs: bool,
) -> Result<Option<String>, InterpreterError> {
  let mut inner = pair.into_inner();

  // First child: tag identifier (e.g., "g" in "g /: f[g[x_]] := ...")
  let tag_name = inner.next().unwrap().as_str().to_owned();

  // Next children come from BaseFunctionCall: Identifier BracketArgs+
  // We need to extract the outer function name and its arguments
  let func_call_pair = inner.next().unwrap(); // BaseFunctionCall
  let func_call_expr = syntax::pair_to_expr(func_call_pair);

  // Remaining child: the body expression
  let body_pair = inner.next().unwrap();
  let body_expr = syntax::pair_to_expr(body_pair);
  let body_expr = if evaluate_rhs {
    evaluator::evaluate_expr_to_expr(&body_expr)?
  } else {
    body_expr
  };

  // Extract outer function name and args from the LHS
  let (outer_func, lhs_args): (String, Vec<syntax::Expr>) =
    match &func_call_expr {
      syntax::Expr::FunctionCall { name, args } => {
        (name.clone(), args.to_vec())
      }
      syntax::Expr::CurriedCall { func, args } => {
        // Chained calls like f[g[x_]][y_] - use the full expression
        // For now, handle simple case only
        if let syntax::Expr::FunctionCall { name, .. } = func.as_ref() {
          (name.clone(), args.clone())
        } else {
          return Err(InterpreterError::EvaluationError(
            "TagSetDelayed: LHS must be a function call".into(),
          ));
        }
      }
      _ => {
        return Err(InterpreterError::EvaluationError(
          "TagSetDelayed: LHS must be a function call".into(),
        ));
      }
    };

  // Process each argument in the LHS to extract patterns
  // Arguments that are function calls with head == tag get destructured
  let mut params = Vec::new();
  let mut conditions: Vec<Option<syntax::Expr>> = Vec::new();
  let mut defaults: Vec<Option<syntax::Expr>> = Vec::new();
  let mut heads: Vec<Option<String>> = Vec::new();
  let mut final_body = body_expr.clone();

  for (i, arg) in lhs_args.iter().enumerate() {
    match arg {
      // Function call argument like g[x_] — extract inner patterns
      syntax::Expr::FunctionCall {
        name: arg_func_name,
        args: inner_args,
      } => {
        let param_name = format!("_up{}", i);
        heads.push(Some(arg_func_name.clone()));

        // Add length condition to ensure correct number of inner args
        if !inner_args.is_empty() {
          conditions.push(Some(syntax::Expr::Comparison {
            operands: vec![
              syntax::Expr::FunctionCall {
                name: "Length".to_string(),
                args: vec![syntax::Expr::Identifier(param_name.clone())].into(),
              },
              syntax::Expr::Integer(inner_args.len() as i128),
            ],
            operators: vec![syntax::ComparisonOp::SameQ],
          }));
        } else {
          conditions.push(None);
        }

        // Substitute inner pattern names in the body with Part[param, index]
        for (j, inner_arg) in inner_args.iter().enumerate() {
          let (pat_name, _pat_head) = extract_pattern_info_from_expr(inner_arg);
          if !pat_name.is_empty() {
            let part_expr = syntax::Expr::FunctionCall {
              name: "Part".to_string(),
              args: vec![
                syntax::Expr::Identifier(param_name.clone()),
                syntax::Expr::Integer((j + 1) as i128),
              ]
              .into(),
            };
            final_body =
              syntax::substitute_variable(&final_body, &pat_name, &part_expr);
          }
        }

        params.push(param_name);
        defaults.push(None);
      }
      // Simple pattern like x_ or x_Head
      _ => {
        let (pat_name, head) = extract_pattern_info_from_expr(arg);
        if pat_name.is_empty() && head.is_none() {
          // Literal value — create SameQ condition
          let param_name = format!("_up{}", i);
          let eval_arg = evaluator::evaluate_expr_to_expr(arg)?;
          conditions.push(Some(syntax::Expr::Comparison {
            operands: vec![
              syntax::Expr::Identifier(param_name.clone()),
              eval_arg,
            ],
            operators: vec![syntax::ComparisonOp::SameQ],
          }));
          params.push(param_name);
        } else {
          params.push(pat_name);
          conditions.push(None);
        }
        defaults.push(None);
        heads.push(head);
      }
    }
  }

  // Store in UPVALUES for introspection and cleanup.
  // If an upvalue with the same original LHS already exists, replace it.
  let lhs_str = syntax::expr_to_string(&func_call_expr);
  UPVALUES.with(|m| {
    let mut defs = m.borrow_mut();
    let entry = defs.entry(tag_name).or_insert_with(Vec::new);
    if let Some(pos) =
      entry.iter().position(|(_, _, _, _, _, _, orig_lhs, _)| {
        syntax::expr_to_string(orig_lhs) == lhs_str
      })
    {
      entry[pos] = (
        outer_func.clone(),
        params.clone(),
        conditions.clone(),
        defaults.clone(),
        heads.clone(),
        final_body.clone(),
        func_call_expr.clone(),
        body_expr.clone(),
      );
    } else {
      entry.push((
        outer_func.clone(),
        params.clone(),
        conditions.clone(),
        defaults.clone(),
        heads.clone(),
        final_body.clone(),
        func_call_expr.clone(),
        body_expr.clone(),
      ));
    }
  });

  // Also store in FUNC_DEFS under the outer function name so that
  // the existing function matching infrastructure picks it up.
  // Remove any existing upvalue definition with the same params/heads
  // before inserting (to avoid duplicates on redefinition).
  let blank_types = vec![1u8; params.len()];
  FUNC_DEFS.with(|m| {
    let mut defs = m.borrow_mut();
    let entry = defs.entry(outer_func).or_insert_with(Vec::new);
    entry.retain(|(p, _, _, h, bt, _)| {
      !(p == &params && h == &heads && bt == &blank_types)
    });
    // UpValue definitions go at the beginning for priority
    entry.insert(
      0,
      (params, conditions, defaults, heads, blank_types, final_body),
    );
  });

  if evaluate_rhs {
    Ok(Some(syntax::expr_to_string(&body_expr)))
  } else {
    Ok(None)
  }
}

/// Handle TagUnset: tag /: f[args...] =.  (removes an upvalue)
fn execute_tag_unset(pair: Pair<Rule>) -> Result<(), InterpreterError> {
  let mut inner = pair.into_inner();

  // First child: tag identifier
  let tag_name = inner.next().unwrap().as_str().to_owned();

  // Next child: BaseFunctionCall (the LHS pattern)
  let func_call_pair = inner.next().unwrap();
  let func_call_expr = syntax::pair_to_expr(func_call_pair);

  // Extract outer function name from the LHS
  let outer_func = match &func_call_expr {
    syntax::Expr::FunctionCall { name, .. } => name.clone(),
    syntax::Expr::CurriedCall { func, .. } => {
      if let syntax::Expr::FunctionCall { name, .. } = func.as_ref() {
        name.clone()
      } else {
        return Ok(());
      }
    }
    _ => return Ok(()),
  };

  let lhs_str = syntax::expr_to_string(&func_call_expr);

  // Remove matching entries from UPVALUES
  let removed = UPVALUES.with(|m| {
    let mut defs = m.borrow_mut();
    let mut removed_entries = Vec::new();
    let mut should_remove_key = false;
    if let Some(entry) = defs.get_mut(&tag_name) {
      entry.retain(
        |(
          _of,
          _params,
          _conds,
          _defaults,
          _heads,
          _body,
          orig_lhs,
          _orig_body,
        )| {
          let orig_lhs_str = syntax::expr_to_string(orig_lhs);
          if orig_lhs_str == lhs_str {
            removed_entries
              .push((_params.clone(), syntax::expr_to_string(_body)));
            false
          } else {
            true
          }
        },
      );
      if entry.is_empty() {
        should_remove_key = true;
      }
    }
    if should_remove_key {
      defs.remove(&tag_name);
    }
    removed_entries
  });

  // Also remove from FUNC_DEFS
  if !removed.is_empty() {
    FUNC_DEFS.with(|m| {
      if let Some(entry) = m.borrow_mut().get_mut(&outer_func) {
        for (params, body_str) in &removed {
          entry.retain(|(p, _, _, _, _, b)| {
            !(p == params && syntax::expr_to_string(b) == *body_str)
          });
        }
      }
    });
  }

  Ok(())
}

/// Extract pattern name and head from an Expr (for TagSetDelayed processing).
/// Similar to extract_pattern_info in evaluator.rs but works on Expr nodes.
fn extract_pattern_info_from_expr(
  expr: &syntax::Expr,
) -> (String, Option<String>) {
  match expr {
    syntax::Expr::Pattern { name, head, .. } => (name.clone(), head.clone()),
    syntax::Expr::PatternOptional { name, head, .. } => {
      (name.clone(), head.clone())
    }
    syntax::Expr::PatternTest { name, head, .. } => {
      (name.clone(), head.clone())
    }
    syntax::Expr::Identifier(name) => {
      // Could be "x_Integer" or "x_" in text form
      if let Some(pos) = name.find('_') {
        let pat_name = name[..pos].to_string();
        let head = &name[pos + 1..];
        if head.is_empty() {
          (pat_name, None)
        } else {
          (pat_name, Some(head.to_string()))
        }
      } else {
        (String::new(), None) // Not a pattern
      }
    }
    _ => (String::new(), None),
  }
}
