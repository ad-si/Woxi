#[allow(unused_imports)]
use super::*;
use std::cell::RefCell;
use std::collections::HashMap;

// Stream registry for open streams (InputStream/OutputStream)
#[derive(Clone, Debug)]
enum StreamKind {
  Text(String), // content of the string
  // Only the native build opens files: `OpenRead`/`OpenWrite`/`OpenAppend`
  // are compiled out on wasm, where there is no local filesystem.
  #[cfg(not(target_arch = "wasm32"))]
  File(String), // file path
  // A file specification starting with `!` names an external command
  // instead of a file; the command runs through the shell and its
  // standard output is what the stream reads.
  #[cfg(not(target_arch = "wasm32"))]
  Command(std::rc::Rc<RefCell<CommandStreamState>>),
  // The write end of the same convention: `OpenWrite["!command"]` feeds
  // what is written to it into the command's standard input.
  #[cfg(not(target_arch = "wasm32"))]
  CommandSink(std::rc::Rc<RefCell<CommandSinkState>>),
}

/// State of a `"!command"` stream: the running child process plus the part
/// of its standard output consumed so far. Output already handed out has to
/// stay addressable because streams are position-indexed, so it is kept in
/// `buffer` rather than discarded.
///
/// The buffer holds raw bytes so `BinaryRead` sees the command's output
/// unaltered; text reads decode it lossily, which leaves byte offsets and
/// string offsets in step for the valid UTF-8 that text reads assume.
#[cfg(not(target_arch = "wasm32"))]
#[derive(Debug)]
struct CommandStreamState {
  child: Option<std::process::Child>,
  reader: Option<std::io::BufReader<std::process::ChildStdout>>,
  buffer: Vec<u8>,
  eof: bool,
}

/// State of a `"!command"` output stream: the running child process and the
/// handle on its standard input. Closing the stream drops the handle, which
/// is the command's end-of-input, and then waits for it to finish.
#[cfg(not(target_arch = "wasm32"))]
#[derive(Debug)]
struct CommandSinkState {
  child: Option<std::process::Child>,
  stdin: Option<std::process::ChildStdin>,
}

/// Build the shell invocation that runs `command`.
#[cfg(not(target_arch = "wasm32"))]
fn shell_command(command: &str) -> std::process::Command {
  let mut cmd = if cfg!(target_os = "windows") {
    std::process::Command::new("cmd")
  } else {
    std::process::Command::new("sh")
  };
  cmd.arg(if cfg!(target_os = "windows") {
    "/C"
  } else {
    "-c"
  });
  cmd.arg(command);
  cmd
}

/// Start `command` through the shell, with its standard output piped back
/// to us. Standard input and standard error are inherited so that filters
/// like `"!cat"` or `"!grep foo"` read the script's own input and report
/// their errors on the terminal, as they do in wolframscript.
#[cfg(not(target_arch = "wasm32"))]
fn spawn_command_stream(command: &str) -> std::io::Result<CommandStreamState> {
  use std::process::Stdio;
  let mut child = shell_command(command)
    .stdin(Stdio::inherit())
    .stdout(Stdio::piped())
    .stderr(Stdio::inherit())
    .spawn()?;
  let reader = child.stdout.take().map(std::io::BufReader::new);
  Ok(CommandStreamState {
    child: Some(child),
    reader,
    buffer: Vec::new(),
    eof: false,
  })
}

/// Start `command` through the shell with its standard input piped from us.
/// Its output streams straight to the terminal, as it does in wolframscript.
#[cfg(not(target_arch = "wasm32"))]
fn spawn_command_sink(command: &str) -> std::io::Result<CommandSinkState> {
  use std::process::Stdio;
  let mut child = shell_command(command)
    .stdin(Stdio::piped())
    .stdout(Stdio::inherit())
    .stderr(Stdio::inherit())
    .spawn()?;
  let stdin = child.stdin.take();
  Ok(CommandSinkState {
    child: Some(child),
    stdin,
  })
}

/// Feed `bytes` to an open command sink. `false` if the command is gone —
/// it exited early, or the stream has already been closed.
#[cfg(not(target_arch = "wasm32"))]
fn command_sink_write(state: &RefCell<CommandSinkState>, bytes: &[u8]) -> bool {
  use std::io::Write;
  let mut state = state.borrow_mut();
  let Some(stdin) = state.stdin.as_mut() else {
    return false;
  };
  stdin.write_all(bytes).is_ok() && stdin.flush().is_ok()
}

/// Close a command sink: drop the pipe so the command sees end-of-input,
/// then wait for it to finish so its output lands before evaluation goes on.
#[cfg(not(target_arch = "wasm32"))]
fn command_sink_close(state: &RefCell<CommandSinkState>) {
  let mut state = state.borrow_mut();
  drop(state.stdin.take());
  if let Some(mut child) = state.child.take() {
    let _ = child.wait();
  }
}

/// Run `command` through the shell, feed it `input`, and wait for it.
/// `false` if the shell could not be started at all.
#[cfg(not(target_arch = "wasm32"))]
fn run_command_with_input(command: &str, input: &[u8]) -> bool {
  let Ok(state) = spawn_command_sink(command) else {
    return false;
  };
  let state = RefCell::new(state);
  command_sink_write(&state, input);
  command_sink_close(&state);
  true
}

/// Release the pipe and reap the child once its output is exhausted.
#[cfg(not(target_arch = "wasm32"))]
fn command_stream_finish(state: &mut CommandStreamState) {
  state.eof = true;
  state.reader = None;
  if let Some(mut child) = state.child.take() {
    let _ = child.wait();
  }
}

/// Pull one more line of the child's output into the buffer.
/// Returns `false` once the command's output is exhausted.
#[cfg(not(target_arch = "wasm32"))]
fn command_stream_read_line(state: &mut CommandStreamState) -> bool {
  use std::io::BufRead;
  if state.eof {
    return false;
  }
  let Some(reader) = state.reader.as_mut() else {
    command_stream_finish(state);
    return false;
  };
  let mut chunk: Vec<u8> = Vec::new();
  match reader.read_until(b'\n', &mut chunk) {
    Ok(0) | Err(_) => {
      command_stream_finish(state);
      false
    }
    Ok(_) => {
      state.buffer.extend_from_slice(&chunk);
      true
    }
  }
}

/// Pull the rest of the child's output into the buffer.
#[cfg(not(target_arch = "wasm32"))]
fn command_stream_read_all(state: &mut CommandStreamState) {
  while command_stream_read_line(state) {}
}

/// Stop a command stream that is closed before its output ran out, so a
/// still-running command (`"!yes"`, an interactive filter) does not keep
/// the interpreter waiting.
#[cfg(not(target_arch = "wasm32"))]
fn command_stream_close(state: &RefCell<CommandStreamState>) {
  let mut state = state.borrow_mut();
  state.reader = None;
  if let Some(mut child) = state.child.take() {
    let _ = child.kill();
    let _ = child.wait();
  }
  state.eof = true;
}

/// Run `command` through the shell and return the bytes it writes to
/// standard output. `None` if the shell could not be started at all.
#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn run_command_capture_bytes(command: &str) -> Option<Vec<u8>> {
  let mut state = spawn_command_stream(command).ok()?;
  command_stream_read_all(&mut state);
  Some(state.buffer)
}

/// Run `command` through the shell and return everything it writes to
/// standard output as text. `None` if the shell could not be started at all.
#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn run_command_capture(command: &str) -> Option<String> {
  run_command_capture_bytes(command)
    .map(|bytes| String::from_utf8_lossy(&bytes).into_owned())
}

/// Split a Wolfram file specification into the external command it names.
/// `"!sort -u"` runs `sort -u`; anything without the leading `!` is a
/// plain file path.
#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn command_file_spec(spec: &str) -> Option<&str> {
  spec.strip_prefix('!')
}

/// The registry id of an `InputStream[name, id]` / `OutputStream[name, id]`.
fn io_stream_id(expr: &Expr) -> Option<usize> {
  let Expr::FunctionCall { name, args } = expr else {
    return None;
  };
  if (name != "InputStream" && name != "OutputStream") || args.len() != 2 {
    return None;
  }
  match &args[1] {
    Expr::Integer(id) => Some(*id as usize),
    _ => None,
  }
}

/// Bytes for a binary read. An open stream is served from the registry, so
/// a `"!command"` pipe reads the command's output; `path` (the stream's
/// name) is the fallback for a stream that is no longer open.
#[cfg(not(target_arch = "wasm32"))]
fn io_binary_bytes(expr: &Expr, path: &str) -> std::io::Result<Vec<u8>> {
  if let Some(id) = io_stream_id(expr)
    && let Some(bytes) = get_stream_bytes(id)
  {
    return Ok(bytes);
  }
  match command_file_spec(path) {
    Some(command) => run_command_capture_bytes(command)
      .ok_or_else(|| std::io::Error::other("cannot start the shell")),
    None => std::fs::read(path),
  }
}

/// Where a write function sends its bytes.
#[cfg(not(target_arch = "wasm32"))]
enum WriteTarget {
  /// Append to a file.
  File(String),
  /// The standard input of an open `OpenWrite["!command"]` stream.
  Sink(std::rc::Rc<RefCell<CommandSinkState>>),
  /// A `"!command"` named directly instead of through an open stream: the
  /// command runs for this one write and is fed exactly its bytes.
  Command(String),
}

/// Resolve the first argument of `Write`/`WriteString`/`BinaryWrite` to the
/// thing that receives the bytes. `None` for anything unwritable, which
/// leaves the call unevaluated.
#[cfg(not(target_arch = "wasm32"))]
fn io_write_target(expr: &Expr) -> Option<WriteTarget> {
  match expr {
    Expr::String(spec) => Some(match command_file_spec(spec) {
      Some(command) => WriteTarget::Command(command.to_string()),
      None => WriteTarget::File(spec.clone()),
    }),
    Expr::FunctionCall { name, args }
      if (name == "OutputStream" || name == "InputStream")
        && args.len() == 2 =>
    {
      let Expr::Integer(id) = &args[1] else {
        return None;
      };
      match get_stream_kind(*id as usize)?.0 {
        StreamKind::File(path) => Some(WriteTarget::File(path)),
        StreamKind::CommandSink(state) => Some(WriteTarget::Sink(state)),
        // Neither a string stream nor the read end of a pipe is writable.
        StreamKind::Text(_) | StreamKind::Command(_) => None,
      }
    }
    _ => None,
  }
}

/// Shared body of `WriteString` and `WriteLine`. `args[0]` is the channel and
/// `args[1..]` the things to write; they are concatenated, `terminator` is
/// appended, and the result goes out in a single write. `WriteLine` is just
/// `WriteString` with a newline terminator, so both come through here.
#[cfg(not(target_arch = "wasm32"))]
fn write_string_to_channel(
  args: &[Expr],
  terminator: &str,
  caller: &str,
) -> Result<Expr, InterpreterError> {
  // A `"!command"` named directly runs once per call, so the whole
  // argument list has to reach it in a single write.
  let mut text = String::new();
  for arg in &args[1..] {
    match arg {
      Expr::String(s) => text.push_str(s),
      other => text.push_str(&crate::syntax::expr_to_string(other)),
    }
  }
  text.push_str(terminator);

  // Special-case the standard streams so `WriteString["stdout", …]` and
  // `WriteString[$Output, …]` write to the process's stdout, matching
  // wolframscript. `$Output`/`"stdout"` map to stdout, `$Messages`/
  // `"stderr"` to stderr. Stdout writes also go through the captured
  // buffer (like Print) so they appear in `interpret_with_stdout`.
  let std_target = match &args[0] {
    Expr::String(name) if name == "stdout" => Some(true),
    Expr::String(name) if name == "stderr" => Some(false),
    Expr::Identifier(name) if name == "$Output" => Some(true),
    Expr::Identifier(name) if name == "$Messages" => Some(false),
    _ => None,
  };
  if let Some(is_stdout) = std_target {
    use std::io::Write;
    if is_stdout {
      if !crate::is_quiet_print() {
        print!("{text}");
        let _ = std::io::stdout().flush();
      }
      crate::capture_stdout_raw(&text);
    } else {
      eprint!("{text}");
      let _ = std::io::stderr().flush();
    }
    return Ok(Expr::Identifier("Null".to_string()));
  }

  let Some(target) = io_write_target(&args[0]) else {
    return Ok(unevaluated(caller, args));
  };
  write_target_bytes(&target, text.as_bytes(), caller)?;
  Ok(Expr::Identifier("Null".to_string()))
}

/// Send `bytes` to a write target, appending for files. `caller` only names
/// the function in the error message.
#[cfg(not(target_arch = "wasm32"))]
fn write_target_bytes(
  target: &WriteTarget,
  bytes: &[u8],
  caller: &str,
) -> Result<(), InterpreterError> {
  match target {
    WriteTarget::File(path) => {
      use std::io::Write;
      let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .map_err(|e| {
          InterpreterError::EvaluationError(format!(
            "{caller}: cannot open {path}: {e}"
          ))
        })?;
      file.write_all(bytes).map_err(|e| {
        InterpreterError::EvaluationError(format!("{caller}: write error: {e}"))
      })
    }
    WriteTarget::Sink(state) => {
      command_sink_write(state, bytes);
      Ok(())
    }
    WriteTarget::Command(command) => {
      if run_command_with_input(command, bytes) {
        Ok(())
      } else {
        Err(InterpreterError::EvaluationError(format!(
          "{caller}: cannot open !{command}."
        )))
      }
    }
  }
}

#[derive(Clone, Debug)]
#[allow(unused)] // id is unused
struct OpenStream {
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
fn register_stream(name: String, kind: StreamKind) -> usize {
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
fn close_stream(id: usize) -> Option<(String, StreamKind)> {
  STREAM_REGISTRY
    .with(|reg| reg.borrow_mut().remove(&id).map(|s| (s.name, s.kind)))
}

/// Check if a stream is open
fn is_stream_open(id: usize) -> bool {
  STREAM_REGISTRY.with(|reg| reg.borrow().contains_key(&id))
}

/// Look up a stream's kind and current read position.
/// The kind is cloned so the registry is not kept borrowed across a read,
/// which for a command stream can block on the child process.
fn get_stream_kind(id: usize) -> Option<(StreamKind, usize)> {
  STREAM_REGISTRY
    .with(|reg| reg.borrow().get(&id).map(|s| (s.kind.clone(), s.position)))
}

/// Get the remaining content of a stream (for reading)
fn get_stream_content(id: usize) -> Option<(String, usize)> {
  let (kind, position) = get_stream_kind(id)?;
  let content = match &kind {
    StreamKind::Text(text) => text.clone(),
    #[cfg(not(target_arch = "wasm32"))]
    StreamKind::File(path) => std::fs::read_to_string(path).unwrap_or_default(),
    // Whole-stream reads have to wait for the command to finish.
    #[cfg(not(target_arch = "wasm32"))]
    StreamKind::Command(state) => {
      let mut state = state.borrow_mut();
      command_stream_read_all(&mut state);
      String::from_utf8_lossy(&state.buffer).into_owned()
    }
    // Nothing was ever read into a command sink.
    #[cfg(not(target_arch = "wasm32"))]
    StreamKind::CommandSink(_) => String::new(),
  };
  Some((content, position))
}

/// The raw bytes behind a stream, for `BinaryRead`. Command streams are
/// drained to the end, like any other whole-stream read.
#[cfg(not(target_arch = "wasm32"))]
fn get_stream_bytes(id: usize) -> Option<Vec<u8>> {
  match get_stream_kind(id)?.0 {
    StreamKind::Text(text) => Some(text.into_bytes()),
    StreamKind::File(path) => std::fs::read(path).ok(),
    StreamKind::Command(state) => {
      let mut state = state.borrow_mut();
      command_stream_read_all(&mut state);
      Some(state.buffer.clone())
    }
    StreamKind::CommandSink(_) => Some(Vec::new()),
  }
}

/// Content for a line-oriented read. A command stream only pulls as much of
/// its child's output as the next line needs, so `ReadLine` keeps working
/// line by line on a live pipe instead of blocking until the command exits.
#[cfg(not(target_arch = "wasm32"))]
fn get_stream_line_content(id: usize) -> Option<(String, usize)> {
  if let Some((StreamKind::Command(state), position)) = get_stream_kind(id) {
    let mut state = state.borrow_mut();
    while !state.eof
      && !state.buffer[position.min(state.buffer.len())..].contains(&b'\n')
    {
      command_stream_read_line(&mut state);
    }
    return Some((
      String::from_utf8_lossy(&state.buffer).into_owned(),
      position,
    ));
  }
  get_stream_content(id)
}

/// Get the current read position of a stream
fn get_stream_position(id: usize) -> Option<usize> {
  STREAM_REGISTRY.with(|reg| reg.borrow().get(&id).map(|s| s.position))
}

/// Set the read position of a stream to an absolute position
fn set_stream_position(id: usize, new_position: usize) {
  STREAM_REGISTRY.with(|reg| {
    let mut registry = reg.borrow_mut();
    if let Some(stream) = registry.get_mut(&id) {
      stream.position = new_position;
    }
  });
}

/// Advance the read position of a stream
fn advance_stream_position(id: usize, new_position: usize) {
  STREAM_REGISTRY.with(|reg| {
    let mut registry = reg.borrow_mut();
    if let Some(stream) = registry.get_mut(&id) {
      stream.position = new_position;
    }
  });
}

// Virtual working-directory stack used by SetDirectory / ResetDirectory.
//
// We deliberately do NOT call `std::env::set_current_dir` here: that mutates
// process-wide state, and cargo runs tests in parallel threads within a
// single process. Mutating the real CWD from one test races against any
// other test that resolves relative paths (Import, FileNames, etc.), causing
// flaky failures in CI. Instead we track a per-thread virtual stack; the top
// of the stack is what `Directory[]` reports, and the process CWD is used as
// the fallback when the stack is empty.
#[cfg(not(target_arch = "wasm32"))]
thread_local! {
  static DIRECTORY_STACK: std::cell::RefCell<Vec<String>> = const { std::cell::RefCell::new(Vec::new()) };
}

#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn virtual_current_dir() -> String {
  DIRECTORY_STACK
    .with(|s| s.borrow().last().cloned())
    .unwrap_or_else(|| {
      std::env::current_dir()
        .map(|p| p.to_string_lossy().into_owned())
        .unwrap_or_default()
    })
}

/// The file a `Get` / `Needs` argument names.
///
/// A context name — `"MyPaclet`"` — is looked up in the loaded paclet
/// directories and then along `$Path`; anything else is a file name, resolved
/// against the virtual working directory so a preceding `SetDirectory` is
/// honoured the way it is in wolframscript. `None` when no such file exists.
#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn resolve_get_target(name: &str) -> Option<std::path::PathBuf> {
  if name.ends_with('`') {
    return crate::functions::paclet::resolve_context(name);
  }
  let requested = std::path::Path::new(name);
  let resolved = if requested.is_absolute() {
    requested.to_path_buf()
  } else {
    std::path::PathBuf::from(virtual_current_dir()).join(requested)
  };
  resolved.is_file().then_some(resolved)
}

/// Read and evaluate the file at `path`, returning its last result.
/// `None` when the file cannot be read.
#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn evaluate_file(
  path: &std::path::Path,
) -> Option<Result<Expr, InterpreterError>> {
  let content = std::fs::read_to_string(path).ok()?;
  // The read file is the input file for the duration of the read, so
  // `$InputFileName` must point at it and not at the enclosing script.
  let _input_file_name = InputFileNameGuard::install(path);
  Some(evaluate_source(&content))
}

/// Evaluate Wolfram source, returning its last result — what `Get` yields
/// for whatever it read, be that a file or a command's output.
#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn evaluate_source(content: &str) -> Result<Expr, InterpreterError> {
  // Use interpret to evaluate the content (handles all node types
  // including FunctionDefinition, Expression, etc.)
  let result_str = crate::interpret(content)?;
  // `interpret` reports a `Null` result as the "\0" display-suppression
  // sentinel; as the value of `Get` it is the symbol `Null` again.
  if result_str == "\0" {
    return Ok(Expr::Identifier("Null".to_string()));
  }
  Ok(
    crate::syntax::string_to_expr(&result_str)
      .unwrap_or(Expr::Identifier(result_str)),
  )
}

/// Scopes `$InputFileName` to the file a `Get` is currently reading.
///
/// While a file is being read it *is* the input file, so `$InputFileName`
/// has to name it rather than the enclosing script. Restoring on drop (rather
/// than after the evaluation) keeps nested and sequential `Get`s correct even
/// when the read file raises an error.
#[cfg(not(target_arch = "wasm32"))]
struct InputFileNameGuard {
  prev: Option<crate::StoredValue>,
}

#[cfg(not(target_arch = "wasm32"))]
impl InputFileNameGuard {
  fn install(path: &std::path::Path) -> Self {
    let value = crate::StoredValue::ExprVal(Expr::String(
      path.to_string_lossy().into_owned(),
    ));
    let prev = crate::ENV
      .with(|e| e.borrow_mut().insert("$InputFileName".to_string(), value));
    Self { prev }
  }
}

#[cfg(not(target_arch = "wasm32"))]
impl Drop for InputFileNameGuard {
  fn drop(&mut self) {
    crate::ENV.with(|e| {
      let mut env = e.borrow_mut();
      match self.prev.take() {
        Some(v) => {
          env.insert("$InputFileName".to_string(), v);
        }
        None => {
          env.remove("$InputFileName");
        }
      }
    });
  }
}

pub fn dispatch_io_functions(
  name: &str,
  args: &[Expr],
) -> Option<Result<Expr, InterpreterError>> {
  match name {
    // Message[sym::tag, args...] — emit a message and return Null. Only matches
    // when the first argument is a MessageName; other shapes fall through to
    // stay unevaluated.
    "Message" if !args.is_empty() => {
      if let Expr::FunctionCall {
        name: mn_name,
        args: mn_args,
      } = &args[0]
        && mn_name == "MessageName"
        && mn_args.len() == 2
      {
        let sym_name = match &mn_args[0] {
          Expr::Identifier(s) => s.clone(),
          other => crate::syntax::expr_to_string(other),
        };
        let tag = match &mn_args[1] {
          Expr::String(s) => s.clone(),
          Expr::Identifier(s) => s.clone(),
          other => crate::syntax::expr_to_string(other),
        };
        // Evaluate MessageName[sym, tag]. If it resolves to a String, use it
        // as the text; otherwise treat the text as unset.
        let resolved = crate::evaluator::evaluate_expr_to_expr(&args[0]);
        let text = match &resolved {
          Ok(Expr::String(s)) => s.clone(),
          _ => "-- Message text not found --".to_string(),
        };
        // Fill the `1`, `2`, ... template slots with the extra arguments
        // (rendered in output form, so strings appear unquoted), matching
        // wolframscript: Message[f::mymsg, 42] shows "Custom 42 here.".
        let mut filled = text;
        for (i, arg) in args[1..].iter().enumerate() {
          let placeholder = format!("`{}`", i + 1);
          if filled.contains(&placeholder) {
            let shown =
              crate::syntax::format_expr(arg, crate::syntax::ExprForm::Output);
            filled = filled.replace(&placeholder, &shown);
          }
        }
        // Route through emit_message so the message is captured (Check
        // reacts to user messages), respects Quiet/Off, participates in
        // General::stop suppression, and reaches the same stream as
        // built-in messages.
        crate::emit_message(&format!("{sym_name}::{tag}: {filled}"));
        return Some(Ok(Expr::Identifier("Null".to_string())));
      }
    }
    // HTTPRequest[url] / HTTPRequest[url, assoc] / HTTPRequest[assoc] —
    // symbolic HTTP request object; no network access is performed.
    // The one-argument URL form canonicalizes to HTTPRequest[url, <||>],
    // matching wolframscript; other shapes stay as given.
    "HTTPRequest" if !args.is_empty() => {
      return Some(crate::functions::http_ast::http_request_ast(args));
    }
    // URLRead[req] / URLRead[url] — send the HTTP request through curl and
    // return the HTTPResponse object (or Failure["ConnectionFailure", …]).
    #[cfg(not(target_arch = "wasm32"))]
    "URLRead" if args.len() == 1 => {
      return Some(crate::functions::http_ast::url_read_ast(&args[0]));
    }
    // URLFetch[url] / URLFetch[url, params] — minimal stub.
    // Returns $Failed for URLs that lack a host (e.g. "https://"), matching
    // wolframscript's behavior. Network fetches are out of scope for the
    // CLI/snapshot test loop, so any other URL also returns $Failed.
    "URLFetch" if args.len() == 1 || args.len() == 2 => {
      if let Expr::String(_) = &args[0] {
        return Some(Ok(Expr::Identifier("$Failed".to_string())));
      }
    }
    // Environment["name"] — return the named environment variable value
    "Environment" if args.len() == 1 => {
      if let Expr::String(var_name) = &args[0] {
        return Some(Ok(match std::env::var(var_name) {
          Ok(val) => Expr::String(val),
          Err(_) => Expr::Identifier("$Failed".to_string()),
        }));
      }
      return Some(Ok(unevaluated("Environment", args)));
    }
    // GetEnvironment[] — all environment variables as a List of rules.
    "GetEnvironment" if args.is_empty() => {
      let rules: Vec<Expr> = std::env::vars()
        .map(|(k, v)| Expr::Rule {
          pattern: Box::new(Expr::String(k)),
          replacement: Box::new(Expr::String(v)),
        })
        .collect();
      return Some(Ok(Expr::List(rules.into())));
    }
    // GetEnvironment["name"] — return "name" -> "value" rule
    // GetEnvironment[{"n1","n2"}] — list of rules
    "GetEnvironment" if args.len() == 1 => {
      let make_rule = |var: &str| -> Expr {
        Expr::Rule {
          pattern: Box::new(Expr::String(var.to_string())),
          replacement: Box::new(match std::env::var(var) {
            Ok(val) => Expr::String(val),
            Err(_) => Expr::Identifier("None".to_string()),
          }),
        }
      };
      match &args[0] {
        Expr::String(var) => return Some(Ok(make_rule(var))),
        Expr::List(items) => {
          let rules: Vec<Expr> = items
            .iter()
            .map(|item| match item {
              Expr::String(v) => make_rule(v),
              _ => item.clone(),
            })
            .collect();
          return Some(Ok(Expr::List(rules.into())));
        }
        _ => {}
      }
      return Some(Ok(unevaluated("GetEnvironment", args)));
    }
    // SetEnvironment["name" -> "value"]   — set an environment variable
    // SetEnvironment["name" -> None]      — unset an environment variable
    // SetEnvironment[{rule1, rule2, ...}] — apply multiple rules
    // Returns Null on success, $Failed if any value is not a string or None.
    "SetEnvironment" if args.len() == 1 => {
      fn apply_rule(rule: &Expr) -> Option<bool> {
        let (pat, val) = match rule {
          Expr::Rule {
            pattern,
            replacement,
          }
          | Expr::RuleDelayed {
            pattern,
            replacement,
          } => (pattern.as_ref(), replacement.as_ref()),
          _ => return None,
        };
        let var = match pat {
          Expr::String(s) => s.clone(),
          _ => return Some(false),
        };
        match val {
          Expr::String(v) => {
            // SAFETY: Woxi is single-threaded in the REPL / CLI path.
            unsafe { std::env::set_var(&var, v) };
            Some(true)
          }
          Expr::Identifier(name) if name == "None" => {
            unsafe { std::env::remove_var(&var) };
            Some(true)
          }
          _ => {
            eprintln!(
              "SetEnvironment::setraw: {} must be a string or None.",
              crate::syntax::expr_to_string(val)
            );
            Some(false)
          }
        }
      }
      let ok = match &args[0] {
        Expr::List(rules) => {
          let mut all_ok = true;
          for r in rules {
            match apply_rule(r) {
              Some(true) => {}
              _ => all_ok = false,
            }
          }
          all_ok
        }
        other => matches!(apply_rule(other), Some(true)),
      };
      return Some(Ok(if ok {
        Expr::Identifier("Null".to_string())
      } else {
        Expr::Identifier("$Failed".to_string())
      }));
    }
    // Streams[] — return list of open streams (stdout and stderr)
    "Streams" if args.is_empty() => {
      return Some(Ok(Expr::List(
        vec![
          call(
            "OutputStream",
            vec![Expr::String("stdout".to_string()), Expr::Integer(1)],
          ),
          call(
            "OutputStream",
            vec![Expr::String("stderr".to_string()), Expr::Integer(2)],
          ),
        ]
        .into(),
      )));
    }
    // Streams["name"] — filter streams by name
    "Streams" if args.len() == 1 => {
      if let Expr::String(name_filter) = &args[0] {
        let all_streams = [("stdout", 1), ("stderr", 2)];
        let matching: Vec<Expr> = all_streams
          .iter()
          .filter(|(n, _)| *n == name_filter.as_str())
          .map(|(n, id)| {
            call(
              "OutputStream",
              vec![Expr::String(n.to_string()), Expr::Integer(*id)],
            )
          })
          .collect();
        return Some(Ok(Expr::List(matching.into())));
      }
      return Some(Ok(Expr::List(vec![].into())));
    }
    // ReadList[source] or ReadList[source, type] or ReadList[source, type, n]
    "ReadList" if !args.is_empty() && args.len() <= 3 => {
      return Some(crate::functions::string_ast::read_list_ast(args));
    }
    // ReadString["file"] — read a host-registered virtual file (WASM).
    // The browser has no local filesystem, so the virtual store registered
    // via `set_virtual_file` is the only file source.
    #[cfg(target_arch = "wasm32")]
    "ReadString" if args.len() == 1 => {
      let filename = match &args[0] {
        Expr::String(s) => s.clone(),
        _ => {
          return Some(Ok(unevaluated("ReadString", args)));
        }
      };
      let Some(bytes) = crate::wasm::virtual_file(&filename) else {
        crate::emit_message(&format!(
          "ReadString::noopen: Cannot open {}.",
          filename
        ));
        return Some(Ok(Expr::Identifier("$Failed".to_string())));
      };
      return Some(match String::from_utf8(bytes) {
        Ok(content) => Ok(Expr::String(content)),
        Err(_) => Err(InterpreterError::EvaluationError(format!(
          "ReadString: \"{}\" is not valid UTF-8 text",
          filename
        ))),
      });
    }
    // ReadString[src] — everything left in `src`;
    // ReadString[src, term] — up to the next terminator.
    // `src` is a file path or an InputStream, which the read consumes.
    #[cfg(not(target_arch = "wasm32"))]
    "ReadString" if args.len() == 1 || args.len() == 2 => {
      // Only a string terminator is meaningful.
      let terminator = match args.get(1) {
        None => None,
        Some(Expr::String(t)) if !t.is_empty() => Some(t.clone()),
        Some(other) => {
          crate::emit_message(&format!(
            "ReadString::iterm: Invalid terminator value {}.",
            crate::syntax::expr_to_string(other)
          ));
          return Some(Ok(unevaluated("ReadString", args)));
        }
      };

      let (content, position, stream_id) = match &args[0] {
        Expr::String(path) => {
          let content = match command_file_spec(path) {
            Some(command) => run_command_capture(command),
            None => std::fs::read_to_string(path).ok(),
          };
          if let Some(c) = content {
            (c, 0usize, None)
          } else {
            crate::emit_message(&format!(
              "ReadString::noopen: Cannot open {path}."
            ));
            return Some(Ok(Expr::Identifier("$Failed".to_string())));
          }
        }
        Expr::FunctionCall {
          name: stream_head,
          args: stream_args,
        } if stream_head == "InputStream" && stream_args.len() == 2 => {
          let Some(Expr::Integer(id)) = stream_args.get(1) else {
            return Some(Ok(unevaluated("ReadString", args)));
          };
          match get_stream_content(*id as usize) {
            Some((c, pos)) => (c, pos, Some(*id as usize)),
            None => {
              return Some(Ok(Expr::Identifier("EndOfFile".to_string())));
            }
          }
        }
        _ => return Some(Ok(unevaluated("ReadString", args))),
      };

      let rest = &content[position.min(content.len())..];
      let Some((text, consumed)) =
        read_string_chunk(rest, terminator.as_deref())
      else {
        return Some(Ok(Expr::Identifier("EndOfFile".to_string())));
      };
      if let Some(id) = stream_id {
        set_stream_position(id, position + consumed);
      }
      return Some(Ok(Expr::String(text)));
    }
    // FileTemplate[src] / FileTemplate[src, args] — read a template file from
    // disk and produce a TemplateObject (the same object StringTemplate would
    // build from the file's contents). `src` may be a path string or a
    // File["path"] wrapper.
    #[cfg(not(target_arch = "wasm32"))]
    "FileTemplate" if args.len() == 1 || args.len() == 2 => {
      let filename = match &args[0] {
        Expr::String(s) => s.clone(),
        Expr::FunctionCall { name, args: inner }
          if name == "File"
            && inner.len() == 1
            && matches!(&inner[0], Expr::String(_)) =>
        {
          match &inner[0] {
            Expr::String(s) => s.clone(),
            _ => unreachable!(),
          }
        }
        // URL[…] / CloudObject[…] and other specifications are left
        // unevaluated (network access is out of scope).
        _ => {
          return Some(Ok(unevaluated("FileTemplate", args)));
        }
      };
      // Resolve relative paths against the virtual working directory.
      let requested = std::path::Path::new(&filename);
      let resolved = if requested.is_absolute() {
        requested.to_path_buf()
      } else {
        std::path::PathBuf::from(virtual_current_dir()).join(requested)
      };
      let Ok(content) = std::fs::read_to_string(&resolved) else {
        crate::emit_message_to_stdout(&format!(
          "StringTemplate::fnfnd: File \"{filename}\" not found."
        ));
        return Some(Ok(Expr::Identifier("$Failed".to_string())));
      };
      let bound_args = if args.len() == 2 {
        Some(args[1].clone())
      } else {
        None
      };
      return Some(Ok(crate::functions::string_ast::build_template_object(
        &content,
        bound_args,
        "TextString",
      )));
    }
    // XMLTemplate[src] / XMLTemplate[src, args] — like StringTemplate but with
    // InsertionFunction -> HTMLFragment. `src` may be a literal template string
    // or a File["path"] wrapper that is read from disk. The template string may
    // embed `<* expr *>` sections in addition to `` `slot` `` markers.
    "XMLTemplate" if args.len() == 1 || args.len() == 2 => {
      let content = match &args[0] {
        Expr::String(s) => s.clone(),
        #[cfg(not(target_arch = "wasm32"))]
        Expr::FunctionCall { name, args: inner }
          if name == "File"
            && inner.len() == 1
            && matches!(&inner[0], Expr::String(_)) =>
        {
          let filename = match &inner[0] {
            Expr::String(s) => s.clone(),
            _ => unreachable!(),
          };
          let requested = std::path::Path::new(&filename);
          let resolved = if requested.is_absolute() {
            requested.to_path_buf()
          } else {
            std::path::PathBuf::from(virtual_current_dir()).join(requested)
          };
          if let Ok(c) = std::fs::read_to_string(&resolved) {
            c
          } else {
            crate::emit_message_to_stdout(&format!(
              "XMLTemplate::fnfnd: File \"{filename}\" not found."
            ));
            return Some(Ok(Expr::Identifier("$Failed".to_string())));
          }
        }
        // URL[…] / CloudObject[…] and other specifications are left
        // unevaluated (network access is out of scope).
        _ => {
          return Some(Ok(unevaluated("XMLTemplate", args)));
        }
      };
      let bound_args = if args.len() == 2 {
        Some(args[1].clone())
      } else {
        None
      };
      return Some(Ok(crate::functions::string_ast::build_template_object(
        &content,
        bound_args,
        "HTMLFragment",
      )));
    }
    // Get[file] — read and evaluate a file, returning the last result
    #[cfg(not(target_arch = "wasm32"))]
    "Get" if args.len() == 1 => {
      let filename = match &args[0] {
        Expr::String(s) => s.clone(),
        Expr::Identifier(s) => s.clone(),
        _ => {
          return Some(Ok(unevaluated("Get", args)));
        }
      };
      // A package that ships with the Wolfram Language loads as a no-op:
      // Woxi keeps every built-in in one namespace, so there is nothing to
      // read and nothing to define.
      if crate::utils::is_standard_distribution_context(&filename) {
        return Some(Ok(Expr::Identifier("Null".to_string())));
      }
      // `"!command"` evaluates the code the command writes to its standard
      // output, the read counterpart of `Put["!command"]`.
      if let Some(command) = command_file_spec(&filename) {
        let Some(content) = run_command_capture(command) else {
          crate::emit_message_to_stdout(&format!(
            "Get::noopen: Cannot open {filename}."
          ));
          return Some(Ok(Expr::Identifier("$Failed".to_string())));
        };
        return Some(evaluate_source(&content));
      }
      let resolved = resolve_get_target(&filename);
      let loaded = resolved.as_deref().and_then(evaluate_file);
      let Some(result) = loaded else {
        // wolframscript prints this message to stdout (verified with
        // `wolframscript -file`), so mirror it into the captured buffer to
        // keep snapshot/playground/Jupyter output byte-for-byte consistent.
        crate::emit_message_to_stdout(&format!(
          "Get::noopen: Cannot open {filename}."
        ));
        return Some(Ok(Expr::Identifier("$Failed".to_string())));
      };
      return Some(result);
    }
    // PacletDirectoryLoad[] / PacletDirectoryLoad[dir1, dir2, …] /
    // PacletDirectoryLoad[{dir1, dir2, …}] — register directories the
    // paclet manager searches, and return every directory now loaded.
    // `PacletDirectoryUnload` takes the same arguments and unregisters them.
    // Directory names have to be given either all as separate arguments or
    // all in one list; a mixture stays unevaluated, as in wolframscript.
    #[cfg(not(target_arch = "wasm32"))]
    "PacletDirectoryLoad" | "PacletDirectoryUnload" => {
      let string_dir = |e: &Expr| match e {
        Expr::String(dir) => Some(dir.clone()),
        _ => None,
      };
      let requested: Vec<String> = if let [Expr::List(dirs)] = args {
        let Some(dirs) =
          dirs.iter().map(string_dir).collect::<Option<Vec<_>>>()
        else {
          return Some(Ok(unevaluated(name, args)));
        };
        // Unloading a *list* of directories unregisters only its last
        // entry in wolframscript — verified against 14.1; Woxi mirrors
        // that so scripts see identical `PacletDirectoryLoad[]` output.
        if name == "PacletDirectoryUnload" {
          dirs.last().into_iter().cloned().collect()
        } else {
          dirs
        }
      } else {
        let Some(dirs) =
          args.iter().map(string_dir).collect::<Option<Vec<_>>>()
        else {
          return Some(Ok(unevaluated(name, args)));
        };
        dirs
      };
      let loaded = if name == "PacletDirectoryLoad" {
        crate::functions::paclet::load_directories(&requested)
      } else {
        crate::functions::paclet::unload_directories(&requested)
      };
      return Some(Ok(Expr::List(
        loaded.into_iter().map(Expr::String).collect(),
      )));
    }
    // Put[expr1, expr2, ..., "file"] — write expressions to a file
    #[cfg(not(target_arch = "wasm32"))]
    "Put" if !args.is_empty() => {
      let filename = match args.last().unwrap() {
        Expr::String(s) => s.clone(),
        _ => {
          return Some(Ok(unevaluated("Put", args)));
        }
      };
      let exprs = &args[..args.len() - 1];
      let content = exprs
        .iter()
        .map(crate::syntax::expr_to_string)
        .collect::<Vec<_>>()
        .join("\n");
      let to_write = if exprs.is_empty() {
        String::new()
      } else {
        format!("{content}\n")
      };
      // `"!command"` feeds the expressions to a command instead of a file.
      if let Some(command) = command_file_spec(&filename) {
        if run_command_with_input(command, to_write.as_bytes()) {
          return Some(Ok(Expr::Identifier("Null".to_string())));
        }
        crate::emit_message(&format!("Put::noopen: Cannot open {filename}."));
        return Some(Ok(Expr::Identifier("$Failed".to_string())));
      }
      match std::fs::write(&filename, to_write) {
        Ok(()) => return Some(Ok(Expr::Identifier("Null".to_string()))),
        Err(_e) => {
          crate::emit_message(&format!("Put::noopen: Cannot open {filename}."));
          return Some(Ok(Expr::Identifier("$Failed".to_string())));
        }
      }
    }
    // PutAppend[expr1, expr2, ..., "file"] — append expressions to a file
    #[cfg(not(target_arch = "wasm32"))]
    "PutAppend" if !args.is_empty() => {
      let filename = match args.last().unwrap() {
        Expr::String(s) => s.clone(),
        _ => {
          return Some(Ok(unevaluated("PutAppend", args)));
        }
      };
      let exprs = &args[..args.len() - 1];
      let content = exprs
        .iter()
        .map(crate::syntax::expr_to_string)
        .collect::<Vec<_>>()
        .join("\n");
      if !exprs.is_empty() {
        use std::io::Write;
        let to_write = format!("{content}\n");
        // A pipe has nothing to append to, so `"!command"` just runs.
        if let Some(command) = command_file_spec(&filename) {
          if run_command_with_input(command, to_write.as_bytes()) {
            return Some(Ok(Expr::Identifier("Null".to_string())));
          }
          crate::emit_message(&format!(
            "PutAppend::noopen: Cannot open {filename}."
          ));
          return Some(Ok(Expr::Identifier("$Failed".to_string())));
        }
        if let Ok(mut file) = std::fs::OpenOptions::new()
          .create(true)
          .append(true)
          .open(&filename)
        {
          if file.write_all(to_write.as_bytes()).is_err() {
            crate::emit_message(&format!(
              "PutAppend::noopen: Cannot open {filename}."
            ));
            return Some(Ok(Expr::Identifier("$Failed".to_string())));
          }
        } else {
          crate::emit_message(&format!(
            "PutAppend::noopen: Cannot open {filename}."
          ));
          return Some(Ok(Expr::Identifier("$Failed".to_string())));
        }
      }
      return Some(Ok(Expr::Identifier("Null".to_string())));
    }
    #[cfg(not(target_arch = "wasm32"))]
    "Export" if args.len() >= 2 => {
      let filename = match &args[0] {
        Expr::String(s) => s.clone(),
        other => {
          return Some(Err(InterpreterError::EvaluationError(format!(
            "Export: first argument must be a filename string, got {}",
            crate::syntax::expr_to_string(other)
          ))));
        }
      };
      // Determine the export format from the explicit third argument or,
      // failing that, from the filename extension.
      let explicit_fmt = args.get(2).and_then(|a| {
        if let Expr::String(s) = a {
          Some(s.to_ascii_uppercase())
        } else {
          None
        }
      });
      let ext_fmt = std::path::Path::new(&filename)
        .extension()
        .and_then(|e| e.to_str())
        .map(str::to_ascii_uppercase);
      let fmt = explicit_fmt.or(ext_fmt).unwrap_or_default();

      // Exporting a graphic writes it to the file; it must not also appear as
      // inline output in visual frontends (playground, woxi-studio). Evaluating
      // `args[1]` (e.g. `BarChart[...]`) already pushed its SVG into the capture
      // buffer, so drop that entry. The `-Graphics-` placeholder case is handled
      // in the generic branch below, after its SVG is read from the buffer.
      if let Expr::Graphics { svg, .. } = &args[1] {
        crate::remove_captured_graphics(svg);
      }

      // Handle Image export.  Vector formats (SVG) wrap the raster in a
      // base64-encoded PNG <image> element so the file is a valid SVG;
      // every other format is written as a raster file by the image crate.
      if let Expr::Image {
        width,
        height,
        channels,
        data,
        ..
      } = &args[1]
      {
        if fmt == "SVG" {
          let svg = crate::functions::image_ast::image_to_svg_document(
            *width, *height, *channels, data,
          );
          if let Err(e) = std::fs::write(&filename, &svg).map_err(|e| {
            InterpreterError::EvaluationError(format!("Export: {e}"))
          }) {
            return Some(Err(e));
          }
          return Some(Ok(Expr::String(filename)));
        }
        if let Err(e) = crate::functions::image_ast::export_image(
          &filename, *width, *height, *channels, data,
        ) {
          return Some(Err(e));
        }
        return Some(Ok(Expr::String(filename)));
      }

      if fmt == "XLSX" {
        if let Err(e) =
          crate::functions::xlsx_ast::xlsx_export_file(&filename, &args[1])
        {
          return Some(Err(e));
        }
        return Some(Ok(Expr::String(filename)));
      }

      if fmt == "PDF" {
        let svg = expr_to_svg(&args[1]);
        match svg_to_pdf_bytes(&svg) {
          Ok(pdf_bytes) => {
            if let Err(e) = std::fs::write(&filename, &pdf_bytes).map_err(|e| {
              InterpreterError::EvaluationError(format!("Export: {e}"))
            }) {
              return Some(Err(e));
            }
            return Some(Ok(Expr::String(filename)));
          }
          Err(e) => return Some(Err(e)),
        }
      }

      if fmt == "SVG" {
        let svg = expr_to_svg(&args[1]);
        // expr_to_svg returns an empty string when a graphics head fails to
        // render; fall back to the text rendering so the file stays valid SVG.
        let svg = if svg.is_empty() {
          expr_text_svg(&args[1])
        } else {
          svg
        };
        let svg = embed_used_fonts(&svg);
        if let Err(e) = std::fs::write(&filename, &svg).map_err(|e| {
          InterpreterError::EvaluationError(format!("Export: {e}"))
        }) {
          return Some(Err(e));
        }
        return Some(Ok(Expr::String(filename)));
      }

      // Raster image formats: rasterize the SVG and write via the image crate.
      if matches!(
        fmt.as_str(),
        "PNG" | "JPG" | "JPEG" | "GIF" | "BMP" | "TIF" | "TIFF"
      ) {
        // Parse ImageResolution option (default 96 DPI to match
        // usvg's default output resolution).
        let mut dpi: f64 = 96.0;
        // Frame delay in hundredths of a second (GIF's native unit).
        // Default 1/8 s = 12 (matches Mathematica's 8 fps default for
        // animated GIF export).
        let mut frame_delay_hundredths: u16 = 12;
        for opt in &args[2..] {
          if let Expr::Rule {
            pattern,
            replacement,
          } = opt
            && let Expr::Identifier(k) = pattern.as_ref()
          {
            match k.as_str() {
              "ImageResolution" => match replacement.as_ref() {
                Expr::Integer(n) => dpi = *n as f64,
                Expr::Real(f) => dpi = *f,
                _ => {
                  return Some(Err(InterpreterError::EvaluationError(
                    "Export: ImageResolution must be a numeric value".into(),
                  )));
                }
              },
              "AnimationRate" | "FrameRate" => {
                // Frames per second → hundredths-of-a-second per frame.
                let fps = match replacement.as_ref() {
                  Expr::Integer(n) => *n as f64,
                  Expr::Real(f) => *f,
                  _ => 30.0,
                };
                if fps > 0.0 {
                  frame_delay_hundredths =
                    (100.0 / fps).round().clamp(1.0, 65535.0) as u16;
                }
              }
              _ => {}
            }
          }
        }

        // Animated GIF path: when exporting a list of graphics to GIF,
        // rasterize each element as a frame.
        if fmt == "GIF"
          && let Expr::List(items) = &args[1]
          && items.len() >= 2
          && items.iter().all(is_rasterizable_frame)
        {
          let mut frames =
            Vec::<crate::functions::image_ast::GifFrame>::with_capacity(
              items.len(),
            );
          for item in items {
            let svg = expr_to_svg(item);
            match crate::functions::image_ast::rasterize_svg(&svg, dpi) {
              Ok(Expr::Image {
                width,
                height,
                channels,
                ref data,
                ..
              }) => {
                let dyn_img =
                  crate::functions::image_ast::expr_to_dynamic_image(
                    width, height, channels, data,
                  );
                frames.push(crate::functions::image_ast::GifFrame {
                  image: dyn_img.to_rgba8(),
                  delay_hundredths: frame_delay_hundredths,
                });
              }
              Ok(_) => unreachable!("rasterize_svg returns Expr::Image"),
              Err(e) => return Some(Err(e)),
            }
          }
          if let Err(e) =
            crate::functions::image_ast::export_animated_gif(&filename, frames)
          {
            return Some(Err(e));
          }
          return Some(Ok(Expr::String(filename)));
        }

        let svg = expr_to_svg(&args[1]);
        match crate::functions::image_ast::rasterize_svg(&svg, dpi) {
          Ok(Expr::Image {
            width,
            height,
            channels,
            ref data,
            ..
          }) => {
            if let Err(e) = crate::functions::image_ast::export_image(
              &filename, width, height, channels, data,
            ) {
              return Some(Err(e));
            }
            return Some(Ok(Expr::String(filename)));
          }
          Ok(_) => unreachable!("rasterize_svg returns Expr::Image"),
          Err(e) => return Some(Err(e)),
        }
      }
      // WAV export of a playable sound (Play[…] / Sound[…] / Audio[…]).
      if matches!(fmt.as_str(), "WAV" | "WAVE")
        && let Some(bytes) =
          crate::functions::sound::expr_to_wav_bytes(&args[1])
      {
        if let Err(e) = std::fs::write(&filename, &bytes).map_err(|e| {
          InterpreterError::EvaluationError(format!("Export: {e}"))
        }) {
          return Some(Err(e));
        }
        return Some(Ok(Expr::String(filename)));
      }

      // MIDI export of a computational-music object (MusicScore / MusicVoice / …).
      if (fmt == "MID" || fmt == "MIDI")
        && let Some(bytes) =
          crate::functions::music_midi::music_to_midi(&args[1])
      {
        if let Err(e) = std::fs::write(&filename, &bytes).map_err(|e| {
          InterpreterError::EvaluationError(format!("Export: {e}"))
        }) {
          return Some(Err(e));
        }
        return Some(Ok(Expr::String(filename)));
      }

      // The second argument has already been evaluated, which triggers
      // capture_graphics() for Plot expressions.  Grab the SVG.
      let content = match &args[1] {
        Expr::Graphics { svg: svg_data, .. } => svg_data.clone(),
        Expr::Identifier(s) if s == "-Graphics-" || s == "-Graphics3D-" => {
          match crate::get_captured_graphics().ok_or_else(|| {
            InterpreterError::EvaluationError(
              "Export: no graphics to export".into(),
            )
          }) {
            Ok(v) => {
              // Written to the file, so don't also render it inline.
              crate::remove_captured_graphics(&v);
              v
            }
            Err(e) => return Some(Err(e)),
          }
        }
        Expr::String(s) => s.clone(),
        other => crate::syntax::expr_to_string(other),
      };
      if let Err(e) = std::fs::write(&filename, &content)
        .map_err(|e| InterpreterError::EvaluationError(format!("Export: {e}")))
      {
        return Some(Err(e));
      }
      return Some(Ok(Expr::String(filename)));
    }
    // Browser (WASM) `Export`: there is no filesystem, so instead of writing to
    // disk we serialize the value and hand the bytes to the host via
    // `record_exported_file`, which surfaces them as downloads. Only the
    // formats whose encoders compile to `wasm32` are supported; native-only
    // formats (raster images, PDF, XLSX) return a clear error.
    #[cfg(target_arch = "wasm32")]
    "Export" if args.len() >= 2 => {
      let filename = match &args[0] {
        Expr::String(s) => s.clone(),
        other => {
          return Some(Err(InterpreterError::EvaluationError(format!(
            "Export: first argument must be a filename string, got {}",
            crate::syntax::expr_to_string(other)
          ))));
        }
      };
      // Format from an explicit third-argument string, else the file extension.
      let explicit_fmt = args.get(2).and_then(|a| {
        if let Expr::String(s) = a {
          Some(s.to_ascii_uppercase())
        } else {
          None
        }
      });
      let ext_fmt = std::path::Path::new(&filename)
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_ascii_uppercase());
      let fmt = explicit_fmt.or(ext_fmt).unwrap_or_default();
      let data = &args[1];

      // Exporting a graphic writes it to the file; it must not also appear as
      // inline output in the playground. Evaluating `args[1]` already pushed its
      // SVG into the capture buffer, so drop that entry. The `-Graphics-`
      // placeholder case is handled in the generic branch below, after its SVG
      // is read from the buffer.
      if let Expr::Graphics { svg, .. } = data {
        crate::remove_captured_graphics(svg);
      }

      // Set to Some(format) when `bytes` is SVG the host must rasterize in the
      // browser (graphics → PNG/JPEG), rather than ready-to-save file bytes.
      let mut rasterize: Option<String> = None;

      let bytes: Vec<u8> = match fmt.as_str() {
        "CSV" => export_string_csv(data, ',', true, true).into_bytes(),
        // Symbolic XML (an XMLElement or a whole XMLObject document) is
        // written back out as markup.
        "XML" => match crate::functions::xml_ast::xml_to_string(data) {
          Some(text) => text.into_bytes(),
          None => {
            return Some(Err(InterpreterError::EvaluationError(
              "Export: value is not symbolic XML".into(),
            )));
          }
        },
        "TSV" => export_string_csv(data, '\t', true, true).into_bytes(),
        "JSON" | "RAWJSON" => match export_string_json(data, 0, false) {
          Some(mut json) => {
            json.push('\n');
            json.into_bytes()
          }
          None => {
            return Some(Err(InterpreterError::EvaluationError(
              "Export: value cannot be serialized to JSON".into(),
            )));
          }
        },
        "SVG" => {
          let svg = expr_to_svg(data);
          // expr_to_svg is empty when a graphics head fails to render; fall
          // back to the text rendering so the file stays valid SVG.
          let svg = if svg.is_empty() {
            expr_text_svg(data)
          } else {
            svg
          };
          embed_used_fonts(&svg).into_bytes()
        }
        "WAV" | "WAVE" => {
          match crate::functions::sound::expr_to_wav_bytes(data) {
            Some(b) => b,
            None => {
              return Some(Err(InterpreterError::EvaluationError(
                "Export: value is not a playable sound".into(),
              )));
            }
          }
        }
        "MID" | "MIDI" => {
          match crate::functions::music_midi::music_to_midi(data) {
            Some(b) => b,
            None => {
              return Some(Err(InterpreterError::EvaluationError(
                "Export: value is not a music object".into(),
              )));
            }
          }
        }
        // Raster image formats. An existing Image is encoded here directly
        // (the `image` crate compiles to wasm). Graphics/plots can't be
        // rasterized in wasm (the SVG rasterizer, resvg, is native-only), so
        // their SVG is handed to the host to rasterize via a browser canvas —
        // which supports PNG and JPEG only. GIF/BMP/TIFF of a plot is rejected.
        "PNG" | "JPG" | "JPEG" | "GIF" | "BMP" | "TIF" | "TIFF" => match data {
          Expr::Image {
            width,
            height,
            channels,
            data: pixels,
            ..
          } => match crate::functions::image_ast::export_image_bytes(
            &fmt, *width, *height, *channels, pixels,
          ) {
            Ok(b) => b,
            Err(e) => return Some(Err(e)),
          },
          _ if matches!(fmt.as_str(), "PNG" | "JPG" | "JPEG") => {
            let svg = expr_to_svg(data);
            let svg = if svg.is_empty() {
              expr_text_svg(data)
            } else {
              svg
            };
            rasterize = Some(if fmt == "PNG" { "png" } else { "jpg" }.into());
            svg.into_bytes()
          }
          _ => {
            return Some(Err(InterpreterError::EvaluationError(format!(
              "Export: {} export of a non-image expression (e.g. a plot) is \
               not supported in the browser; export as PNG or SVG instead",
              fmt
            ))));
          }
        },
        // Formats whose encoders are native-only in the WASM build.
        "PDF" | "XLSX" => {
          return Some(Err(InterpreterError::EvaluationError(format!(
            "Export: \"{}\" export is not supported in the browser",
            fmt
          ))));
        }
        // Text and unrecognized formats: strings verbatim, a list one element
        // per line, graphics as their SVG, other expressions rendered directly.
        _ => {
          let elem = |e: &Expr| match e {
            Expr::String(s) => s.clone(),
            _ => crate::syntax::format_expr(e, crate::syntax::ExprForm::Output),
          };
          let content = match data {
            Expr::Graphics { svg, .. } => svg.clone(),
            Expr::Identifier(s) if s == "-Graphics-" || s == "-Graphics3D-" => {
              let svg = crate::get_captured_graphics().unwrap_or_default();
              // Written to the file, so don't also render it inline.
              crate::remove_captured_graphics(&svg);
              svg
            }
            Expr::String(s) => s.clone(),
            Expr::List(items) => {
              items.iter().map(elem).collect::<Vec<_>>().join("\n")
            }
            other => elem(other),
          };
          content.into_bytes()
        }
      };

      crate::wasm::record_exported_file(
        &filename,
        &bytes,
        rasterize.as_deref(),
      );
      return Some(Ok(Expr::String(filename)));
    }
    "ExportString" if args.len() == 2 || args.len() == 3 => {
      // ExportString[expr, "format"] - return string representation.
      // An optional third argument carries format options; for JSON the
      // "Compact" -> True option emits the value with no extra whitespace.
      let compact = matches!(args.get(2), Some(Expr::Rule { pattern, replacement })
        if matches!(pattern.as_ref(), Expr::String(s) if s == "Compact")
          && matches!(replacement.as_ref(), Expr::Identifier(v) if v == "True"));
      let format_str = match &args[1] {
        Expr::String(s) => s.clone(),
        _ => {
          // Return unevaluated for non-string format
          return Some(Ok(unevaluated("ExportString", args)));
        }
      };
      if format_str == "SVG" || format_str == "PDF" {
        let svg = expr_to_svg(&args[0]);
        if format_str == "PDF" {
          #[cfg(not(target_arch = "wasm32"))]
          {
            match svg_to_pdf_bytes(&svg) {
              Ok(pdf_bytes) => {
                // Return raw PDF bytes as a String (binary content)
                let pdf_str =
                  pdf_bytes.into_iter().map(|b| b as char).collect::<String>();
                return Some(Ok(Expr::String(pdf_str)));
              }
              Err(e) => return Some(Err(e)),
            }
          }
          #[cfg(target_arch = "wasm32")]
          {
            return Some(Ok(unevaluated("ExportString", args)));
          }
        }
        return Some(Ok(Expr::String(embed_used_fonts(&svg))));
      }
      if format_str == "CSV" || format_str == "TSV" {
        let sep = if format_str == "CSV" { ',' } else { '\t' };
        return Some(Ok(Expr::String(export_string_csv(
          &args[0], sep, true, true,
        ))));
      }
      // Symbolic XML is written back out as markup.
      if format_str == "XML" {
        return Some(Ok(
          match crate::functions::xml_ast::xml_to_string(&args[0]) {
            Some(text) => Expr::String(text),
            None => Expr::Identifier("$Failed".to_string()),
          },
        ));
      }
      // "Table" is tab-separated like TSV but leaves strings unquoted and
      // emits no trailing newline.
      if format_str == "Table" {
        return Some(Ok(Expr::String(export_string_csv(
          &args[0], '\t', false, false,
        ))));
      }
      if format_str == "JSON" || format_str == "RawJSON" {
        // A value JSON cannot represent fails the export outright (the
        // offending part has already been reported); the call does not come
        // back unevaluated.
        return Some(Ok(match export_string_json(&args[0], 0, compact) {
          Some(json) => Expr::String(json),
          None => Expr::Identifier("$Failed".to_string()),
        }));
      }
      // "String" is the expression's own text: ExportString[{{1, 2}}, "String"]
      // is "{{1, 2}}", not the row-per-line layout of "Text".
      if format_str == "String" {
        let text = match &args[0] {
          Expr::String(s) => s.clone(),
          other => {
            crate::syntax::format_expr(other, crate::syntax::ExprForm::Output)
          }
        };
        return Some(Ok(Expr::String(text)));
      }
      // "Text"/"Lines"/"List": a string is emitted verbatim; a list has each
      // element rendered (OutputForm, strings unquoted) on its own line; an
      // atom is rendered directly.
      if format_str == "Text" || format_str == "Lines" || format_str == "List" {
        let elem = |e: &Expr| match e {
          Expr::String(s) => s.clone(),
          _ => crate::syntax::format_expr(e, crate::syntax::ExprForm::Output),
        };
        let s = match &args[0] {
          Expr::String(s) => s.clone(),
          Expr::List(items) => {
            items.iter().map(elem).collect::<Vec<_>>().join("\n")
          }
          other => elem(other),
        };
        return Some(Ok(Expr::String(s)));
      }
      // Return unevaluated for unsupported formats
      return Some(Ok(unevaluated("ExportString", args)));
    }
    #[cfg(not(target_arch = "wasm32"))]
    "Find" if args.len() == 2 => {
      // Find[stream_or_file, "text" | {"a", "b", …}] - find first line
      // that contains any of the search strings. Accepts file paths,
      // InputStream[…] / OutputStream[…] backed by either a file or a
      // string buffer. Advances the stream's position past the matched
      // line so consecutive Find calls walk forward.
      let search_terms: Vec<String> = match &args[1] {
        Expr::String(s) => vec![s.clone()],
        Expr::List(items) => {
          let mut terms = Vec::with_capacity(items.len());
          for item in items {
            match item {
              Expr::String(s) => terms.push(s.clone()),
              _ => {
                return Some(Err(InterpreterError::EvaluationError(
                  "Find: second argument must be a string or a list of strings"
                    .into(),
                )));
              }
            }
          }
          terms
        }
        _ => {
          return Some(Err(InterpreterError::EvaluationError(
            "Find: second argument must be a string or a list of strings"
              .into(),
          )));
        }
      };

      // (content, start_pos, optional stream id for position advance)
      let (content, start_pos, stream_id) = match &args[0] {
        // A `"!command"` name searches the command's output.
        Expr::String(path) => {
          let body = match command_file_spec(path) {
            Some(command) => run_command_capture(command).ok_or_else(|| {
              InterpreterError::EvaluationError(format!(
                "Find: cannot run {command}"
              ))
            }),
            None => std::fs::read_to_string(path).map_err(|e| {
              InterpreterError::EvaluationError(format!("Find: {e}"))
            }),
          };
          match body {
            Ok(c) => (c, 0usize, None),
            Err(e) => return Some(Err(e)),
          }
        }
        Expr::FunctionCall {
          name: stream_head,
          args: stream_args,
        } if (stream_head == "InputStream"
          || stream_head == "OutputStream")
          && stream_args.len() == 2 =>
        {
          if let Expr::Integer(id) = &stream_args[1] {
            let id_usize = *id as usize;
            match get_stream_content(id_usize) {
              Some((c, p)) => (c, p, Some(id_usize)),
              None => return Some(Ok(Expr::Identifier("$Failed".to_string()))),
            }
          } else {
            return Some(Ok(Expr::Identifier("$Failed".to_string())));
          }
        }
        _ => {
          let arg_str = crate::syntax::expr_to_string(&args[0]);
          crate::emit_message(&format!(
            "Find::stream: {arg_str} is not a string, SocketObject, InputStream[ ] or OutputStream[ ]."
          ));
          return Some(Ok(Expr::Identifier("$Failed".to_string())));
        }
      };

      let remaining = &content[start_pos.min(content.len())..];
      let mut consumed = 0usize;
      for line in remaining.split_inclusive('\n') {
        let stripped = line
          .strip_suffix('\n')
          .unwrap_or(line)
          .trim_end_matches('\r');
        consumed += line.len();
        if search_terms.iter().any(|t| stripped.contains(t)) {
          if let Some(id) = stream_id {
            set_stream_position(id, start_pos + consumed);
          }
          return Some(Ok(Expr::String(stripped.to_string())));
        }
      }
      if let Some(id) = stream_id {
        set_stream_position(id, content.len());
      }
      return Some(Ok(Expr::Identifier("EndOfFile".to_string())));
    }
    #[cfg(not(target_arch = "wasm32"))]
    "FindList" => {
      // FindList[file(s), text(s)[, n]] — all lines containing any of the
      // search strings (literal, case-sensitive substrings). Errors return
      // $Failed with the matching wolframscript message; a failed file in
      // a file LIST contributes a $Failed element instead.
      let failed = || Some(Ok(Expr::Identifier("$Failed".to_string())));
      let call_display =
        || crate::syntax::expr_to_output(&unevaluated("FindList", args));
      if args.len() < 2 {
        let (tag, noun) = if args.len() == 1 {
          ("argtu", "1 argument")
        } else {
          ("argt", "0 arguments")
        };
        crate::emit_message(&format!(
          "FindList::{tag}: FindList called with {noun}; 2 or 3 arguments are expected."
        ));
        return failed();
      }
      if args.len() > 3 {
        for extra in &args[3..] {
          let is_opt = matches!(
            extra,
            Expr::Rule { .. } | Expr::RuleDelayed { .. } | Expr::List(_)
          );
          if !is_opt {
            crate::emit_message(&format!(
              "FindList::nonopt: Options expected (instead of {}) beyond position 3 in {}. An option must be a rule or a list of rules.",
              crate::syntax::expr_to_string(extra),
              call_display()
            ));
            return failed();
          }
        }
      }
      let terms: Vec<String> = match &args[1] {
        Expr::String(s) => vec![s.clone()],
        Expr::List(items)
          if !items.is_empty()
            && items.iter().all(|it| matches!(it, Expr::String(_))) =>
        {
          items
            .iter()
            .map(|it| {
              let Expr::String(s) = it else { unreachable!() };
              s.clone()
            })
            .collect()
        }
        _ => {
          crate::emit_message(&format!(
            "FindList::strs: A string or nonempty list of strings is expected at position 2 in {}.",
            call_display()
          ));
          return failed();
        }
      };
      let limit: usize = match args.get(2) {
        None => usize::MAX,
        Some(Expr::Integer(n)) if *n >= 0 => *n as usize,
        Some(_) => {
          crate::emit_message(&format!(
            "FindList::intnm: Non-negative machine-sized integer expected at position 3 in {}.",
            call_display()
          ));
          return failed();
        }
      };
      let (files, is_list): (Vec<Expr>, bool) = match &args[0] {
        Expr::String(_) => (vec![args[0].clone()], false),
        Expr::List(items) => (items.iter().cloned().collect(), true),
        other => {
          crate::emit_message(&format!(
            "FindList::stream: {} is not a string, SocketObject, InputStream[ ] or OutputStream[ ].",
            crate::syntax::expr_to_string(other)
          ));
          return failed();
        }
      };
      let mut out: Vec<Expr> = Vec::new();
      let mut found = 0usize;
      for f in &files {
        if found >= limit {
          break;
        }
        let Expr::String(path) = f else {
          crate::emit_message(&format!(
            "FindList::stream: {} is not a string, SocketObject, InputStream[ ] or OutputStream[ ].",
            crate::syntax::expr_to_string(f)
          ));
          if !is_list {
            return failed();
          }
          out.push(Expr::Identifier("$Failed".to_string()));
          continue;
        };
        // A `"!command"` entry searches that command's output.
        let read = match command_file_spec(path) {
          Some(command) => run_command_capture(command).ok_or(()),
          None => std::fs::read_to_string(path).map_err(|_| ()),
        };
        match read {
          Err(()) => {
            crate::emit_message(&format!(
              "FindList::noopen: Cannot open {path}."
            ));
            if !is_list {
              return failed();
            }
            out.push(Expr::Identifier("$Failed".to_string()));
          }
          Ok(content) => {
            for line in content.lines() {
              if found >= limit {
                break;
              }
              let stripped = line.trim_end_matches('\r');
              if terms.iter().any(|t| stripped.contains(t.as_str())) {
                out.push(Expr::String(stripped.to_string()));
                found += 1;
              }
            }
          }
        }
      }
      return Some(Ok(Expr::List(out.into())));
    }
    #[cfg(not(target_arch = "wasm32"))]
    "CreateFile" => {
      let filename_opt = if args.is_empty() {
        None
      } else if let Expr::String(s) = &args[0] {
        Some(s.clone())
      } else {
        let s = expr_to_raw_string(&args[0]);
        Some(s)
      };
      return Some(match crate::utils::create_file(filename_opt) {
        Ok(path) => Ok(Expr::String(path.to_string_lossy().into_owned())),
        Err(err) => Err(InterpreterError::EvaluationError(err.to_string())),
      });
    }
    #[cfg(not(target_arch = "wasm32"))]
    "Directory" if args.is_empty() => {
      return Some(Ok(Expr::String(virtual_current_dir())));
    }
    "NotebookDirectory" if args.is_empty() => {
      return Some(if let Some(dir) = crate::get_notebook_directory() {
        Ok(Expr::String(dir))
      } else {
        crate::emit_message(
          "NotebookDirectory::nosv: The notebook directory is not available outside a notebook front-end.",
        );
        Ok(unevaluated("NotebookDirectory", args))
      });
    }
    #[cfg(not(target_arch = "wasm32"))]
    "ParentDirectory" if args.is_empty() || args.len() == 1 => {
      let base = if args.is_empty() {
        virtual_current_dir()
      } else if let Expr::String(s) = &args[0] {
        s.clone()
      } else {
        return Some(Ok(unevaluated("ParentDirectory", args)));
      };
      let parent = std::path::Path::new(&base)
        .parent()
        .map_or_else(|| base.clone(), |p| p.to_string_lossy().into_owned());
      return Some(Ok(Expr::String(parent)));
    }
    // DirectoryName["path"] or DirectoryName["path", n]
    "DirectoryName" if args.len() == 1 || args.len() == 2 => {
      let path_str = match &args[0] {
        Expr::String(s) => s.clone(),
        _ => {
          return Some(Ok(unevaluated("DirectoryName", args)));
        }
      };
      let n = if args.len() == 2 {
        match &args[1] {
          Expr::Integer(i) if *i >= 1 => *i as usize,
          Expr::Integer(_) => {
            crate::emit_message(
              "DirectoryName::intpm: Positive machine-sized integer expected at position 2 in DirectoryName.",
            );
            return Some(Ok(unevaluated("DirectoryName", args)));
          }
          _ => {
            return Some(Ok(unevaluated("DirectoryName", args)));
          }
        }
      } else {
        1
      };

      let mut result = path_str;
      for _ in 0..n {
        if result.is_empty() {
          break;
        }
        // "/" has no parent
        let trimmed = result.trim_end_matches('/');
        if trimmed.is_empty() {
          // input was "/" or "///" etc.
          result = String::new();
          break;
        }
        // Find the last separator
        if let Some(pos) = trimmed.rfind('/') {
          result = trimmed[..=pos].to_string();
        } else {
          result = String::new();
          break;
        }
      }
      return Some(Ok(Expr::String(result)));
    }
    "ToFileName" if args.len() == 1 || args.len() == 2 => {
      let sep = std::path::MAIN_SEPARATOR.to_string();
      let collect_dirs = |expr: &Expr| -> Option<Vec<String>> {
        match expr {
          Expr::String(s) => Some(vec![s.clone()]),
          Expr::List(parts) => {
            let mut segments = Vec::with_capacity(parts.len());
            for p in parts {
              if let Expr::String(s) = p {
                segments.push(s.clone());
              } else {
                return None;
              }
            }
            Some(segments)
          }
          _ => None,
        }
      };
      if args.len() == 1 {
        if let Some(dirs) = collect_dirs(&args[0]) {
          let joined = dirs.join(&sep);
          return Some(Ok(Expr::String(format!("{joined}{sep}"))));
        }
      } else if let (Some(dirs), Expr::String(file)) =
        (collect_dirs(&args[0]), &args[1])
      {
        let mut all = dirs;
        all.push(file.clone());
        return Some(Ok(Expr::String(all.join(&sep))));
      }
      return Some(Ok(unevaluated("ToFileName", args)));
    }
    "FileNameJoin" if args.len() == 1 || args.len() == 2 => {
      // Detect OperatingSystem option from second argument (a Rule).
      let sep: char = if args.len() == 2 {
        let mut s = std::path::MAIN_SEPARATOR;
        if let Expr::Rule {
          pattern,
          replacement,
        } = &args[1]
          && matches!(pattern.as_ref(),
            Expr::Identifier(n) if n == "OperatingSystem")
          && let Expr::String(os) = replacement.as_ref()
        {
          s = if os == "Windows" { '\\' } else { '/' };
        }
        s
      } else {
        std::path::MAIN_SEPARATOR
      };
      if let Expr::List(parts) = &args[0] {
        let segments: Vec<String> = parts
          .iter()
          .filter_map(|e| {
            if let Expr::String(s) = e {
              Some(s.clone())
            } else {
              None
            }
          })
          .collect();
        if segments.len() == parts.len() {
          let joined = segments.join(&sep.to_string());
          return Some(Ok(Expr::String(joined)));
        }
      }
      return Some(Ok(unevaluated("FileNameJoin", args)));
    }
    "FileNameSplit" if args.len() == 1 => {
      if let Expr::String(s) = &args[0] {
        if s.is_empty() {
          return Some(Ok(Expr::List(vec![].into())));
        }
        let parts: Vec<Expr> = s
          .split('/')
          .collect::<Vec<&str>>()
          .into_iter()
          .enumerate()
          .filter(|(i, part)| !(*i > 0 && part.is_empty()))
          .map(|(_, part)| Expr::String(part.to_string()))
          .collect();
        return Some(Ok(Expr::List(parts.into())));
      }
      return Some(Ok(unevaluated("FileNameSplit", args)));
    }
    "FileNameDepth" if args.len() == 1 => {
      if let Expr::String(s) = &args[0] {
        if s.is_empty() {
          return Some(Ok(Expr::Integer(0)));
        }
        let count = s
          .split('/')
          .enumerate()
          .filter(|(i, part)| !(*i > 0 && part.is_empty()))
          .count() as i128;
        return Some(Ok(Expr::Integer(count)));
      }
      return Some(Ok(unevaluated("FileNameDepth", args)));
    }
    "ExpandFileName" if args.len() == 1 => {
      if let Expr::String(s) = &args[0] {
        // Windows has no HOME; fall back to USERPROFILE the way the rest
        // of the home-directory handling does (SetDirectory[],
        // $HomeDirectory, …).
        let expanded = if let Some(rest) = s.strip_prefix('~') {
          match std::env::var("HOME").or_else(|_| std::env::var("USERPROFILE"))
          {
            Ok(home) => format!("{home}{rest}"),
            Err(_) => s.clone(),
          }
        } else {
          s.clone()
        };
        let path = std::path::PathBuf::from(&expanded);
        let abs = if path.is_relative() {
          if let Ok(cwd) = std::env::current_dir() {
            cwd.join(&path)
          } else {
            path
          }
        } else {
          path
        };
        // Normalize path components (resolve . and ..)
        let mut components = Vec::new();
        for component in abs.components() {
          match component {
            std::path::Component::ParentDir => {
              components.pop();
            }
            std::path::Component::CurDir => {}
            _ => components.push(component),
          }
        }
        let normalized: std::path::PathBuf = components.iter().collect();
        return Some(Ok(Expr::String(
          normalized.to_string_lossy().into_owned(),
        )));
      }
      return Some(Ok(unevaluated("ExpandFileName", args)));
    }
    "URLParse" if args.len() == 1 || args.len() == 2 => {
      return Some(crate::functions::http_ast::url_parse_ast(args));
    }
    // URLBuild[<|"Scheme" -> …, "Domain" -> …, "Path" -> …, …|>] assembles the
    // parts of a URL back into one.
    "URLBuild"
      if args.len() == 1 && matches!(&args[0], Expr::Association(_)) =>
    {
      let Expr::Association(entries) = &args[0] else {
        unreachable!();
      };
      return Some(Ok(Expr::String(url_build_from_parts(entries))));
    }
    "URLBuild" if args.len() == 1 || args.len() == 2 => {
      // URLBuild["url"] => "url"
      // URLBuild[{"base", "path1", ...}] => "base/path1/..."
      // URLBuild[{"base", ...}, {"key" -> "val", ...}] => "base/...?key=val&..."
      let parts = match &args[0] {
        Expr::String(s) => vec![s.clone()],
        Expr::List(items) => {
          let mut strs = Vec::new();
          for item in items {
            match item {
              Expr::String(s) => strs.push(s.clone()),
              other => strs.push(crate::syntax::expr_to_string(other)),
            }
          }
          strs
        }
        _ => {
          return Some(Ok(unevaluated("URLBuild", args)));
        }
      };

      // Build base URL from parts
      let mut url = if parts.is_empty() {
        String::new()
      } else {
        let base = parts[0].trim_end_matches('/').to_string();
        let mut result = base;
        for part in &parts[1..] {
          let segment = part.trim_matches('/');
          if !segment.is_empty() {
            result.push('/');
            result.push_str(segment);
          }
        }
        result
      };

      // Add query parameters
      if args.len() == 2 {
        let query_pairs: Vec<(String, String)> = match &args[1] {
          Expr::List(items) => {
            let mut pairs = Vec::new();
            for item in items {
              match item {
                Expr::Rule {
                  pattern,
                  replacement,
                }
                | Expr::RuleDelayed {
                  pattern,
                  replacement,
                } => {
                  let key = match pattern.as_ref() {
                    Expr::String(s) => s.clone(),
                    other => crate::syntax::expr_to_string(other),
                  };
                  let val = match replacement.as_ref() {
                    Expr::String(s) => s.clone(),
                    other => crate::syntax::expr_to_string(other),
                  };
                  pairs.push((key, val));
                }
                _ => {}
              }
            }
            pairs
          }
          _ => vec![],
        };
        if !query_pairs.is_empty() {
          url.push('?');
          for (i, (key, val)) in query_pairs.iter().enumerate() {
            if i > 0 {
              url.push('&');
            }
            url.push_str(key);
            url.push('=');
            url.push_str(val);
          }
        }
      }

      return Some(Ok(Expr::String(url)));
    }
    // OpenRead[file] — open a file for reading, return InputStream[name, id]
    // OpenRead[file, BinaryFormat -> True] — same; binary mode is handled
    // by BinaryRead at read time, so the option is accepted as a pass-through.
    #[cfg(not(target_arch = "wasm32"))]
    "OpenRead" if (1..=2).contains(&args.len()) => {
      let (filename_arg, _opts) = io_split_filename_and_options(args);
      let filename = match filename_arg {
        Some(Expr::String(s)) => s.clone(),
        Some(other) => {
          return Some(Ok(call1("OpenRead", other.clone())));
        }
        None => {
          return Some(Ok(unevaluated("OpenRead", args)));
        }
      };
      // `"!command"` opens a pipe from an external command instead of a
      // file: the command runs through the shell and the stream reads its
      // standard output.
      let kind = if let Some(command) = command_file_spec(&filename) {
        let Ok(state) = spawn_command_stream(command) else {
          crate::emit_message_to_stdout(&format!(
            "OpenRead::noopen: Cannot open {filename}."
          ));
          return Some(Ok(Expr::Identifier("$Failed".to_string())));
        };
        StreamKind::Command(std::rc::Rc::new(RefCell::new(state)))
      } else {
        if !std::path::Path::new(&filename).exists() {
          crate::emit_message_to_stdout(&format!(
            "OpenRead::noopen: Cannot open {filename}."
          ));
          return Some(Ok(Expr::Identifier("$Failed".to_string())));
        }
        StreamKind::File(filename.clone())
      };
      let id = register_stream(filename.clone(), kind);
      return Some(Ok(call(
        "InputStream",
        vec![Expr::String(filename), Expr::Integer(id as i128)],
      )));
    }
    // OpenWrite[file] — open a file for writing, return OutputStream[name, id]
    // OpenWrite[BinaryFormat -> True] — same, options pass-through.
    #[cfg(not(target_arch = "wasm32"))]
    "OpenWrite" if args.len() <= 2 => {
      let (filename_arg, _opts) = io_split_filename_and_options(args);
      let filename = match filename_arg {
        Some(Expr::String(s)) => s.clone(),
        Some(other) => {
          return Some(Ok(call1("OpenWrite", other.clone())));
        }
        None => {
          let path = match crate::utils::create_file(None)
            .map_err(|e| InterpreterError::EvaluationError(e.to_string()))
          {
            Ok(v) => v,
            Err(e) => return Some(Err(e)),
          };
          path.to_string_lossy().into_owned()
        }
      };
      // `"!command"` opens a pipe *into* an external command: whatever is
      // written to the stream becomes the command's standard input.
      let kind = if let Some(command) = command_file_spec(&filename) {
        let Ok(state) = spawn_command_sink(command) else {
          crate::emit_message_to_stdout(&format!(
            "OpenWrite::noopen: Cannot open {filename}."
          ));
          return Some(Ok(Expr::Identifier("$Failed".to_string())));
        };
        StreamKind::CommandSink(std::rc::Rc::new(RefCell::new(state)))
      } else {
        // Create or truncate the file
        if let Err(e) = std::fs::File::create(&filename).map_err(|e| {
          InterpreterError::EvaluationError(format!(
            "OpenWrite: cannot open {filename}: {e}"
          ))
        }) {
          return Some(Err(e));
        }
        StreamKind::File(filename.clone())
      };
      let id = register_stream(filename.clone(), kind);
      return Some(Ok(call(
        "OutputStream",
        vec![Expr::String(filename), Expr::Integer(id as i128)],
      )));
    }
    // BinaryWrite[stream, bytes]              — write bytes (Integers in 0..255)
    // BinaryWrite[stream, bytes, type]        — write with explicit type spec
    // BinaryWrite[stream, bytes, {types…}]    — per-element types
    //
    // Returns the same `stream`. Supported types: "Byte" (Integer → 1 byte),
    // "Character8" (Integer or 1-char String → 1 byte). The 2-arg form
    // infers the type from the value (Integer → Byte, String → Character8).
    #[cfg(not(target_arch = "wasm32"))]
    "BinaryWrite" if (2..=3).contains(&args.len()) => {
      let Some(target) = io_write_target(&args[0]) else {
        return Some(Ok(unevaluated("BinaryWrite", args)));
      };
      // Render a single value at the given type into the byte buffer.
      // Returns false on an unsupported pairing so the caller can fall
      // back to the unevaluated form.
      fn write_value(out: &mut Vec<u8>, value: &Expr, ty: &str) -> bool {
        match (value, ty) {
          (Expr::Integer(n), "Byte" | "Character8") => {
            out.push((*n & 0xff) as u8);
            true
          }
          (Expr::String(s), "Byte" | "Character8") => {
            out.extend_from_slice(s.as_bytes());
            true
          }
          _ => false,
        }
      }
      let unevaluated = || unevaluated("BinaryWrite", args);
      let bytes: Vec<u8> = if args.len() == 3 {
        // Explicit type spec — either a single type string applied to
        // every value or a list of per-element types.
        let mut out: Vec<u8> = Vec::new();
        let values: Vec<&Expr> = match &args[1] {
          Expr::List(items) => items.iter().collect(),
          v => vec![v],
        };
        match &args[2] {
          Expr::String(ty) => {
            for v in &values {
              if !write_value(&mut out, v, ty) {
                return Some(Ok(unevaluated()));
              }
            }
          }
          Expr::List(types) => {
            if types.len() != values.len() {
              return Some(Ok(unevaluated()));
            }
            for (v, t) in values.iter().zip(types.iter()) {
              let Expr::String(ty) = t else {
                return Some(Ok(unevaluated()));
              };
              if !write_value(&mut out, v, ty) {
                return Some(Ok(unevaluated()));
              }
            }
          }
          _ => return Some(Ok(unevaluated())),
        }
        out
      } else {
        // 2-arg form: infer type from the value(s).
        match &args[1] {
          Expr::Integer(n) => vec![(*n & 0xff) as u8],
          Expr::String(s) => s.as_bytes().to_vec(),
          Expr::List(items) => {
            let mut out = Vec::with_capacity(items.len());
            for it in items {
              match it {
                Expr::Integer(n) => out.push((*n & 0xff) as u8),
                Expr::String(s) => out.extend_from_slice(s.as_bytes()),
                _ => return Some(Ok(unevaluated())),
              }
            }
            out
          }
          _ => return Some(Ok(unevaluated())),
        }
      };
      if let Err(e) = write_target_bytes(&target, &bytes, "BinaryWrite") {
        return Some(Err(e));
      }
      return Some(Ok(args[0].clone()));
    }
    // BinaryRead[stream]                — read 1 Byte (Integer 0..255)
    // BinaryRead[stream, "Byte"]        — read 1 Byte
    // BinaryRead[stream, "Character8"]  — read 1 Char (String of length 1)
    // BinaryRead[stream, {forms…}]      — read N items, returning a List
    //
    // Sequential calls advance the stream's read position so a chain like
    //   BinaryRead[s]; BinaryRead[s, {"Character8","Character8"}]
    // returns the next bytes after the first call rather than starting
    // from offset 0 every time.
    #[cfg(not(target_arch = "wasm32"))]
    "BinaryRead" if (1..=2).contains(&args.len()) => {
      let Some(path) = io_stream_path(&args[0]) else {
        return Some(Ok(unevaluated("BinaryRead", args)));
      };
      // The stream id lets consecutive reads advance the position; an
      // id-less stream always reads from offset 0.
      let stream_id = io_stream_id(&args[0]);
      let bytes = match io_binary_bytes(&args[0], &path) {
        Ok(b) => b,
        Err(e) => {
          return Some(Err(InterpreterError::EvaluationError(format!(
            "BinaryRead: cannot read {path}: {e}"
          ))));
        }
      };
      let start_pos = stream_id.and_then(get_stream_position).unwrap_or(0);
      let form = if args.len() == 2 {
        args[1].clone()
      } else {
        Expr::String("Byte".to_string())
      };
      // Render a single-byte form (Byte/Character8) at the given offset,
      // returning EndOfFile when out of range.
      let read_one = |form: &Expr, offset: usize| -> Option<Expr> {
        match form {
          Expr::String(s) if s == "Byte" => {
            if offset < bytes.len() {
              Some(Expr::Integer(bytes[offset] as i128))
            } else {
              Some(Expr::Identifier("EndOfFile".to_string()))
            }
          }
          Expr::String(s) if s == "Character8" => {
            if offset < bytes.len() {
              // Character8 is a raw byte rendered as a 1-char string;
              // values >127 use the Latin-1 mapping.
              let c = bytes[offset] as char;
              Some(Expr::String(c.to_string()))
            } else {
              Some(Expr::Identifier("EndOfFile".to_string()))
            }
          }
          _ => None,
        }
      };
      match &form {
        Expr::String(_) => {
          let Some(result) = read_one(&form, start_pos) else {
            return Some(Ok(unevaluated("BinaryRead", args)));
          };
          if let Some(id) = stream_id {
            let advance = usize::from(
              !matches!(&result, Expr::Identifier(s) if s == "EndOfFile"),
            );
            set_stream_position(id, start_pos + advance);
          }
          return Some(Ok(result));
        }
        Expr::List(items) => {
          let mut out = Vec::with_capacity(items.len());
          let mut offset = start_pos;
          for it in items {
            let Some(value) = read_one(it, offset) else {
              return Some(Ok(unevaluated("BinaryRead", args)));
            };
            if !matches!(&value, Expr::Identifier(s) if s == "EndOfFile") {
              offset += 1;
            }
            out.push(value);
          }
          if let Some(id) = stream_id {
            set_stream_position(id, offset);
          }
          return Some(Ok(Expr::List(out.into())));
        }
        _ => {
          return Some(Ok(unevaluated("BinaryRead", args)));
        }
      }
    }
    // BinaryReadList[file]            — read all bytes from `file`
    // BinaryReadList[file, "Byte"]    — same
    // BinaryReadList[stream]          — read remaining bytes from stream
    // BinaryReadList[stream, "Byte"]  — same
    //
    // Returns a List of Integers in 0..255. Returns {} on EOF.
    #[cfg(not(target_arch = "wasm32"))]
    "BinaryReadList" if (1..=2).contains(&args.len()) => {
      let path = match &args[0] {
        Expr::String(s) => s.clone(),
        _ => match io_stream_path(&args[0]) {
          Some(p) => p,
          None => {
            return Some(Ok(unevaluated("BinaryReadList", args)));
          }
        },
      };
      let form = if args.len() == 2 {
        args[1].clone()
      } else {
        Expr::String("Byte".to_string())
      };
      // Only "Byte" is supported; other forms fall through unevaluated so
      // callers see the same behaviour as for BinaryRead.
      match &form {
        Expr::String(s) if s == "Byte" => {}
        _ => {
          return Some(Ok(unevaluated("BinaryReadList", args)));
        }
      }
      let bytes = match io_binary_bytes(&args[0], &path) {
        Ok(b) => b,
        Err(e) => {
          return Some(Err(InterpreterError::EvaluationError(format!(
            "BinaryReadList: cannot read {path}: {e}"
          ))));
        }
      };
      let out: Vec<Expr> = bytes
        .into_iter()
        .map(|b| Expr::Integer(b as i128))
        .collect();
      return Some(Ok(Expr::List(out.into())));
    }
    // OpenAppend[file] — open a file for appending, return OutputStream[name, id]
    #[cfg(not(target_arch = "wasm32"))]
    "OpenAppend" if args.len() <= 1 => {
      let filename = if args.is_empty() {
        let path = match crate::utils::create_file(None)
          .map_err(|e| InterpreterError::EvaluationError(e.to_string()))
        {
          Ok(v) => v,
          Err(e) => return Some(Err(e)),
        };
        path.to_string_lossy().into_owned()
      } else {
        match &args[0] {
          Expr::String(s) => s.clone(),
          other => {
            return Some(Ok(call1("OpenAppend", other.clone())));
          }
        }
      };
      // A pipe has no contents to append to, so `"!command"` opens the same
      // stream `OpenWrite` would.
      let kind = if let Some(command) = command_file_spec(&filename) {
        let Ok(state) = spawn_command_sink(command) else {
          crate::emit_message_to_stdout(&format!(
            "OpenAppend::noopen: Cannot open {filename}."
          ));
          return Some(Ok(Expr::Identifier("$Failed".to_string())));
        };
        StreamKind::CommandSink(std::rc::Rc::new(RefCell::new(state)))
      } else {
        // Open for appending (create if not exists)
        if let Err(e) = std::fs::OpenOptions::new()
          .create(true)
          .append(true)
          .open(&filename)
          .map_err(|e| {
            InterpreterError::EvaluationError(format!(
              "OpenAppend: cannot open {filename}: {e}"
            ))
          })
        {
          return Some(Err(e));
        }
        StreamKind::File(filename.clone())
      };
      let id = register_stream(filename.clone(), kind);
      return Some(Ok(call(
        "OutputStream",
        vec![Expr::String(filename), Expr::Integer(id as i128)],
      )));
    }
    // StringToStream["text"] — create an input stream from a string
    "StringToStream" if args.len() == 1 => {
      let text = match &args[0] {
        Expr::String(s) => s.clone(),
        other => {
          return Some(Err(InterpreterError::EvaluationError(format!(
            "StringToStream: argument must be a string, got {}",
            crate::syntax::expr_to_string(other)
          ))));
        }
      };
      let id = register_stream("String".to_string(), StreamKind::Text(text));
      // Use Symbol `String` (not the string literal "String") so the
      // formatted form matches wolframscript: `InputStream[String, id]`.
      return Some(Ok(Expr::FunctionCall {
        name: "InputStream".to_string(),
        args: vec![
          Expr::Identifier("String".to_string()),
          Expr::Integer(id as i128),
        ]
        .into(),
      }));
    }
    // Close[stream] — close an open stream
    "Close" if args.len() == 1 => {
      // Extract stream ID from InputStream[name, id] or OutputStream[name, id]
      match &args[0] {
        Expr::FunctionCall {
          name: stream_head,
          args: stream_args,
        } if (stream_head == "InputStream"
          || stream_head == "OutputStream")
          && stream_args.len() == 2 =>
        {
          let id = match &stream_args[1] {
            Expr::Integer(n) => *n as usize,
            _ => {
              return Some(Ok(unevaluated("Close", args)));
            }
          };
          match close_stream(id) {
            // Close[FileStream] returns the file path as a String;
            // Close[StringToStream[…]] returns the symbol `String`.
            Some((_, StreamKind::Text(_))) => {
              return Some(Ok(Expr::Identifier("String".to_string())));
            }
            #[cfg(not(target_arch = "wasm32"))]
            Some((name, StreamKind::File(_))) => {
              return Some(Ok(Expr::String(name)));
            }
            // Close[OpenRead["!cmd"]] returns the `"!cmd"` spec, and stops
            // the command if it is still producing output.
            #[cfg(not(target_arch = "wasm32"))]
            Some((name, StreamKind::Command(state))) => {
              command_stream_close(&state);
              return Some(Ok(Expr::String(name)));
            }
            // Closing the write end is what tells the command its input has
            // ended, so it runs to completion here.
            #[cfg(not(target_arch = "wasm32"))]
            Some((name, StreamKind::CommandSink(state))) => {
              command_sink_close(&state);
              return Some(Ok(Expr::String(name)));
            }
            None => {
              let stream_str = crate::syntax::expr_to_string(&args[0]);
              crate::emit_message(&format!("{stream_str} is not open."));
              return Some(Ok(unevaluated("Close", args)));
            }
          }
        }
        Expr::String(s) => {
          crate::emit_message(&format!("{s} is not open."));
          return Some(Ok(unevaluated("Close", args)));
        }
        _ => {
          // Anything else is a type error — match wolframscript's message.
          let arg_str = crate::syntax::expr_to_string(&args[0]);
          crate::emit_message_to_stdout(&format!(
            "Close::stream: {arg_str} is not a string, SocketObject, InputStream[ ] or OutputStream[ ]."
          ));
          return Some(Ok(unevaluated("Close", args)));
        }
      }
    }
    // StreamPosition[stream] — get the current position of a stream
    "StreamPosition" if args.len() == 1 => {
      let stream = &args[0];
      match stream {
        Expr::FunctionCall {
          name: stream_head,
          args: stream_args,
        } if (stream_head == "InputStream"
          || stream_head == "OutputStream")
          && stream_args.len() == 2 =>
        {
          if let Expr::Integer(id) = &stream_args[1] {
            if let Some(pos) = get_stream_position(*id as usize) {
              return Some(Ok(Expr::Integer(pos as i128)));
            }
            let stream_str = crate::syntax::expr_to_string(stream);
            crate::emit_message(&format!(
              "StreamPosition::openx: {stream_str} is not open."
            ));
            return Some(Ok(unevaluated("StreamPosition", args)));
          }
          return Some(Ok(unevaluated("StreamPosition", args)));
        }
        Expr::String(s) => {
          crate::emit_message(&format!(
            "StreamPosition::openx: {s} is not open."
          ));
          return Some(Ok(unevaluated("StreamPosition", args)));
        }
        _ => {
          return Some(Ok(unevaluated("StreamPosition", args)));
        }
      }
    }
    // SetStreamPosition[stream, pos] — set the current position of a stream.
    // `pos` is either a non-negative integer (absolute byte offset) or
    // `Infinity` (seek to end of stream). Returns the new position.
    "SetStreamPosition" if args.len() == 2 => {
      let stream = &args[0];
      let is_infinity =
        matches!(&args[1], Expr::Identifier(s) if s == "Infinity");
      let pos_explicit = match &args[1] {
        Expr::Integer(n) => Some(*n as usize),
        _ if is_infinity => None,
        _ => {
          return Some(Ok(unevaluated("SetStreamPosition", args)));
        }
      };
      match stream {
        Expr::FunctionCall {
          name: stream_head,
          args: stream_args,
        } if (stream_head == "InputStream"
          || stream_head == "OutputStream")
          && stream_args.len() == 2 =>
        {
          if let Expr::Integer(id) = &stream_args[1] {
            let id_usize = *id as usize;
            if is_stream_open(id_usize) {
              let stream_len = get_stream_content(id_usize)
                .map_or(0, |(content, _)| content.len());
              let pos = match pos_explicit {
                Some(p) => {
                  if p > stream_len {
                    // wolframscript emits SetStreamPosition::stmrng
                    // and clamps the position to the end of stream.
                    let stream_str = crate::syntax::expr_to_string(stream);
                    crate::emit_message(&format!(
                      "SetStreamPosition::stmrng: Cannot set the current point in {stream_str} to position {p}; the requested position exceeds the length of the stream."
                    ));
                    stream_len
                  } else {
                    p
                  }
                }
                None => stream_len, // Infinity → end of stream
              };
              set_stream_position(id_usize, pos);
              return Some(Ok(Expr::Integer(pos as i128)));
            }
            let stream_str = crate::syntax::expr_to_string(stream);
            crate::emit_message(&format!(
              "SetStreamPosition::openx: {stream_str} is not open."
            ));
            return Some(Ok(unevaluated("SetStreamPosition", args)));
          }
          return Some(Ok(unevaluated("SetStreamPosition", args)));
        }
        _ => {
          return Some(Ok(unevaluated("SetStreamPosition", args)));
        }
      }
    }
    // ReadLine[stream] — read one line from a stream
    // ReadLine["file"] — read first line from a file
    #[cfg(not(target_arch = "wasm32"))]
    "ReadLine" if args.len() == 1 => {
      let (content, position, stream_id) = match &args[0] {
        // ReadLine["file"] — read the first line from a file directly;
        // ReadLine["!cmd"] — the first line of the command's output.
        Expr::String(path) => {
          let content = match command_file_spec(path) {
            Some(command) => run_command_capture(command),
            None => std::fs::read_to_string(path).ok(),
          };
          if let Some(content) = content {
            (content, 0usize, None)
          } else {
            crate::emit_message(&format!(
              "OpenRead::noopen: Cannot open {path}."
            ));
            return Some(Ok(Expr::Identifier("$Failed".to_string())));
          }
        }
        Expr::FunctionCall {
          name: stream_head,
          args: stream_args,
        } if stream_head == "InputStream" && stream_args.len() == 2 => {
          if let Expr::Integer(id) = &stream_args[1] {
            let id = *id as usize;
            match get_stream_line_content(id) {
              Some((content, pos)) => (content, pos, Some(id)),
              None => {
                return Some(Ok(Expr::Identifier("EndOfFile".to_string())));
              }
            }
          } else {
            return Some(Ok(unevaluated("ReadLine", args)));
          }
        }
        _ => {
          return Some(Ok(unevaluated("ReadLine", args)));
        }
      };

      let remaining = &content[position.min(content.len())..];
      if remaining.is_empty() {
        return Some(Ok(Expr::Identifier("EndOfFile".to_string())));
      }

      // Find end of line
      let (line, advance) = if let Some(idx) = remaining.find('\n') {
        (&remaining[..idx], idx + 1)
      } else {
        (remaining, remaining.len())
      };

      let result = Expr::String(line.to_string());

      // Advance position if it's a stream
      if let Some(id) = stream_id {
        advance_stream_position(id, position + advance);
      }

      return Some(Ok(result));
    }
    // Skip[stream, type] / Skip[stream, type, n] — read and discard `n`
    // (default 1) values of the given type, advancing the stream position.
    "Skip" if args.len() == 2 || args.len() == 3 => {
      let stream = &args[0];
      let stream_id = match stream {
        Expr::FunctionCall {
          name: stream_head,
          args: stream_args,
        } if (stream_head == "InputStream"
          || stream_head == "OutputStream")
          && stream_args.len() == 2 =>
        {
          if let Expr::Integer(id) = &stream_args[1] {
            Some(*id as usize)
          } else {
            None
          }
        }
        _ => None,
      };

      let count = if args.len() == 3 {
        match &args[2] {
          Expr::Integer(n) if *n >= 0 => *n as usize,
          _ => {
            return Some(Ok(unevaluated("Skip", args)));
          }
        }
      } else {
        1
      };

      if let Some(id) = stream_id
        && let Some((content, mut position)) = get_stream_content(id)
      {
        let mut hit_eof = false;
        for _ in 0..count {
          let remaining = &content[position.min(content.len())..];
          let (val, advance) = read_single_type(remaining, &args[1]);
          if matches!(&val, Expr::Identifier(s) if s == "EndOfFile") {
            hit_eof = true;
            position = content.len();
            break;
          }
          if advance == 0 {
            hit_eof = true;
            break;
          }
          position += advance;
        }
        advance_stream_position(id, position);
        return Some(Ok(Expr::Identifier(
          if hit_eof { "EndOfFile" } else { "Null" }.to_string(),
        )));
      }

      return Some(Ok(unevaluated("Skip", args)));
    }
    // Read[stream] or Read[stream, type] — read from a stream
    "Read" if !args.is_empty() && args.len() <= 2 => {
      let stream = &args[0];
      let stream_id = match stream {
        Expr::FunctionCall {
          name: stream_head,
          args: stream_args,
        } if (stream_head == "InputStream"
          || stream_head == "OutputStream")
          && stream_args.len() == 2 =>
        {
          if let Expr::Integer(id) = &stream_args[1] {
            Some(*id as usize)
          } else {
            None
          }
        }
        _ => None,
      };

      if let Some(id) = stream_id
        && let Some((content, position)) = get_stream_content(id)
      {
        let remaining = &content[position.min(content.len())..];

        // Determine the read type
        let read_type = if args.len() == 2 {
          &args[1]
        } else {
          &Expr::Identifier("Expression".to_string())
        };

        // Handle list of types: Read[stream, {type1, type2, ...}]
        if let Expr::List(types) = read_type {
          let mut results = Vec::new();
          let mut current_pos = position;
          for t in types {
            let rem = &content[current_pos.min(content.len())..];
            let (val, advance) = read_single_type(rem, t);
            current_pos += advance;
            results.push(val);
          }
          advance_stream_position(id, current_pos);
          return Some(Ok(Expr::List(results.into())));
        }

        let (result, advance) = read_single_type(remaining, read_type);
        advance_stream_position(id, position + advance);
        return Some(Ok(result));
      }

      return Some(Ok(unevaluated("Read", args)));
    }
    // Write[stream, expr1, expr2, ...] — write expressions to a stream in OutputForm
    #[cfg(not(target_arch = "wasm32"))]
    "Write" if args.len() >= 2 => {
      let Some(target) = io_write_target(&args[0]) else {
        return Some(Ok(unevaluated("Write", args)));
      };
      let mut content = String::new();
      for arg in &args[1..] {
        content.push_str(&crate::syntax::expr_to_string(arg));
      }
      content.push('\n');
      if let Err(e) = write_target_bytes(&target, content.as_bytes(), "Write") {
        return Some(Err(e));
      }
      return Some(Ok(Expr::Identifier("Null".to_string())));
    }
    // WriteString[stream, "text1", "text2", ...] — write strings to a stream
    #[cfg(not(target_arch = "wasm32"))]
    "WriteString" if args.len() >= 2 => {
      return Some(write_string_to_channel(args, "", "WriteString"));
    }
    // WriteLine[stream, "text"] — write a string plus a newline to a stream
    #[cfg(not(target_arch = "wasm32"))]
    "WriteLine" if args.len() == 2 => {
      return Some(write_string_to_channel(args, "\n", "WriteLine"));
    }
    // Save["filename", symbol] or Save["filename", {sym1, sym2, ...}]
    // Saves symbol definitions (OwnValues, DownValues, Attributes, Options) to a file
    #[cfg(not(target_arch = "wasm32"))]
    "Save" if args.len() == 2 => {
      let filename = match &args[0] {
        Expr::String(s) => s.clone(),
        _ => {
          return Some(Ok(unevaluated("Save", args)));
        }
      };

      // Collect symbol names from the second argument (held)
      let symbols: Vec<String> = match &args[1] {
        Expr::Identifier(s) => vec![s.clone()],
        Expr::String(s) => vec![s.clone()],
        Expr::List(items) => items
          .iter()
          .filter_map(|item| match item {
            Expr::Identifier(s) => Some(s.clone()),
            Expr::String(s) => Some(s.clone()),
            _ => None,
          })
          .collect(),
        _ => {
          return Some(Ok(unevaluated("Save", args)));
        }
      };

      // Collect all definition lines for all symbols
      let mut all_lines: Vec<String> = Vec::new();

      for sym in &symbols {
        let mut sym_lines: Vec<String> = Vec::new();

        // 1. Attributes (user-set only)
        let user_attrs =
          crate::FUNC_ATTRS.with(|m| m.borrow().get(sym).cloned());
        if let Some(attrs) = user_attrs
          && !attrs.is_empty()
        {
          sym_lines.push(format!(
            "Attributes[{}] = {{{}}}",
            sym,
            attrs.join(", ")
          ));
        }

        // 2. DownValues (function definitions). Includes literal-argument
        // memoizations (e.g. `f[1] = 42`), which live in MEMO_VALUES.
        let down_values = crate::down_values_with_memo(sym);
        if let Some(overloads) = down_values {
          for (params, conditions, defaults, heads, blank_types, body) in
            &overloads
          {
            // List-pattern params (`_lp{i}`) reconstruct to a surface `{…}`
            // pattern with the original element names, body, and `/;` guard.
            if let Some((pattern_args, display_body)) =
              crate::evaluator::assignment::reconstruct_list_downvalue(
                params,
                conditions,
                heads,
                blank_types,
                body,
              )
            {
              let params_str = pattern_args
                .iter()
                .map(crate::syntax::expr_to_string)
                .collect::<Vec<_>>()
                .join(", ");
              sym_lines.push(format!(
                "{}[{}] := {}",
                sym,
                params_str,
                crate::syntax::expr_to_string(&display_body)
              ));
              continue;
            }
            let params_str = params
              .iter()
              .enumerate()
              .map(|(i, p)| {
                // Check if this is a literal-dispatch parameter (_dvN with SameQ condition)
                if (p.starts_with("_dv") || p.starts_with("_lp"))
                  && let Some(Some(cond)) = conditions.get(i)
                  && let Expr::Comparison {
                    operands,
                    operators,
                  } = cond
                  && operators
                    .iter()
                    .any(|op| matches!(op, ComparisonOp::SameQ))
                  && operands.len() == 2
                {
                  // Literal value dispatch: use the value directly
                  return crate::syntax::expr_to_string(&operands[1]);
                }

                let head = heads.get(i).and_then(|h| h.as_ref());
                let default = defaults.get(i).and_then(|d| d.as_ref());
                let condition = conditions.get(i).and_then(|c| c.as_ref());

                let mut param_str = if let Some(h) = head {
                  format!("{p}_{h}")
                } else {
                  format!("{p}_")
                };

                if let Some(def) = default {
                  param_str = format!(
                    "{}:{}",
                    param_str,
                    crate::syntax::expr_to_string(def)
                  );
                }

                if let Some(cond) = condition {
                  param_str = format!(
                    "{} /; {}",
                    param_str,
                    crate::syntax::expr_to_string(cond)
                  );
                }

                param_str
              })
              .collect::<Vec<_>>()
              .join(", ");

            let body_str = crate::syntax::expr_to_string(body);

            // Use = for literal-dispatch (all params are _dvN), := otherwise
            let is_literal_dispatch = params
              .iter()
              .all(|p| p.starts_with("_dv") || p.starts_with("_lp"));
            let assign_op = if is_literal_dispatch { "=" } else { ":=" };

            sym_lines
              .push(format!("{sym}[{params_str}] {assign_op} {body_str}"));
          }
        }

        // 3. OwnValues (variable assignments)
        let own_value = crate::ENV.with(|e| {
          let env = e.borrow();
          env.get(sym).cloned()
        });
        if let Some(stored) = own_value {
          let val_str = match stored {
            crate::StoredValue::ExprVal(e) => crate::syntax::expr_to_string(&e),
            crate::StoredValue::Raw(val) => val,
            crate::StoredValue::Association(items) => {
              let parts: Vec<String> = items
                .iter()
                .map(|(k, v)| {
                  format!("{} -> {}", k, crate::syntax::expr_to_string(v))
                })
                .collect();
              format!("<|{}|>", parts.join(", "))
            }
          };
          sym_lines.push(format!("{sym} = {val_str}"));
        }

        // 4. Options
        let options =
          crate::FUNC_OPTIONS.with(|m| m.borrow().get(sym).cloned());
        if let Some(opts) = options
          && !opts.is_empty()
        {
          let opts_str = opts
            .iter()
            .map(crate::syntax::expr_to_string)
            .collect::<Vec<_>>()
            .join(", ");
          sym_lines.push(format!("Options[{sym}] = {{{opts_str}}}"));
        }

        all_lines.extend(sym_lines);
      }

      // Join definitions with "\n \n" separator and add trailing newline
      let content = if all_lines.is_empty() {
        "\n".to_string()
      } else {
        format!("{}\n", all_lines.join("\n \n"))
      };

      if filename == "stdout" {
        print!("{content}");
        crate::capture_stdout(content.trim_end());
      } else {
        match std::fs::write(&filename, &content) {
          Ok(()) => {}
          Err(_e) => {
            crate::emit_message(&format!(
              "Save::noopen: Cannot open {filename}."
            ));
            return Some(Ok(Expr::Identifier("$Failed".to_string())));
          }
        }
      }

      return Some(Ok(Expr::Identifier("Null".to_string())));
    }
    // FileNames[] — list all files in current directory
    // FileNames["pattern"] — list files matching pattern
    // FileNames["pattern", "dir"] — list files in dir matching pattern
    // FileNames["pattern", "dir", Infinity] — recursive search
    #[cfg(not(target_arch = "wasm32"))]
    "FileNames" if args.len() <= 3 => {
      let pattern = if args.is_empty() {
        "*".to_string()
      } else {
        match &args[0] {
          Expr::String(s) => s.clone(),
          _ => {
            return Some(Ok(unevaluated("FileNames", args)));
          }
        }
      };

      let dir = if args.len() >= 2 {
        match &args[1] {
          Expr::String(s) => s.clone(),
          Expr::List(dirs) => {
            // FileNames["pat", {"dir1", "dir2"}] — search multiple dirs
            let mut all_files = Vec::new();
            let recursive = args.len() >= 3
              && matches!(&args[2], Expr::Identifier(s) if s == "Infinity");
            for d in dirs {
              if let Expr::String(dir_str) = d {
                let mut files =
                  collect_file_names(&pattern, dir_str, recursive);
                all_files.append(&mut files);
              }
            }
            all_files.sort();
            return Some(Ok(Expr::List(
              all_files.into_iter().map(Expr::String).collect(),
            )));
          }
          _ => ".".to_string(),
        }
      } else {
        ".".to_string()
      };

      let recursive = args.len() >= 3
        && matches!(&args[2], Expr::Identifier(s) if s == "Infinity");

      let mut files = collect_file_names(&pattern, &dir, recursive);
      files.sort();
      return Some(Ok(Expr::List(
        files.into_iter().map(Expr::String).collect(),
      )));
    }
    // SetDirectory[] — with no arguments, set to $HomeDirectory.
    #[cfg(not(target_arch = "wasm32"))]
    "SetDirectory" if args.is_empty() => {
      let home = std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
        .unwrap_or_default();
      if home.is_empty() {
        return Some(Err(InterpreterError::EvaluationError(
          "SetDirectory: cannot determine home directory.".into(),
        )));
      }
      match crate::utils::canonicalize(&home) {
        Ok(canonical) if canonical.is_dir() => {
          let new_dir = canonical.to_string_lossy().into_owned();
          DIRECTORY_STACK.with(|s| s.borrow_mut().push(new_dir.clone()));
          return Some(Ok(Expr::String(new_dir)));
        }
        _ => {
          return Some(Err(InterpreterError::EvaluationError(
            "SetDirectory: home directory does not exist.".into(),
          )));
        }
      }
    }
    // SetDirectory["dir"] — push "dir" onto the virtual directory stack.
    // Does not mutate the process CWD; see the note on DIRECTORY_STACK.
    #[cfg(not(target_arch = "wasm32"))]
    "SetDirectory" if args.len() == 1 => {
      let dir = match &args[0] {
        Expr::String(s) => s.clone(),
        _ => {
          return Some(Ok(unevaluated("SetDirectory", args)));
        }
      };
      // Resolve the requested path against the current virtual directory so
      // that relative paths behave like the real Wolfram SetDirectory.
      let requested = std::path::Path::new(&dir);
      let resolved = if requested.is_absolute() {
        requested.to_path_buf()
      } else {
        std::path::PathBuf::from(virtual_current_dir()).join(requested)
      };
      // Canonicalize both to validate existence and normalize the result.
      match crate::utils::canonicalize(&resolved) {
        Ok(canonical) if canonical.is_dir() => {
          let new_dir = canonical.to_string_lossy().into_owned();
          DIRECTORY_STACK.with(|s| s.borrow_mut().push(new_dir.clone()));
          return Some(Ok(Expr::String(new_dir)));
        }
        Ok(_) => {
          return Some(Err(InterpreterError::EvaluationError(format!(
            "SetDirectory: {dir} is not a directory."
          ))));
        }
        Err(e) => {
          return Some(Err(InterpreterError::EvaluationError(format!(
            "SetDirectory: {e}"
          ))));
        }
      }
    }
    // ResetDirectory[] — pop the virtual directory stack and return the
    // restored directory (or the process CWD if the stack becomes empty).
    #[cfg(not(target_arch = "wasm32"))]
    "ResetDirectory" if args.is_empty() => {
      let popped = DIRECTORY_STACK.with(|s| s.borrow_mut().pop());
      match popped {
        Some(_) => {
          return Some(Ok(Expr::String(virtual_current_dir())));
        }
        None => {
          return Some(Err(InterpreterError::EvaluationError(
            "ResetDirectory: directory stack is empty.".into(),
          )));
        }
      }
    }
    // DirectoryStack[] — return the directory stack maintained by
    // SetDirectory/ResetDirectory. Fresh sessions report `{}`.
    #[cfg(not(target_arch = "wasm32"))]
    "DirectoryStack" if args.is_empty() => {
      let stack = DIRECTORY_STACK
        .with(|s| s.borrow().iter().cloned().collect::<Vec<_>>());
      return Some(Ok(Expr::List(
        stack.into_iter().map(Expr::String).collect(),
      )));
    }
    // FileFormat["name"] — return the format string for a file, or
    // emit `FileFormat::nffil` and `$Failed` when missing. Actual
    // format detection isn't implemented yet.
    #[cfg(not(target_arch = "wasm32"))]
    "FileFormat" if args.len() == 1 => {
      let Expr::String(name) = &args[0] else {
        return Some(Ok(unevaluated("FileFormat", args)));
      };
      if !std::path::Path::new(name).exists() {
        crate::emit_message(&format!(
          "FileFormat::nffil: File not found during FileFormat[{name}]."
        ));
        return Some(Ok(Expr::Identifier("$Failed".to_string())));
      }
      return Some(Ok(unevaluated("FileFormat", args)));
    }
    // FileDate["name"] / FileDate["name", "type"] — file timestamps.
    // Woxi doesn't implement the date lookup yet; for missing files it
    // still reproduces wolframscript's error path (`fdnfnd` message
    // and an unevaluated FileDate[…] result).
    #[cfg(not(target_arch = "wasm32"))]
    "FileDate" if args.len() == 1 || args.len() == 2 => {
      let Expr::String(name) = &args[0] else {
        return Some(Ok(unevaluated("FileDate", args)));
      };
      if !std::path::Path::new(name).exists() {
        crate::emit_message(&format!(
          "FileDate::fdnfnd: Directory or file \"{name}\" not found."
        ));
      }
      return Some(Ok(unevaluated("FileDate", args)));
    }
    // FileHash[file] / FileHash[file, type] / FileHash[file, type, format] —
    // the hash of the file's bytes, `MD5` as an Integer by default. Anything
    // that cannot be read emits `FileHash::noopen` and returns `$Failed`; a
    // type or format Hash does not know reports it the same way Hash does.
    #[cfg(not(target_arch = "wasm32"))]
    "FileHash" if (1..=3).contains(&args.len()) => {
      let name = match &args[0] {
        Expr::String(s) => s.clone(),
        // `File["…"]` names a file just as its path does.
        Expr::FunctionCall { name, args: inner }
          if name == "File"
            && inner.len() == 1
            && matches!(&inner[0], Expr::String(_)) =>
        {
          match &inner[0] {
            Expr::String(s) => s.clone(),
            _ => unreachable!(),
          }
        }
        _ => return Some(Ok(unevaluated("FileHash", args))),
      };
      let as_string = |i: usize, default: &str| match args.get(i) {
        None => Some(default.to_string()),
        Some(Expr::String(s)) => Some(s.clone()),
        Some(_) => None,
      };
      let (Some(hash_type), Some(format)) =
        (as_string(1, "MD5"), as_string(2, "Integer"))
      else {
        return Some(Ok(unevaluated("FileHash", args)));
      };
      // wolframscript reports the absolute path, so resolve relative paths
      // against the current working directory.
      let path = std::path::Path::new(&name);
      let Ok(data) = std::fs::read(path) else {
        let abs = if path.is_absolute() {
          name.clone()
        } else {
          std::env::current_dir().map_or_else(
            |_| name.clone(),
            |cwd| cwd.join(path).to_string_lossy().into_owned(),
          )
        };
        crate::emit_message(&format!("FileHash::noopen: Cannot open {abs}."));
        return Some(Ok(Expr::Identifier("$Failed".to_string())));
      };
      let Some(hex) =
        crate::functions::string_ast::hash_bytes(&data, &hash_type)
      else {
        crate::emit_message(&format!(
          "Hash::invhash: {hash_type} is not a valid Hash specification."
        ));
        return Some(Ok(Expr::Identifier("$Failed".to_string())));
      };
      return Some(Ok(
        if let Some(result) =
          crate::functions::string_ast::format_digest(&hex, &format)
        {
          result
        } else {
          crate::emit_message(&format!(
            "FileHash::uform: Invalid hash format {format}."
          ));
          unevaluated("FileHash", args)
        },
      ));
    }
    // FileSize["name"] — the size as Quantity[bytes, "Bytes"] with a Real
    // magnitude. Unlike FileByteCount, errors echo the call unevaluated:
    // ::fdnfnd for a missing path, ::fdir for a directory, ::badfile for a
    // non-string argument. A File["…"] wrapper is accepted.
    #[cfg(not(target_arch = "wasm32"))]
    "FileSize" if args.len() == 1 => {
      let unevaluated = || Some(Ok(unevaluated("FileSize", args)));
      let name = match &args[0] {
        Expr::String(s) => s.clone(),
        Expr::FunctionCall { name, args: fargs }
          if name == "File"
            && fargs.len() == 1
            && matches!(&fargs[0], Expr::String(_)) =>
        {
          match &fargs[0] {
            Expr::String(s) => s.clone(),
            _ => unreachable!(),
          }
        }
        other => {
          crate::emit_message(&format!(
            "FileSize::badfile: The specified argument, {}, should be a valid string or File object.",
            crate::syntax::expr_to_output(other)
          ));
          return unevaluated();
        }
      };
      match std::fs::metadata(&name) {
        Ok(meta) if meta.is_file() => {
          return Some(Ok(Expr::FunctionCall {
            name: "Quantity".to_string(),
            args: vec![
              Expr::Real(meta.len() as f64),
              Expr::String("Bytes".to_string()),
            ]
            .into(),
          }));
        }
        Ok(meta) if meta.is_dir() => {
          crate::emit_message(&format!(
            "FileSize::fdir: The specified path {name} refers to a directory; a file path was expected."
          ));
          return unevaluated();
        }
        _ => {
          crate::emit_message(&format!(
            "FileSize::fdnfnd: Directory or file \"{name}\" not found."
          ));
          return unevaluated();
        }
      }
    }
    // FileByteCount["name"] — size in bytes, or emit `fdnfnd` and
    // return `$Failed` when the file is missing.
    #[cfg(not(target_arch = "wasm32"))]
    "FileByteCount" if args.len() == 1 => {
      let Expr::String(name) = &args[0] else {
        return Some(Ok(unevaluated("FileByteCount", args)));
      };
      match std::fs::metadata(name) {
        Ok(meta) if meta.is_file() => {
          return Some(Ok(Expr::Integer(meta.len() as i128)));
        }
        _ => {
          crate::emit_message(&format!(
            "FileByteCount::fdnfnd: Directory or file \"{name}\" not found."
          ));
          return Some(Ok(Expr::Identifier("$Failed".to_string())));
        }
      }
    }
    // AbsoluteFileName["name"] — return the absolute path if the file
    // exists, otherwise emit `AbsoluteFileName::fdnfnd` and return
    // `$Failed` (matching wolframscript).
    #[cfg(not(target_arch = "wasm32"))]
    "AbsoluteFileName" if args.len() == 1 => {
      let Expr::String(name) = &args[0] else {
        return Some(Ok(unevaluated("AbsoluteFileName", args)));
      };
      if let Ok(p) = crate::utils::canonicalize(name) {
        return Some(Ok(Expr::String(p.to_string_lossy().into_owned())));
      }
      crate::emit_message(&format!(
        "AbsoluteFileName::fdnfnd: Directory or file \"{name}\" not found."
      ));
      return Some(Ok(Expr::Identifier("$Failed".to_string())));
    }
    // FindFile["name"] — return the absolute path if the file exists,
    // else `$Failed`. A context string like "MyPaclet`" resolves through
    // the loaded paclet directories and `$Path`, just as `Get` does.
    #[cfg(not(target_arch = "wasm32"))]
    "FindFile" if args.len() == 1 => {
      let Expr::String(name) = &args[0] else {
        return Some(Ok(unevaluated("FindFile", args)));
      };
      if name.ends_with('`') {
        return Some(Ok(
          match crate::functions::paclet::resolve_context(name) {
            Some(path) => Expr::String(path.to_string_lossy().into_owned()),
            None => Expr::Identifier("$Failed".to_string()),
          },
        ));
      }
      // Any other context-ish name can't be resolved to a file on disk.
      // Match wolframscript's `$Failed` return.
      if name.contains('`') {
        return Some(Ok(Expr::Identifier("$Failed".to_string())));
      }
      return Some(Ok(match crate::utils::canonicalize(name) {
        Ok(p) => Expr::String(p.to_string_lossy().into_owned()),
        Err(_) => Expr::Identifier("$Failed".to_string()),
      }));
    }
    // FileNameDrop["path", n] — drop n path components
    "FileNameDrop" if !args.is_empty() && args.len() <= 2 => {
      if let Expr::String(path) = &args[0] {
        let n = if args.len() == 2 {
          expr_to_i128(&args[1])?
        } else {
          -1 // default: drop last component
        };
        let sep = std::path::MAIN_SEPARATOR_STR;
        let parts: Vec<&str> = path.split(sep).collect();
        let total = parts.len() as i128;
        let result = if n >= 0 {
          // Drop first n components
          let skip = (n as usize).min(parts.len());
          parts[skip..].join(sep)
        } else {
          // Drop last |n| components
          let keep = (total + n).max(0) as usize;
          parts[..keep].join(sep)
        };
        return Some(Ok(Expr::String(result)));
      }
    }
    "FileNameTake" if !args.is_empty() && args.len() <= 2 => {
      if let Expr::String(path) = &args[0] {
        // Path components, matching FileNameSplit: split on '/', dropping
        // empty segments except a leading one (the absolute-root marker).
        let components: Vec<String> = path
          .split('/')
          .enumerate()
          .filter(|(i, part)| !(*i > 0 && part.is_empty()))
          .map(|(_, part)| part.to_string())
          .collect();
        let total = components.len() as i128;
        // Root-aware join: a slice consisting only of the leading "" marker
        // (or otherwise joining to nothing) is the absolute root "/".
        let join = |parts: &[String]| -> String {
          let joined = parts.join("/");
          if joined.is_empty() && !parts.is_empty() {
            "/".to_string()
          } else {
            joined
          }
        };
        // Resolve the take specification into a 0-indexed `[start, end)` range.
        let slice: Option<(usize, usize)> = match args.get(1) {
          // Default: just the last component.
          None => {
            if total == 0 {
              Some((0, 0))
            } else {
              Some(((total - 1) as usize, total as usize))
            }
          }
          Some(Expr::Integer(n)) => {
            if *n >= 0 {
              Some((0, (*n).clamp(0, total) as usize))
            } else {
              Some(((total + *n).max(0) as usize, total as usize))
            }
          }
          Some(Expr::List(range)) if range.len() == 2 => {
            if let (Expr::Integer(m), Expr::Integer(nn)) =
              (&range[0], &range[1])
            {
              let resolve =
                |idx: i128| if idx < 0 { total + idx } else { idx - 1 };
              let s = resolve(*m);
              let e = resolve(*nn);
              if s < 0 || e >= total || s > e {
                None
              } else {
                Some((s as usize, (e + 1) as usize))
              }
            } else {
              None
            }
          }
          _ => None,
        };
        if let Some((s, e)) = slice
          && s <= e
          && e <= components.len()
        {
          return Some(Ok(Expr::String(join(&components[s..e]))));
        }
      }
      return Some(Ok(unevaluated("FileNameTake", args)));
    }
    // Input[] / Input[prompt] / InputString[] / InputString[prompt] —
    // wolframscript prints the prompt to stdout (no trailing newline) and
    // then reads one line from stdin, returning `EndOfFile` once stdin is
    // exhausted. `InputString` yields the raw line; `Input` parses and
    // evaluates it as a Wolfram expression (blank line → Null, syntax error
    // → $Failed with the usual message), exactly like ToExpression.
    // Embedders that don't own stdin never get a line, so they keep the
    // non-interactive `EndOfFile` result.
    "Input" | "InputString" if args.len() <= 1 => {
      if let Some(arg) = args.first() {
        let prompt = match arg {
          Expr::String(p) => p.clone(),
          _ => crate::syntax::expr_to_string(arg),
        };
        if !crate::is_quiet_print() {
          use std::io::Write as _;
          print!("{prompt}");
          let _ = std::io::stdout().flush();
        }
        crate::capture_stdout_raw(&prompt);
      }
      let Some(line) = crate::read_stdin_line() else {
        return Some(Ok(Expr::Identifier("EndOfFile".to_string())));
      };
      if name == "InputString" {
        return Some(Ok(Expr::String(line)));
      }
      return Some(crate::functions::string_ast::to_expression_ast_as(
        &[Expr::String(line)],
        "Syntax",
      ));
    }
    _ => {}
  }
  None
}

/// Helper for Read: read a single value of a given type from remaining stream content.
/// Returns (result_expr, bytes_consumed).
fn read_single_type(remaining: &str, read_type: &Expr) -> (Expr, usize) {
  let type_name = match read_type {
    Expr::Identifier(s) => s.as_str(),
    _ => "Expression",
  };

  if remaining.is_empty() {
    return (Expr::Identifier("EndOfFile".to_string()), 0);
  }

  match type_name {
    "Word" => {
      // Skip leading whitespace
      let trimmed = remaining.trim_start();
      let skipped = remaining.len() - trimmed.len();
      if trimmed.is_empty() {
        return (Expr::Identifier("EndOfFile".to_string()), remaining.len());
      }
      // Read until whitespace
      let end = trimmed
        .find(|c: char| c.is_whitespace())
        .unwrap_or(trimmed.len());
      let word = &trimmed[..end];
      (Expr::String(word.to_string()), skipped + end)
    }
    "Number" => {
      // Skip leading whitespace
      let trimmed = remaining.trim_start();
      let skipped = remaining.len() - trimmed.len();
      if trimmed.is_empty() {
        return (Expr::Identifier("EndOfFile".to_string()), remaining.len());
      }
      // Try to parse a number
      let mut end = 0;
      let chars: Vec<char> = trimmed.chars().collect();
      // Optional sign
      if end < chars.len() && (chars[end] == '+' || chars[end] == '-') {
        end += 1;
      }
      // Digits before decimal
      let start_digits = end;
      while end < chars.len() && chars[end].is_ascii_digit() {
        end += 1;
      }
      let has_int_part = end > start_digits;
      // Decimal point and more digits
      let mut is_real = false;
      if end < chars.len() && chars[end] == '.' {
        is_real = true;
        end += 1;
        while end < chars.len() && chars[end].is_ascii_digit() {
          end += 1;
        }
      }
      if end == 0 || (!has_int_part && !is_real) {
        return (Expr::Identifier("$Failed".to_string()), skipped);
      }
      let num_str = &trimmed[..end];
      if is_real {
        if let Ok(f) = num_str.parse::<f64>() {
          return (Expr::Real(f), skipped + end);
        }
      } else if let Ok(n) = num_str.parse::<i128>() {
        return (Expr::Integer(n), skipped + end);
      }
      (Expr::Identifier("$Failed".to_string()), skipped)
    }
    "String" => {
      // Read until newline
      let end = remaining.find('\n').unwrap_or(remaining.len());
      let line = &remaining[..end];
      let advance = if end < remaining.len() { end + 1 } else { end };
      (Expr::String(line.to_string()), advance)
    }
    "Character" => {
      let ch = remaining.chars().next().unwrap();
      (Expr::String(ch.to_string()), ch.len_utf8())
    }
    // "Expression", and any unrecognized type, reads a whole expression.
    _ => {
      // Read the next top-level Wolfram expression. Strings, comments,
      // and bracketed groups are allowed to span newlines, so we scan
      // forward tracking quote/bracket depth instead of cutting at the
      // first newline — otherwise inputs like `"Tengo una\nvaca."`
      // would only read the unbalanced opener `"Tengo una`.
      let trimmed = remaining.trim_start();
      let skipped = remaining.len() - trimmed.len();
      if trimmed.is_empty() {
        return (Expr::Identifier("EndOfFile".to_string()), remaining.len());
      }
      let bytes = trimmed.as_bytes();
      let mut i = 0;
      let mut depth: i32 = 0;
      let mut in_string = false;
      while i < bytes.len() {
        let c = bytes[i];
        if in_string {
          if c == b'\\' && i + 1 < bytes.len() {
            // Escape: skip next byte too.
            i += 2;
            continue;
          }
          if c == b'"' {
            in_string = false;
          }
          i += 1;
          continue;
        }
        match c {
          b'"' => in_string = true,
          b'(' | b'[' | b'{' => depth += 1,
          b')' | b']' | b'}' => depth -= 1,
          b'\n' if depth == 0 => break,
          _ => {}
        }
        i += 1;
      }
      let end = i;
      let line = &trimmed[..end];
      let advance = if skipped + end < remaining.len() {
        skipped + end + 1
      } else {
        remaining.len()
      };
      match crate::interpret(line) {
        Ok(result_str) => {
          let expr = crate::syntax::string_to_expr(&result_str)
            .unwrap_or(Expr::Identifier(result_str));
          (expr, advance)
        }
        Err(_) => (Expr::Identifier("$Failed".to_string()), advance),
      }
    }
  }
}

/// Split an `OpenWrite[…]` / `OpenRead[…]` arg list into the optional
/// filename argument and the remaining option-Rule arguments. Used so the
/// `BinaryFormat -> True` option can be passed through alongside (or
/// instead of) a filename.
#[cfg(not(target_arch = "wasm32"))]
fn io_split_filename_and_options(args: &[Expr]) -> (Option<&Expr>, Vec<&Expr>) {
  let mut filename = None;
  let mut opts = Vec::new();
  for a in args {
    if matches!(a, Expr::Rule { .. } | Expr::RuleDelayed { .. }) {
      opts.push(a);
    } else if filename.is_none() {
      filename = Some(a);
    } else {
      opts.push(a);
    }
  }
  (filename, opts)
}

/// Extract the file path backing an `InputStream[name, id]` /
/// `OutputStream[name, id]` expression. Used by `BinaryWrite` /
/// `BinaryRead` to find the underlying file.
#[cfg(not(target_arch = "wasm32"))]
fn io_stream_path(expr: &Expr) -> Option<String> {
  let Expr::FunctionCall { name, args } = expr else {
    return None;
  };
  if name != "InputStream" && name != "OutputStream" {
    return None;
  }
  if args.is_empty() {
    return None;
  }
  match &args[0] {
    Expr::String(s) => Some(s.clone()),
    _ => None,
  }
}

/// Render a single delimited-table cell. Numeric/symbolic atoms are emitted
/// bare. When `quote_strings` is set (CSV/TSV) strings are wrapped in `"…"`
/// with embedded `"` doubled, matching wolframscript; the `"Table"` format
/// passes `false`, emitting strings verbatim.
fn csv_cell(expr: &Expr, quote_strings: bool) -> String {
  let quoted = |text: String| -> String {
    if quote_strings {
      format!("\"{}\"", text.replace('"', "\"\""))
    } else {
      text
    }
  };
  match expr {
    Expr::String(s) => quoted(s.clone()),
    // Machine numbers are written bare; a bigger integer is quoted, like
    // every other non-machine value.
    Expr::Integer(n) if i64::try_from(*n).is_ok() => n.to_string(),
    Expr::Real(_) => crate::syntax::expr_to_string(expr),
    // The booleans are lower-cased and left unquoted.
    Expr::Identifier(name) if name == "True" || name == "False" => {
      name.to_lowercase()
    }
    // A list is written out; any other compound expression has no CSV
    // representation and becomes the placeholder `-Head-`.
    Expr::List(_) => quoted(crate::syntax::expr_to_string(expr)),
    Expr::FunctionCall { .. }
    | Expr::BinaryOp { .. }
    | Expr::UnaryOp { .. }
    | Expr::Association(_)
    | Expr::Comparison { .. }
      if !matches!(expr, Expr::FunctionCall { name, .. } if name == "Rational") =>
    {
      let head =
        crate::functions::predicate_ast::head_ast(std::slice::from_ref(expr))
          .map_or_else(
            |_| "Expression".to_string(),
            |h| crate::syntax::expr_to_string(&h),
          );
      quoted(format!("-{head}-"))
    }
    // Symbols, rationals and the other atoms keep their text, quoted.
    _ => quoted(crate::syntax::expr_to_string(expr)),
  }
}

/// Serialize an expression to Wolfram's pretty-printed JSON (tab-indented,
/// `"key":value` with no space after the colon, `true`/`false`/`null`, empty
/// containers inline as `[]` / `{}`). `indent` is the tab depth of the value's
/// opening bracket. Returns `None` for any value JSON cannot represent, so the
/// caller leaves `ExportString` unevaluated.
fn export_string_json(
  expr: &Expr,
  indent: usize,
  compact: bool,
) -> Option<String> {
  // Format a Real as JSON: a finite decimal with at least one fractional
  // digit (3.0 -> "3.0", not Wolfram's bare "3.").
  fn real_json(f: f64) -> Option<String> {
    if !f.is_finite() {
      return None;
    }
    let s = format!("{f}");
    Some(if s.contains('.') || s.contains('e') || s.contains('E') {
      s
    } else {
      format!("{s}.0")
    })
  }
  fn escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    for c in s.chars() {
      match c {
        '"' => out.push_str("\\\""),
        '\\' => out.push_str("\\\\"),
        '\n' => out.push_str("\\n"),
        '\t' => out.push_str("\\t"),
        '\r' => out.push_str("\\r"),
        _ => out.push(c),
      }
    }
    out
  }
  match expr {
    Expr::Integer(n) => Some(n.to_string()),
    Expr::BigInteger(n) => Some(n.to_string()),
    Expr::Real(f) => real_json(*f),
    // JSON has no rationals, so they go out as their machine value.
    Expr::FunctionCall { name, args }
      if name == "Rational" && args.len() == 2 =>
    {
      match (&args[0], &args[1]) {
        (Expr::Integer(n), Expr::Integer(d)) if *d != 0 => {
          real_json(*n as f64 / *d as f64)
        }
        _ => None,
      }
    }
    Expr::String(s) => Some(format!("\"{}\"", escape(s))),
    Expr::Identifier(s) if s == "True" => Some("true".to_string()),
    Expr::Identifier(s) if s == "False" => Some("false".to_string()),
    Expr::Identifier(s) if s == "Null" || s == "None" => {
      Some("null".to_string())
    }
    Expr::List(items) => {
      if items.is_empty() {
        return Some("[]".to_string());
      }
      let mut parts = Vec::with_capacity(items.len());
      for it in items {
        parts.push(export_string_json(it, indent + 1, compact)?);
      }
      if compact {
        return Some(format!("[{}]", parts.join(",")));
      }
      let inner = "\t".repeat(indent + 1);
      let body: Vec<String> =
        parts.iter().map(|p| format!("{inner}{p}")).collect();
      Some(format!("[\n{}\n{}]", body.join(",\n"), "\t".repeat(indent)))
    }
    Expr::Association(pairs) => {
      if pairs.is_empty() {
        return Some("{}".to_string());
      }
      let mut parts = Vec::with_capacity(pairs.len());
      for (k, v) in pairs {
        // Only string keys exist in JSON; anything else fails the export
        // rather than being stringified into a key that was never there.
        let Expr::String(key) = k else {
          crate::emit_message(
            "Export::jsonassockeynstr: Association contains a non-string key.",
          );
          return None;
        };
        let key = key.clone();
        parts.push(format!(
          "\"{}\":{}",
          escape(&key),
          export_string_json(v, indent + 1, compact)?
        ));
      }
      if compact {
        return Some(format!("{{{}}}", parts.join(",")));
      }
      let inner = "\t".repeat(indent + 1);
      let body: Vec<String> =
        parts.iter().map(|p| format!("{inner}{p}")).collect();
      Some(format!(
        "{{\n{}\n{}}}",
        body.join(",\n"),
        "\t".repeat(indent)
      ))
    }
    other => {
      crate::emit_message(&format!(
        "Export::jsonstrictencoding: Expression {} cannot be exported as JSON.",
        crate::syntax::format_expr(other, crate::syntax::ExprForm::Output)
      ));
      None
    }
  }
}

/// Serialize an expression to CSV (or TSV when `sep` is `\t`).
/// A list-of-lists is rendered one row per inner list; a flat list is
/// rendered one element per row. Other expressions become a single row.
/// Each row is terminated with a newline (Wolfram's `ExportString` always
/// emits a trailing newline after the last record).
/// Assemble the `URLParse` parts of an association back into a URL.
///
/// A `Domain` (or an absolute `Path` under a scheme, as in `file:///tmp/f`)
/// introduces the `//` authority marker; without one the scheme is followed
/// directly by the path, as in `mailto:a@b.c`. A `Path` given as a list is
/// joined with `/`, so a leading `""` is what makes it absolute.
fn url_build_from_parts(entries: &[(Expr, Expr)]) -> String {
  let lookup = |key: &str| -> Option<&Expr> {
    entries
      .iter()
      .find(|(k, _)| matches!(k, Expr::String(s) if s == key))
      .map(|(_, v)| v)
      .filter(|v| !matches!(v, Expr::Identifier(s) if s == "None"))
  };
  let text = |e: &Expr| match e {
    Expr::String(s) => s.clone(),
    other => crate::syntax::expr_to_string(other),
  };

  let scheme = lookup("Scheme").map(&text);
  let user = lookup("User").map(&text);
  let domain = lookup("Domain").map(&text);
  let port = lookup("Port").map(&text);
  let path = lookup("Path").map(|p| match p {
    Expr::List(segments) => {
      segments.iter().map(&text).collect::<Vec<_>>().join("/")
    }
    other => text(other),
  });
  let fragment = lookup("Fragment").map(&text);
  let query: Vec<(String, String)> = match lookup("Query") {
    Some(Expr::List(items)) => items
      .iter()
      .filter_map(|item| match item {
        Expr::Rule {
          pattern,
          replacement,
        }
        | Expr::RuleDelayed {
          pattern,
          replacement,
        } => Some((text(pattern), text(replacement))),
        _ => None,
      })
      .collect(),
    Some(Expr::Association(pairs)) => {
      pairs.iter().map(|(k, v)| (text(k), text(v))).collect()
    }
    _ => Vec::new(),
  };

  let mut url = String::new();
  if let Some(scheme) = &scheme {
    url.push_str(scheme);
    url.push(':');
  }
  let absolute_path = path.as_deref().is_some_and(|p| p.starts_with('/'));
  if domain.is_some() || (scheme.is_some() && absolute_path) {
    url.push_str("//");
    if let Some(user) = &user {
      url.push_str(user);
      url.push('@');
    }
    if let Some(domain) = &domain {
      url.push_str(domain);
    }
    if let Some(port) = &port {
      url.push(':');
      url.push_str(port);
    }
  }
  if let Some(path) = &path {
    url.push_str(path);
  }
  if !query.is_empty() {
    url.push('?');
    let encoded: Vec<String> = query
      .iter()
      .map(|(k, v)| {
        format!(
          "{}={}",
          crate::functions::string_ast::url_query_component(k),
          crate::functions::string_ast::url_query_component(v)
        )
      })
      .collect();
    url.push_str(&encoded.join("&"));
  }
  if let Some(fragment) = &fragment {
    url.push('#');
    url.push_str(fragment);
  }
  url
}

fn export_string_csv(
  expr: &Expr,
  sep: char,
  quote_strings: bool,
  trailing_newline: bool,
) -> String {
  let cell = |e: &Expr| csv_cell(e, quote_strings);
  let row_strs = |row: &Expr| -> String {
    if let Expr::List(items) = row {
      items
        .iter()
        .map(&cell)
        .collect::<Vec<_>>()
        .join(&sep.to_string())
    } else {
      cell(row)
    }
  };
  let rows: Vec<String> = match expr {
    Expr::List(items) => {
      let any_nested = items.iter().any(|e| matches!(e, Expr::List(_)));
      if any_nested {
        items.iter().map(row_strs).collect()
      } else {
        items.iter().map(&cell).collect()
      }
    }
    _ => vec![cell(expr)],
  };
  let mut out = rows.join("\n");
  if trailing_newline {
    out.push('\n');
  }
  out
}

/// Collect file names matching a glob pattern in a directory.
#[cfg(not(target_arch = "wasm32"))]
fn collect_file_names(
  pattern: &str,
  dir: &str,
  recursive: bool,
) -> Vec<String> {
  use std::path::Path;

  let dir_path = Path::new(dir);
  if !dir_path.is_dir() {
    return Vec::new();
  }

  let mut results = Vec::new();
  collect_files_recursive(dir_path, dir, pattern, recursive, &mut results);
  results
}

#[cfg(not(target_arch = "wasm32"))]
fn collect_files_recursive(
  path: &std::path::Path,
  base_dir: &str,
  pattern: &str,
  recursive: bool,
  results: &mut Vec<String>,
) {
  let Ok(entries) = std::fs::read_dir(path) else {
    return;
  };

  for entry in entries.flatten() {
    let file_name = entry.file_name().to_string_lossy().to_string();
    let file_type = entry.file_type();

    if let Ok(ft) = file_type {
      if glob_match(pattern, &file_name) {
        if base_dir == "." {
          results.push(file_name.clone());
        } else {
          let rel = entry.path();
          let rel_str = rel.to_string_lossy().to_string();
          results.push(rel_str);
        }
      }
      if ft.is_dir() && recursive {
        collect_files_recursive(
          &entry.path(),
          base_dir,
          pattern,
          true,
          results,
        );
      }
    }
  }
}

/// Simple glob pattern matching supporting * and ?
#[cfg(not(target_arch = "wasm32"))]
fn glob_match(pattern: &str, text: &str) -> bool {
  let p: Vec<char> = pattern.chars().collect();
  let t: Vec<char> = text.chars().collect();
  glob_match_impl(&p, &t)
}

#[cfg(not(target_arch = "wasm32"))]
fn glob_match_impl(pattern: &[char], text: &[char]) -> bool {
  if pattern.is_empty() {
    return text.is_empty();
  }
  if pattern[0] == '*' {
    // Try matching * with 0 or more characters
    for i in 0..=text.len() {
      if glob_match_impl(&pattern[1..], &text[i..]) {
        return true;
      }
    }
    false
  } else if text.is_empty() {
    false
  } else if pattern[0] == '?' || pattern[0] == text[0] {
    glob_match_impl(&pattern[1..], &text[1..])
  } else {
    false
  }
}

/// Render a text-mode SVG fallback for non-graphics expressions.
fn expr_text_svg(expr: &Expr) -> String {
  let boxes = super::complex_and_special::expr_to_box_form(expr);
  // The box form of a machine/arbitrary-precision Real carries a trailing
  // backtick precision marker (`2.``), matching wolframscript's `MakeBoxes`.
  // A typeset SVG display (which `Export[…, "SVG"]` emulates) suppresses the
  // marker and shows the notebook OutputForm — machine reals at 6 significant
  // figures — so strip it before layout, exactly as `generate_output_svg` does
  // for the Playground/Studio result SVG.
  let boxes = crate::strip_number_precision_markers(&boxes);
  boxes_to_text_svg(&boxes)
}

/// Render box-form expressions to a text SVG.
fn boxes_to_text_svg(boxes: &Expr) -> String {
  let layout = crate::functions::graphics::layout_box(boxes, 14.0);
  crate::functions::graphics::layout_to_svg(&layout, "currentColor")
}

/// Convert an expression to its SVG string representation.
/// True if `expr` is a graphics-like value that `expr_to_svg` will render
/// to a non-trivial SVG. Used to decide whether a `List` passed to
/// `Export[..., "gif"]` should be treated as an animated frame sequence.
#[cfg(not(target_arch = "wasm32"))]
fn is_rasterizable_frame(expr: &Expr) -> bool {
  matches!(expr, Expr::Graphics { .. } | Expr::Image { .. })
    || matches!(
      expr,
      Expr::FunctionCall { name, args }
        if (name == "Graphics" || name == "Graphics3D") && !args.is_empty()
    )
}

/// True if `expr` is an already-rendered graphic that can be drawn as a
/// cell inside a list/grid combination (see the `Expr::List` arms of
/// `expr_to_svg`).
fn is_inline_graphic(expr: &Expr) -> bool {
  match expr {
    Expr::Graphics { .. } | Expr::Image { .. } => true,
    Expr::FunctionCall { name, args }
      if (name == "Legended" || name == "Pane") && !args.is_empty() =>
    {
      is_inline_graphic(&args[0])
    }
    _ => false,
  }
}

/// The layout wrappers whose *cells* `expr_to_svg` composes: each one
/// arranges its parts in a grid, so a picture anywhere inside has to be
/// drawn rather than written out as source.
const LAYOUT_HEADS: &[&str] = &["Grid", "Column", "Row", "Labeled"];

/// True if `expr` is a picture, or a layout or display wrapper — `Grid`,
/// `Column`, `Row`, `Labeled`, `Pane`, `LocatorPane`, `Dynamic` — that has
/// one somewhere inside. Only then does composing it through
/// [`expr_to_svg`] beat the typeset-text fallback: a `Grid` of plain
/// strings already lays out correctly as text. Kept apart from the
/// "produces a graphic" head test so nothing tries to give a wrapper an
/// `ImageSize` of its own.
pub(crate) fn lays_out_a_graphic(expr: &Expr) -> bool {
  if is_inline_graphic(expr) {
    return true;
  }
  // A held `Graphics[…]` / `Graphics3D[…]` call is a picture too —
  // `expr_to_svg` renders one — and that is the form the args of a
  // display wrapper like `Labeled` arrive in.
  if matches!(expr, Expr::FunctionCall { name, args }
    if (name == "Graphics" || name == "Graphics3D") && !args.is_empty())
  {
    return true;
  }
  match expr {
    Expr::List(items) => items.iter().any(lays_out_a_graphic),
    // `Pane` and `Deploy` are transparent here: their own arms export what
    // they wrap. A `Framed` picture is drawn and framed by its own
    // renderer, so it counts as a picture too.
    Expr::FunctionCall { name, args }
      if LAYOUT_HEADS.contains(&name.as_str())
        || name == "Pane"
        || name == "Deploy"
        || name == "Framed" =>
    {
      args.iter().any(lays_out_a_graphic)
    }
    // `LocatorPane[locators, body, …]` shows `body`; its own arm draws the
    // locators onto it.
    Expr::FunctionCall { name, args }
      if name == "LocatorPane" && args.len() >= 2 =>
    {
      lays_out_a_graphic(&args[1])
    }
    // `Item[expr, opts…]` is a layout cell: what it displays is `expr`.
    Expr::FunctionCall { name, args } if name == "Item" && !args.is_empty() => {
      lays_out_a_graphic(&args[0])
    }
    // `ClickPane[expr, …]` shows `expr`; clicking it is the affordance the
    // handler is for, and contributes nothing to the picture.
    Expr::FunctionCall { name, args }
      if name == "ClickPane" && args.len() >= 2 =>
    {
      lays_out_a_graphic(&args[0])
    }
    // `Dynamic[graphic]` is a picture too — its arm shows what it holds.
    Expr::FunctionCall { name, args }
      if name == "Dynamic" && args.len() == 1 =>
    {
      lays_out_a_graphic(&args[0])
    }
    _ => false,
  }
}

/// The primitives a locator pane draws at one locator position.
///
/// `Appearance -> {g1, g2, …}` gives one picture per locator, each drawn
/// about the origin, so the marker is that picture translated onto the
/// point — this is how a Demonstration labels the vertices of a shape it
/// lets you drag. A single picture is used for every locator, `None` draws
/// nothing, and with no `Appearance` at all the Wolfram default is a small
/// circle with a crosshair through it, sized here as a fraction of the plot
/// range so it keeps its proportions whatever the scale.
fn locator_marker(
  appearance: Option<&Expr>,
  index: usize,
  point: &Expr,
  span: f64,
) -> Option<Expr> {
  let translate =
    |prims: &Expr| call("Translate", vec![prims.clone(), point.clone()]);
  // The primitives of an `Appearance` picture: `Graphics[prims, opts…]`
  // contributes `prims`; anything else is drawn as given.
  let primitives = |g: &Expr| match g {
    Expr::FunctionCall { name, args }
      if name == "Graphics" && !args.is_empty() =>
    {
      args[0].clone()
    }
    other => other.clone(),
  };
  match appearance {
    Some(Expr::Identifier(n)) if n == "None" => None,
    Some(Expr::List(items)) if !items.is_empty() => {
      Some(translate(&primitives(&items[index % items.len()])))
    }
    Some(single) => Some(translate(&primitives(single))),
    None => {
      let r = span * 0.015;
      let arm = span * 0.03;
      let offset = |dx: f64, dy: f64| Expr::List(vec![num(dx), num(dy)].into());
      let line =
        |a: Expr, b: Expr| call("Line", vec![Expr::List(vec![a, b].into())]);
      Some(translate(&Expr::List(
        vec![
          call("Circle", vec![offset(0.0, 0.0), num(r)]),
          line(offset(-arm, 0.0), offset(arm, 0.0)),
          line(offset(0.0, -arm), offset(0.0, arm)),
        ]
        .into(),
      )))
    }
  }
}

/// A machine real as an expression, for the synthesized marker geometry.
fn num(v: f64) -> Expr {
  Expr::Real(v)
}

/// Where a locator pane's locators currently are. `Dynamic[sym]` reads
/// `sym`, and so does the two-argument `Dynamic[sym, setter]` — the setter
/// is what runs when a locator is dragged and says nothing about where it
/// is now. A Demonstration often writes the whole thing as
/// `pt = …; Dynamic[pt, setter]`, computing the position from its sliders
/// first, so a leading statement is evaluated for that effect before the
/// last one is read.
fn locator_positions(arg: &Expr) -> Option<Expr> {
  match arg {
    Expr::CompoundExpr(items) => {
      let (last, leading) = items.split_last()?;
      for item in leading {
        let _ = crate::evaluator::evaluate_expr_to_expr(item);
      }
      locator_positions(last)
    }
    Expr::FunctionCall { name, args }
      if name == "Dynamic" && !args.is_empty() =>
    {
      crate::evaluator::evaluate_expr_to_expr(&args[0]).ok()
    }
    other => crate::evaluator::evaluate_expr_to_expr(other).ok(),
  }
}

/// `LocatorPane[locators, body, …]` as the picture it displays: the body
/// graphic with a marker drawn at every locator. Dragging a locator is a
/// front-end affordance; what the picture itself carries is the markers, so
/// this is what exporting or laying one out has to produce instead of the
/// call written out as source. `None` when the arguments don't fit.
fn locator_pane_graphic(args: &[Expr]) -> Option<Expr> {
  if args.len() < 2 {
    return None;
  }
  let positions = locator_positions(&args[0])?;
  let is_point = |e: &Expr| {
    matches!(e, Expr::List(c) if c.len() == 2
      && c.iter().all(|v| crate::functions::math_ast::try_eval_to_f64(v).is_some()))
  };
  let points: Vec<Expr> = match &positions {
    p if is_point(p) => vec![p.clone()],
    Expr::List(items) if items.iter().all(is_point) => items.to_vec(),
    _ => return None,
  };

  // The body must be a graphic; its options are kept so the pane inherits
  // the plot range, axes and grid lines the body asked for. A pane whose
  // picture depends on the locators is written `Dynamic@Graphics[…]`, and
  // `Dynamic` is HoldFirst, so what it holds is released here.
  let held = match &args[1] {
    Expr::FunctionCall { name, args: inner }
      if name == "Dynamic" && !inner.is_empty() =>
    {
      &inner[0]
    }
    other => other,
  };
  let body = crate::evaluator::evaluate_expr_to_expr(held).ok()?;
  let (body_prims, body_opts) = match &body {
    Expr::FunctionCall { name, args }
      if name == "Graphics" && !args.is_empty() =>
    {
      (args[0].clone(), args[1..].to_vec())
    }
    _ => return None,
  };

  // The default marker is sized against the plot range, so it stays a small
  // fraction of the picture whatever the coordinates are.
  let span = body_opts
    .iter()
    .find_map(|o| match o {
      Expr::Rule { pattern, replacement }
        if matches!(pattern.as_ref(), Expr::Identifier(p) if p == "PlotRange") =>
      {
        match replacement.as_ref() {
          Expr::List(axes) if axes.len() == 2 => {
            let extent = |e: &Expr| match e {
              Expr::List(b) if b.len() == 2 => {
                let f = crate::functions::math_ast::try_eval_to_f64;
                Some((f(&b[1])? - f(&b[0])?).abs())
              }
              _ => None,
            };
            Some(extent(&axes[0])?.max(extent(&axes[1])?))
          }
          _ => None,
        }
      }
      _ => None,
    })
    .unwrap_or(1.0);

  let appearance = args[2..].iter().find_map(|o| match o {
    Expr::Rule { pattern, replacement }
      if matches!(pattern.as_ref(), Expr::Identifier(p) if p == "Appearance") =>
    {
      crate::evaluator::evaluate_expr_to_expr(replacement).ok()
    }
    _ => None,
  });

  let mut prims = vec![body_prims];
  for (i, point) in points.iter().enumerate() {
    if let Some(marker) = locator_marker(appearance.as_ref(), i, point, span) {
      prims.push(marker);
    }
  }
  let mut graphics_args = vec![Expr::List(prims.into())];
  graphics_args.extend(body_opts);
  Some(call("Graphics", graphics_args))
}

/// A string shown through `Text` loses the quotation marks its own
/// OutputForm carries — that is what `Text` asks for. Only a string that
/// would be typeset on its own is affected (directly, or as the content of
/// a `Style`); the parts of a `Row` already render as text.
fn unquoted_display_string(expr: &Expr) -> Expr {
  match expr {
    Expr::String(s) => Expr::Identifier(s.clone()),
    Expr::FunctionCall { name, args }
      if name == "Style" && !args.is_empty() =>
    {
      let mut args = args.to_vec();
      args[0] = unquoted_display_string(&args[0]);
      Expr::FunctionCall {
        name: name.clone(),
        args: args.into(),
      }
    }
    other => other.clone(),
  }
}

/// The rows of cells a layout wrapper contributes to a combined grid.
/// `Grid` keeps its own row structure, `Column` stacks, `Row` spreads;
/// rows with no cells (a `{}` spacer row in a `Grid`) are dropped, since
/// the grid engine has nothing to place for them.
fn layout_rows(name: &str, args: &[Expr]) -> Vec<Vec<Expr>> {
  let items: &[Expr] = match args.first() {
    Some(Expr::List(items)) => items,
    _ => return vec![],
  };
  match name {
    "Grid" => items
      .iter()
      .map(|row| match row {
        Expr::List(cells) => cells.to_vec(),
        other => vec![other.clone()],
      })
      .filter(|cells| !cells.is_empty())
      .collect(),
    "Column" => items.iter().map(|c| vec![c.clone()]).collect(),
    _ => vec![items.to_vec()],
  }
}

/// True if the expression is (or, within nested lists, contains) a `Framed`
/// or `Highlighted` display wrapper.
fn contains_framed_or_highlighted(expr: &Expr) -> bool {
  match expr {
    Expr::FunctionCall { name, .. } => {
      name == "Framed" || name == "Highlighted"
    }
    Expr::List(items) => items.iter().any(contains_framed_or_highlighted),
    _ => false,
  }
}

/// The composed SVG of `Labeled[graphic, label, pos…]` — the label set
/// beside the picture. `None` when the content is not a picture, so a
/// label on plain text is left to the text renderer. (`Grid`, `Column`
/// and `Row` reach the visual pipeline through passes of their own.)
pub(crate) fn labeled_display_svg(expr: &Expr) -> Option<String> {
  let is_labeled = matches!(expr, Expr::FunctionCall { name, args }
    if name == "Labeled" && args.len() >= 2 && lays_out_a_graphic(&args[0]));
  is_labeled.then(|| expr_to_svg(expr))
}

/// The picture behind a wrapper that only shows one once its content
/// evaluates. `None` when it does not, which leaves the call to the arms
/// below — a `Dynamic` whose body errors still prints as itself.
fn evaluated_wrapper_svg(expr: &Expr) -> Option<String> {
  let Expr::FunctionCall { name, args } = expr else {
    return None;
  };
  match name.as_str() {
    // `Dynamic[expr]` displays the value `expr` has now — a front end
    // shows the content, never the wrapper. (Script-mode *text* output
    // keeps the wrapper, matching wolframscript; this is the picture path,
    // which is what a notebook shows.)
    "Dynamic" if args.len() == 1 => {
      let value = crate::evaluator::evaluate_expr_to_expr(&args[0]).ok()?;
      Some(expr_to_svg(&unquoted_display_string(&value)))
    }
    // `LocatorPane[locators, body, …]` displays `body` with a marker on
    // every locator.
    "LocatorPane" if args.len() >= 2 => {
      let drawn = locator_pane_graphic(args)?;
      Some(expr_to_svg(
        &crate::evaluator::evaluate_expr_to_expr(&drawn).unwrap_or(drawn),
      ))
    }
    _ => None,
  }
}

pub(crate) fn expr_to_svg(expr: &Expr) -> String {
  if let Some(svg) = evaluated_wrapper_svg(expr) {
    return svg;
  }
  match expr {
    Expr::Graphics { svg: svg_data, .. } => svg_data.clone(),
    // `Pane[content, opts…]` only constrains its content's size; the
    // FrontEnd displays the content, so exporting one exports what it
    // wraps. (The notebook display pipeline already unwraps it — without
    // this, `Export[…, Pane[graphic]]` wrote the expression as text.)
    // `Deploy` is the same: it only makes its content non-selectable.
    Expr::FunctionCall { name, args }
      if (name == "Pane" || name == "Deploy") && !args.is_empty() =>
    {
      expr_to_svg(&args[0])
    }
    // `Item[expr, opts…]` is a layout cell; the options place it and what
    // it displays is `expr`.
    Expr::FunctionCall { name, args } if name == "Item" && !args.is_empty() => {
      expr_to_svg(&args[0])
    }
    // `ClickPane[expr, …]` displays `expr`. What it adds is a click
    // handler, which the picture itself does not carry — this is how a
    // Demonstration lets you draw on a grid.
    Expr::FunctionCall { name, args }
      if name == "ClickPane" && args.len() >= 2 =>
    {
      expr_to_svg(
        &crate::evaluator::evaluate_expr_to_expr(&args[0])
          .unwrap_or_else(|_| args[0].clone()),
      )
    }
    // Outside a `Graphics`, `Text[expr]` only asks for `expr` to be shown
    // in text rather than mathematical form — it contributes no box of its
    // own, so export what it holds. (In a `Grid` cell this is the
    // difference between the typeset row and the string `Text[Row[…]]`.)
    Expr::FunctionCall { name, args } if name == "Text" && args.len() == 1 => {
      expr_to_svg(&unquoted_display_string(&args[0]))
    }
    // `Style[content, directives…]` shows `content`; the directives only
    // change how it is set. The picture path has to look through the
    // wrapper, or a styled layout — `Style[Row[{n, " leaves, ", …}],
    // FontSize -> 16, Blue]`, how a Demonstration captions its plot —
    // falls through to the text renderer, which prints the call's own
    // source instead of the row it wraps. A `Style` is inherited by what
    // it wraps, so its directives are pushed into the layout's items and
    // the item renderer applies them there. A styled `Grid` is left to the
    // arm below, which hands the directives to the grid renderer whole —
    // they colour its frame and dividers, not only its cells.
    Expr::FunctionCall { name, args }
      if name == "Style"
        && !args.is_empty()
        && !matches!(&args[0], Expr::FunctionCall { name, args }
          if (name == "Grid" || name == "TextGrid") && !args.is_empty()) =>
    {
      let inner = crate::functions::graphics::style_pushed_into_layout(
        &args[0],
        &args[1..],
      )
      .unwrap_or_else(|| args[0].clone());
      expr_to_svg(&inner)
    }
    // `Labeled[content, label]` (and `…, pos]`) puts the label beside the
    // content — below it by default. With a picture as the content the
    // FrontEnd stacks the two, so exporting one has to compose them
    // instead of writing the whole call out as text.
    Expr::FunctionCall { name, args }
      if name == "Labeled"
        && args.len() >= 2
        && lays_out_a_graphic(&args[0]) =>
    {
      let content = expr_to_svg(&args[0]);
      // A label is shown as text, so a string one loses its quotes.
      let label = expr_to_svg(&unquoted_display_string(&args[1]));
      let position = match args.get(2) {
        Some(Expr::Identifier(p)) => p.as_str(),
        _ => "Bottom",
      };
      let rows = match position {
        "Top" => vec![vec![label], vec![content]],
        "Left" => vec![vec![label, content]],
        "Right" => vec![vec![content, label]],
        _ => vec![vec![content], vec![label]],
      };
      crate::functions::graphics::combine_graphics_svgs(&rows)
        .unwrap_or_else(|| expr_text_svg(expr))
    }
    // `Grid`/`Column`/`Row` holding a picture are composed cell by cell —
    // pictures drawn, everything else typeset — rather than dumped as
    // source. Layouts of plain text keep the text renderer, which already
    // aligns them.
    Expr::FunctionCall { name, args }
      if LAYOUT_HEADS.contains(&name.as_str())
        && !args.is_empty()
        && lays_out_a_graphic(expr) =>
    {
      // `Column` places its own items, so an alignment argument is honoured
      // — `Column[{picture, legend}, Center]` centres the narrow legend
      // under the wide picture. The generic composition below packs every
      // row to the left instead.
      if name == "Column"
        && args.len() >= 2
        && let Some(svg) = crate::functions::graphics::column_to_svg(args)
      {
        return svg;
      }
      let rows: Vec<Vec<String>> = layout_rows(name, args)
        .iter()
        .map(|cells| cells.iter().map(expr_to_svg).collect())
        .collect();
      if rows.is_empty() {
        expr_text_svg(expr)
      } else {
        crate::functions::graphics::combine_graphics_svgs(&rows)
          .unwrap_or_else(|| expr_text_svg(expr))
      }
    }
    // A list of graphics renders as `{g1, g2, …}` with the plots drawn
    // inline (matching how wolframscript displays such a list), instead
    // of falling through to the text renderer which would dump
    // `GraphicsBox[]` placeholders.
    Expr::List(items)
      if !items.is_empty() && items.iter().all(is_inline_graphic) =>
    {
      let svgs: Vec<String> = items.iter().map(expr_to_svg).collect();
      if svgs.iter().all(|s| !s.is_empty())
        && let Some(svg) = crate::functions::graphics::graphics_list_svg(&svgs)
      {
        svg
      } else {
        expr_text_svg(expr)
      }
    }
    // A 2-D list of graphics renders as a grid of the plots.
    Expr::List(items)
      if !items.is_empty()
        && items.iter().all(|item| {
          matches!(item, Expr::List(inner)
            if !inner.is_empty() && inner.iter().all(is_inline_graphic))
        }) =>
    {
      let rows: Vec<Vec<String>> = items
        .iter()
        .map(|item| match item {
          Expr::List(inner) => inner.iter().map(expr_to_svg).collect(),
          _ => unreachable!(),
        })
        .collect();
      if rows.iter().flatten().all(|s: &String| !s.is_empty())
        && let Some(svg) =
          crate::functions::graphics::combine_graphics_svgs(&rows)
      {
        svg
      } else {
        expr_text_svg(expr)
      }
    }
    // Legended[graphics, legend]: the wrapped graphics carry the legend
    // baked into their SVG (e.g. PeriodicTablePlot["Phase"]).
    Expr::FunctionCall { name, args }
      if name == "Legended" && !args.is_empty() =>
    {
      expr_to_svg(&args[0])
    }
    // ComputationalMusic objects render as musical-staff notation.
    Expr::FunctionCall { name, .. }
      if crate::functions::music_ast::MUSIC_OBJECT_HEADS
        .contains(&name.as_str())
        && crate::functions::music_render::music_to_svg(expr).is_some() =>
    {
      crate::functions::music_render::music_to_svg(expr).unwrap_or_default()
    }
    // A plain list of music events (e.g. {MusicNote[…], MusicNote[…]}) keeps
    // its list structure: `{ <staff>, <staff>, … }`, each element drawn as its
    // own staff rather than a bracketed expression dump.
    _ if crate::functions::music_ast::is_music_object_list(expr) => {
      if let Some(svg) = crate::functions::music_render::music_list_to_svg(expr)
      {
        svg
      } else {
        expr_text_svg(expr)
      }
    }
    Expr::Identifier(s) if s == "-Graphics-" || s == "-Graphics3D-" => {
      crate::get_captured_graphics().unwrap_or_default()
    }
    Expr::FunctionCall {
      name: gfx_name,
      args: gfx_args,
    } if gfx_name == "Graphics" && !gfx_args.is_empty() => {
      if let Ok(Expr::Graphics {
        svg: ref svg_data, ..
      }) = crate::functions::graphics::graphics_ast(gfx_args)
      {
        svg_data.clone()
      } else {
        String::new()
      }
    }
    Expr::FunctionCall {
      name: gfx_name,
      args: gfx_args,
    } if gfx_name == "Graphics3D" && !gfx_args.is_empty() => {
      if let Ok(Expr::Graphics {
        svg: ref svg_data, ..
      }) = crate::functions::plot3d::graphics3d_ast(gfx_args)
      {
        svg_data.clone()
      } else {
        String::new()
      }
    }
    Expr::FunctionCall {
      name: grid_name,
      args: grid_args,
    } if (grid_name == "Grid" || grid_name == "TextGrid")
      && !grid_args.is_empty() =>
    {
      if crate::functions::graphics::grid_ast(grid_args).is_ok() {
        crate::get_captured_graphics().unwrap_or_default()
      } else {
        String::new()
      }
    }
    // TraditionalForm[Grid[...]] or TraditionalForm[TextGrid[...]]
    Expr::FunctionCall {
      name: tf_name,
      args: tf_args,
    } if tf_name == "TraditionalForm"
      && tf_args.len() == 1
      && matches!(
        &tf_args[0],
        Expr::FunctionCall { name, args }
        if (name == "Grid" || name == "TextGrid") && !args.is_empty()
      ) =>
    {
      if let Expr::FunctionCall {
        args: grid_args, ..
      } = &tf_args[0]
      {
        if crate::functions::graphics::grid_ast(grid_args).is_ok() {
          crate::get_captured_graphics().unwrap_or_default()
        } else {
          String::new()
        }
      } else {
        String::new()
      }
    }
    Expr::FunctionCall {
      name: style_name,
      args: style_args,
    } if style_name == "Style"
      && style_args.len() >= 2
      && matches!(
        &style_args[0],
        Expr::FunctionCall { name, args }
        if (name == "Grid" || name == "TextGrid") && !args.is_empty()
      ) =>
    {
      if let Expr::FunctionCall {
        args: grid_args, ..
      } = &style_args[0]
      {
        let style =
          crate::functions::graphics::parse_grid_style(&style_args[1..]);
        if crate::functions::graphics::grid_ast_styled(grid_args, &style)
          .is_ok()
        {
          crate::get_captured_graphics().unwrap_or_default()
        } else {
          String::new()
        }
      } else {
        String::new()
      }
    }
    Expr::FunctionCall {
      name: row_name,
      args: row_args,
    } if row_name == "Row" && !row_args.is_empty() => {
      if let Some(svg) = crate::row_svg_with_rendered_items(row_args) {
        svg
      } else {
        expr_text_svg(expr)
      }
    }
    Expr::FunctionCall {
      name: fr_name,
      args: fr_args,
    } if fr_name == "Framed" && !fr_args.is_empty() => {
      crate::functions::graphics::framed_to_svg(fr_args)
        .unwrap_or_else(|| expr_text_svg(expr))
    }
    Expr::FunctionCall {
      name: hl_name,
      args: hl_args,
    } if hl_name == "Highlighted" && !hl_args.is_empty() => {
      crate::functions::graphics::highlighted_to_svg(hl_args)
        .unwrap_or_else(|| expr_text_svg(expr))
    }
    // A list with Framed/Highlighted elements renders as a row `{1, |2|, …}`
    // with the wrapped items drawn as boxes, matching the inline display path.
    Expr::List(items) if items.iter().any(contains_framed_or_highlighted) => {
      crate::functions::graphics::row_with_framed_to_svg(items)
        .unwrap_or_else(|| expr_text_svg(expr))
    }
    Expr::FunctionCall {
      name: ds_name,
      args: ds_args,
    } if ds_name == "Dataset" && !ds_args.is_empty() => {
      if let Some(svg) = crate::functions::graphics::dataset_to_svg(&ds_args[0])
      {
        svg
      } else {
        expr_text_svg(expr)
      }
    }
    Expr::FunctionCall {
      name: tab_name,
      args: tab_args,
    } if tab_name == "Tabular" && tab_args.len() >= 2 => {
      if let Some(svg) =
        crate::functions::graphics::tabular_to_svg(&tab_args[0], &tab_args[1])
      {
        svg
      } else {
        expr_text_svg(expr)
      }
    }
    Expr::FunctionCall {
      name: tf_name,
      args: tf_args,
    } if tf_name == "TreeForm" && !tf_args.is_empty() => {
      if let Ok(Expr::Graphics {
        svg: ref svg_data, ..
      }) = crate::functions::tree_form::tree_form_ast(tf_args)
      {
        svg_data.clone()
      } else {
        String::new()
      }
    }
    Expr::FunctionCall {
      name: tg_name,
      args: tg_args,
    } if tg_name == "TreeGraph" && !tg_args.is_empty() => {
      if let Ok(Expr::Graphics {
        svg: ref svg_data, ..
      }) = crate::functions::tree_form::tree_graph_ast(tg_args)
      {
        svg_data.clone()
      } else {
        String::new()
      }
    }
    Expr::FunctionCall {
      name: g_name,
      args: g_args,
    } if g_name == "Graph" && g_args.len() >= 2 => {
      if let Ok(Expr::Graphics {
        svg: ref svg_data, ..
      }) = crate::functions::graph::graph_ast(g_args)
      {
        svg_data.clone()
      } else {
        String::new()
      }
    }
    Expr::FunctionCall {
      name: rg_name,
      args: rg_args,
    } if rg_name == "Region" && !rg_args.is_empty() => {
      if let Some(Expr::Graphics { ref svg, .. }) =
        crate::functions::region::region_to_graphics(rg_args)
      {
        svg.clone()
      } else {
        expr_text_svg(expr)
      }
    }
    Expr::FunctionCall {
      name: pc_name,
      args: pc_args,
    } if (pc_name == "PolarCurve" || pc_name == "FilledPolarCurve")
      && !pc_args.is_empty() =>
    {
      if let Some(Expr::Graphics { ref svg, .. }) =
        crate::functions::graphics::polar_curve_to_graphics(pc_name, pc_args)
      {
        svg.clone()
      } else {
        expr_text_svg(expr)
      }
    }
    Expr::FunctionCall { name: do_name, .. } if do_name == "DateObject" => {
      crate::functions::datetime_ast::date_object_panel_svg(expr)
        .unwrap_or_else(|| expr_text_svg(expr))
    }
    Expr::FunctionCall {
      name: mr_name,
      args: mr_args,
    } if mr_name == "MeshRegion" && mr_args.len() == 2 => {
      if let Some(svg) =
        crate::functions::voronoi::mesh_region_to_svg(&mr_args[0], &mr_args[1])
      {
        svg
      } else {
        expr_text_svg(expr)
      }
    }
    // DisplayForm[boxes] / RawBoxes[boxes] — render box expressions to SVG
    Expr::FunctionCall {
      name: box_name,
      args: box_args,
    } if (box_name == "DisplayForm" || box_name == "RawBoxes")
      && box_args.len() == 1 =>
    {
      boxes_to_text_svg(&box_args[0])
    }
    // QuestionObject[…] — render as a question panel (prompt, answer
    // choices / input field, Submit button).
    Expr::FunctionCall { name: qo_name, .. } if qo_name == "QuestionObject" => {
      crate::functions::assessment_render::question_object_to_svg(expr)
        .unwrap_or_else(|| expr_text_svg(expr))
    }
    // Molecule[…] — render the 2-D structure diagram, prefixed with an XML
    // declaration as wolframscript's SVG export is (a standalone document).
    Expr::FunctionCall { name: mol_name, .. } if mol_name == "Molecule" => {
      match crate::functions::molecule_render::molecule_to_svg(expr) {
        Some(svg) => {
          format!("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n{svg}")
        }
        None => expr_text_svg(expr),
      }
    }
    // MoleculePlot[mol] — render the full 2-D skeletal structure diagram.
    Expr::FunctionCall {
      name: mp_name,
      args: mp_args,
    } if mp_name == "MoleculePlot" && mp_args.len() == 1 => {
      crate::functions::molecule_render::molecule_to_svg(&mp_args[0])
        .unwrap_or_else(|| expr_text_svg(expr))
    }
    // Image[…] — embed the raster as a base64 PNG inside an <image> element
    // so the SVG stays a valid vector wrapper around the pixel data.
    Expr::Image {
      width,
      height,
      channels,
      data,
      ..
    } => crate::functions::image_ast::image_to_svg_document(
      *width, *height, *channels, data,
    ),
    // TableForm[data, opts…] — render the aligned grid (matching how visual
    // frontends show it), instead of dumping the raw `TableForm[…]` text.
    Expr::FunctionCall { name, args }
      if name == "TableForm" && !args.is_empty() =>
    {
      // A table of Graphics is combined into a grid of the plots.
      if let Some(svg) = tableform_graphics_grid_svg(args) {
        return svg;
      }
      let rendered = crate::render_tableform_if_needed(expr.clone());
      if let Expr::Graphics { svg, .. } = &rendered {
        svg.clone()
      } else {
        expr_text_svg(expr)
      }
    }
    // MatrixForm[data] — render as a parenthesized matrix.
    Expr::FunctionCall { name, args }
      if name == "MatrixForm" && !args.is_empty() =>
    {
      let _ = args;
      let rendered = crate::render_matrixform_if_needed(expr.clone());
      if let Expr::Graphics { svg, .. } = &rendered {
        svg.clone()
      } else {
        expr_text_svg(expr)
      }
    }
    // TraditionalForm[list] — render the list as a parenthesized matrix.
    Expr::FunctionCall { name, args }
      if name == "TraditionalForm"
        && args.len() == 1
        && matches!(&args[0], Expr::List(_)) =>
    {
      let _ = args;
      let rendered = crate::render_traditionalform_list_if_needed(expr.clone());
      if let Expr::Graphics { svg, .. } = &rendered {
        svg.clone()
      } else {
        expr_text_svg(expr)
      }
    }
    // Style[TableForm|MatrixForm[data], directives…] — render the styled grid
    // (font size, weight, color) rather than the raw expression text.
    Expr::FunctionCall { name, args }
      if name == "Style"
        && args.len() >= 2
        && matches!(
          &args[0],
          Expr::FunctionCall { name: inner, args: inner_args }
          if (inner == "TableForm" || inner == "MatrixForm")
            && !inner_args.is_empty()
        ) =>
    {
      if let Some(svg) = styled_tableform_svg(&args[0], &args[1..]) {
        svg
      } else {
        expr_text_svg(expr)
      }
    }
    // Number-format wrappers around a table — e.g.
    // `ScientificForm[TableForm[m], 3]` or
    // `PaddedForm[BaseForm[TableForm[m], 2], 8]` — format each cell and render
    // the resulting grid.
    Expr::FunctionCall { name, .. }
      if is_number_format_head(name)
        && distribute_format_over_tableform(expr).is_some() =>
    {
      let table = distribute_format_over_tableform(expr).unwrap();
      let rendered = crate::render_tableform_if_needed(table);
      if let Expr::Graphics { svg, .. } = &rendered {
        svg.clone()
      } else {
        expr_text_svg(expr)
      }
    }
    other => expr_text_svg(other),
  }
}

/// Heads of number-display wrappers that format a value without changing it.
fn is_number_format_head(name: &str) -> bool {
  matches!(
    name,
    "ScientificForm"
      | "EngineeringForm"
      | "NumberForm"
      | "PaddedForm"
      | "AccountingForm"
      | "BaseForm"
  )
}

/// If `expr` is one or more nested number-format wrappers around a
/// `TableForm[data, opts…]`, push those wrappers down onto every data cell and
/// return the rewritten `TableForm`. Returns `None` when no `TableForm` is
/// found under the wrappers.
fn distribute_format_over_tableform(expr: &Expr) -> Option<Expr> {
  // Peel the format wrappers (outermost first), remembering each head and its
  // trailing arguments, until we reach the TableForm.
  let mut wrappers: Vec<(String, Vec<Expr>)> = Vec::new();
  let mut cur = expr;
  loop {
    match cur {
      Expr::FunctionCall { name, args }
        if is_number_format_head(name) && !args.is_empty() =>
      {
        wrappers.push((name.clone(), args[1..].to_vec()));
        cur = &args[0];
      }
      Expr::FunctionCall { name, args }
        if name == "TableForm" && !args.is_empty() =>
      {
        if wrappers.is_empty() {
          return None;
        }
        // Re-wrap each leaf cell, innermost wrapper applied first so nesting
        // order matches the original (outer wrapper stays outermost).
        let new_data = map_leaf_cells(&args[0], &|v| {
          let mut e = v.clone();
          for (head, targs) in wrappers.iter().rev() {
            let mut a = vec![e];
            a.extend(targs.iter().cloned());
            e = Expr::FunctionCall {
              name: head.clone(),
              args: a.into(),
            };
          }
          e
        });
        let mut new_args = vec![new_data];
        new_args.extend(args[1..].iter().cloned());
        return Some(call("TableForm", new_args));
      }
      _ => return None,
    }
  }
}

/// Apply `f` to every non-list leaf of a (possibly nested) list, preserving the
/// list structure.
fn map_leaf_cells(expr: &Expr, f: &dyn Fn(&Expr) -> Expr) -> Expr {
  match expr {
    Expr::List(items) => {
      Expr::List(items.iter().map(|e| map_leaf_cells(e, f)).collect())
    }
    other => f(other),
  }
}

/// Render `TableForm[graphics…]` — a table whose cells are `Graphics`
/// objects — as a combined grid of the plots. Returns `None` when the data
/// isn't a (ragged) 2-D list of inline graphics.
fn tableform_graphics_grid_svg(args: &[Expr]) -> Option<String> {
  let Expr::List(rows) = args.first()? else {
    return None;
  };
  // A cell is a graphic if it is already rendered (`Expr::Graphics`) or is a
  // `Graphics[…]` / `Graphics3D[…]` call that `expr_to_svg` renders on demand.
  fn is_graphic_cell(e: &Expr) -> bool {
    is_inline_graphic(e)
      || matches!(e, Expr::FunctionCall { name, args }
        if (name == "Graphics" || name == "Graphics3D") && !args.is_empty())
  }
  // Every row must itself be a non-empty list of graphics; a table of plain
  // values is handled by the ordinary grid path instead. At least one cell
  // must be a genuinely rendered graphic (not only bare `Graphics[…]` echoes)
  // so ordinary symbolic tables don't get diverted here.
  if rows.is_empty()
    || !rows.iter().all(|r| {
      matches!(r, Expr::List(cells)
        if !cells.is_empty() && cells.iter().all(is_graphic_cell))
    })
  {
    return None;
  }
  let grid: Vec<Vec<String>> = rows
    .iter()
    .map(|r| match r {
      Expr::List(cells) => cells.iter().map(expr_to_svg).collect(),
      _ => unreachable!(),
    })
    .collect();
  if grid.iter().flatten().any(std::string::String::is_empty) {
    return None;
  }
  crate::functions::graphics::combine_graphics_svgs(&grid)
}

/// Render `Style[TableForm|MatrixForm[data], directives…]` to a styled grid
/// SVG. Falls back to `None` when the data can't be reshaped into a grid.
fn styled_tableform_svg(inner: &Expr, directives: &[Expr]) -> Option<String> {
  let Expr::FunctionCall {
    name: head,
    args: inner_args,
  } = inner
  else {
    return None;
  };
  let style = crate::functions::graphics::parse_grid_style(directives);
  let rendered = if head == "MatrixForm" {
    // MatrixForm reshapes the same way as TableForm, but is drawn with
    // enclosing parentheses.
    let (grid_args, _gaps) = crate::tableform_grid_args(inner_args)?;
    crate::functions::graphics::grid_ast_styled_with_parens(&grid_args, &style)
  } else {
    let (grid_args, _gaps) = crate::tableform_grid_args(inner_args)?;
    crate::functions::graphics::grid_ast_styled(&grid_args, &style)
  }
  .ok()?;
  if let Expr::Graphics { svg, .. } = &rendered {
    Some(svg.clone())
  } else {
    None
  }
}

/// Fonts bundled for embedding into exported SVGs. These are the same faces the
/// PDF/PNG rasterizers map the generic CSS families onto (see
/// `svg_to_pdf_bytes` and `image_ast::load_embedded_fonts`), so vector export
/// stays visually consistent with raster export and renders identically on
/// systems where the fonts aren't installed.
const ATKINSON_MONO_TTF: &[u8] = include_bytes!(
  "../../../resources/AtkinsonHyperlegibleMono-VariableFont_wght.ttf"
);
const ATKINSON_NEXT_TTF: &[u8] = include_bytes!(
  "../../../resources/AtkinsonHyperlegibleNext-VariableFont_wght.ttf"
);

/// Base64 `data:` URL for a bundled font, encoded once and cached for reuse
/// across exports.
fn font_data_url(
  cache: &'static std::sync::OnceLock<String>,
  bytes: &'static [u8],
) -> &'static str {
  cache.get_or_init(|| {
    use base64::Engine;
    let b64 = base64::engine::general_purpose::STANDARD.encode(bytes);
    format!("data:font/ttf;base64,{b64}")
  })
}

/// A single `@font-face` CSS rule embedding `family` from `data_url`.
fn font_face_rule(family: &str, data_url: &str) -> String {
  format!(
    "@font-face {{ font-family: \"{family}\"; font-weight: normal; \
font-style: normal; src: url(\"{data_url}\") format(\"truetype\"); }}\n"
  )
}

/// Embed the fonts an exported SVG uses so it renders identically without those
/// fonts installed on the viewer's system.
///
/// Woxi's renderers emit only the generic CSS families `monospace`,
/// `sans-serif` and `serif` (and bare `<text>` with no family); hosts normally
/// map these onto the bundled Atkinson faces via their own CSS — see
/// `tests/playground/style.css`. A standalone exported SVG has no such host, so
/// we inline that same mapping together with `@font-face` rules carrying the
/// base64-encoded font data.
///
/// We also bake the preferred Atkinson family directly onto each text
/// element's `font-family` attribute, keeping the generic family as a
/// fallback (e.g. `"Atkinson Hyperlegible Next, sans-serif"`). CSS-aware
/// renderers apply the equivalent `svg text {}` rule; renderers that read the
/// presentation attribute but ignore that type-selector rule still pick up the
/// embedded face from the attribute. Both paths therefore converge on
/// Atkinson. The Mono face is embedded only when the document actually uses
/// monospace text, to avoid bloating graphics that don't.
fn embed_used_fonts(svg: &str) -> String {
  // Only SVG documents that actually draw text need embedded fonts.
  if !svg.contains("<svg") || !svg.contains("<text") {
    return svg.to_string();
  }

  // Sans-serif (Atkinson Hyperlegible Next) is the default for every text
  // element, so it is always embedded once there is any text.
  static NEXT_URL: std::sync::OnceLock<String> = std::sync::OnceLock::new();
  let mut style = font_face_rule(
    "Atkinson Hyperlegible Next",
    font_data_url(&NEXT_URL, ATKINSON_NEXT_TTF),
  );

  // Monospace text is tagged with `monospace`, a `Mono` family, or `Courier`
  // (see the selectors below); embed the Mono face only when a `font-family`
  // attribute actually requests one. Scanning attribute values rather than the
  // whole document avoids pulling the face in for a text label that merely
  // contains one of those words.
  let needs_mono = svg.match_indices("font-family=\"").any(|(i, m)| {
    svg[i + m.len()..].split('"').next().is_some_and(|val| {
      val.contains("monospace")
        || val.contains("Mono")
        || val.contains("Courier")
    })
  });
  if needs_mono {
    static MONO_URL: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    style.push_str(&font_face_rule(
      "Atkinson Hyperlegible Mono",
      font_data_url(&MONO_URL, ATKINSON_MONO_TTF),
    ));
  }

  style.push_str(
    "svg text { font-family: \"Atkinson Hyperlegible Next\", sans-serif; }\n",
  );
  if needs_mono {
    style.push_str(
      "svg text[font-family~=\"monospace\"], \
svg text[font-family*=\"Mono\"], \
svg text[font-family*=\"Courier\"] \
{ font-family: \"Atkinson Hyperlegible Mono\", monospace; }\n",
    );
  }

  // Bake the preferred Atkinson family onto each text element's `font-family`
  // attribute, keeping the generic family as a fallback. The generic token is
  // retained so the mono `[font-family*="Mono"]` CSS selectors below still
  // match. Exact-quoted matches make this idempotent (an already-rewritten
  // value no longer equals the generic form).
  let body = svg
    .replace(
      "font-family=\"sans-serif\"",
      "font-family=\"Atkinson Hyperlegible Next, sans-serif\"",
    )
    .replace(
      "font-family=\"monospace\"",
      "font-family=\"Atkinson Hyperlegible Mono, monospace\"",
    )
    .replace(
      "font-family=\"serif\"",
      "font-family=\"Atkinson Hyperlegible Next, serif\"",
    );

  // Insert the style block immediately after the opening `<svg …>` tag. The
  // tag's attribute values never contain `>`, so the first `>` after `<svg`
  // reliably closes it.
  let Some(open) = body.find("<svg") else {
    return body;
  };
  let Some(rel) = body[open..].find('>') else {
    return body;
  };
  let insert_at = open + rel + 1;
  let style_block =
    format!("\n<defs><style type=\"text/css\">\n{style}</style></defs>");

  let mut out = String::with_capacity(body.len() + style_block.len());
  out.push_str(&body[..insert_at]);
  out.push_str(&style_block);
  out.push_str(&body[insert_at..]);
  out
}

/// Convert an SVG string to PDF bytes using svg2pdf.
#[cfg(not(target_arch = "wasm32"))]
fn svg_to_pdf_bytes(svg_str: &str) -> Result<Vec<u8>, InterpreterError> {
  use std::sync::Arc as StdArc;

  let mut fontdb = svg2pdf::usvg::fontdb::Database::new();
  // Load system fonts first, then embedded fonts + generic-family aliases.
  // load_system_fonts() resets the generic family mappings, so our
  // set_sans_serif_family() etc. must come *after* it.
  fontdb.load_system_fonts();
  fontdb.load_font_data(
    include_bytes!(
      "../../../resources/AtkinsonHyperlegibleMono-VariableFont_wght.ttf"
    )
    .to_vec(),
  );
  fontdb.load_font_data(
    include_bytes!(
      "../../../resources/AtkinsonHyperlegibleNext-VariableFont_wght.ttf"
    )
    .to_vec(),
  );
  fontdb.set_monospace_family("Atkinson Hyperlegible Mono");
  fontdb.set_sans_serif_family("Atkinson Hyperlegible Next");
  fontdb.set_serif_family("Atkinson Hyperlegible Next");
  fontdb.set_cursive_family("Atkinson Hyperlegible Next");
  fontdb.set_fantasy_family("Atkinson Hyperlegible Next");

  let opt = svg2pdf::usvg::Options {
    fontdb: StdArc::new(fontdb),
    ..Default::default()
  };

  let tree = svg2pdf::usvg::Tree::from_str(svg_str, &opt).map_err(|e| {
    InterpreterError::EvaluationError(format!(
      "Export PDF: SVG parse error: {e}"
    ))
  })?;

  let pdf_bytes = svg2pdf::to_pdf(
    &tree,
    svg2pdf::ConversionOptions::default(),
    svg2pdf::PageOptions::default(),
  )
  .map_err(|e| {
    InterpreterError::EvaluationError(format!(
      "Export PDF: conversion error: {e}"
    ))
  })?;

  Ok(pdf_bytes)
}

/// The text `ReadString` returns from the remaining input, plus how many bytes
/// it consumes. Without a terminator that is everything; with one, a leading
/// terminator is skipped first and the read stops *at* the next one, leaving
/// it in place — so repeated reads yield the separated fields while a
/// following plain read still sees the separator. Returns `None` at the end of
/// the input, which `ReadString` reports as `EndOfFile`.
#[cfg(not(target_arch = "wasm32"))]
fn read_string_chunk(
  rest: &str,
  terminator: Option<&str>,
) -> Option<(String, usize)> {
  if rest.is_empty() {
    return None;
  }
  let Some(term) = terminator else {
    return Some((rest.to_string(), rest.len()));
  };
  let (skipped, body) = match rest.strip_prefix(term) {
    Some(after) => (term.len(), after),
    None => (0, rest),
  };
  if body.is_empty() {
    return None;
  }
  let end = body.find(term).unwrap_or(body.len());
  Some((body[..end].to_string(), skipped + end))
}

pub(crate) fn readlist_inputstream(
  args: &[Expr],
) -> Result<String, InterpreterError> {
  let Expr::Integer(id) = &args[1] else {
    return Err(InterpreterError::EvaluationError(
      "ReadList: invalid stream object".into(),
    ));
  };
  let stream_id = *id as usize;
  let Some((kind, _)) = get_stream_kind(stream_id) else {
    return Err(InterpreterError::EvaluationError(
      "ReadList: stream is not open".into(),
    ));
  };
  // A file that cannot be read is the one case with its own message; every
  // other kind (including a command's output) is drained by
  // `get_stream_content`, which reads it whole.
  #[cfg(not(target_arch = "wasm32"))]
  if let StreamKind::File(path) = &kind
    && !std::path::Path::new(path).is_file()
  {
    return Err(InterpreterError::EvaluationError(format!(
      "ReadList::noopen: Cannot open {path}."
    )));
  }
  Ok(get_stream_content(stream_id).map_or_else(String::new, |(c, _)| c))
}
