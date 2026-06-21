//! Interactive Read-Eval-Print Loop for Woxi.
//!
//! `woxi repl` starts an interactive session that evaluates Wolfram Language
//! input line by line. Unlike `woxi eval` (a fresh process per expression),
//! the REPL keeps all interpreter state — variable bindings, function
//! definitions, `%` / `Out[]` history — alive across evaluations in a single
//! process, matching wolframscript's terminal REPL.
//!
//! Two input front-ends are used:
//!   * an interactive line editor (history, cursor editing) when stdin is a
//!     terminal, and
//!   * a plain buffered reader when stdin is piped (scripts, tests), which
//!     keeps captured output free of terminal escape sequences.

use std::io::{self, BufRead, IsTerminal, Write};

use rustyline::DefaultEditor;
use rustyline::error::ReadlineError;

use woxi::{InterpreterError, interpret};

/// Run the interactive REPL until the user quits (`Quit`/`Exit`, EOF, or two
/// consecutive interrupts).
pub fn run() {
  // Persist `%` / `Out[]` history across evaluations and route messages /
  // `Print` to real stdout (same as `woxi eval`).
  woxi::set_repl_mode(true);
  woxi::set_messages_to_stdout(true);

  let interactive = io::stdin().is_terminal();
  if interactive {
    print_banner();
  }

  let mut session = Session::new();
  if interactive {
    run_interactive(&mut session);
  } else {
    run_piped(&mut session);
  }
}

/// Print the startup banner (interactive sessions only).
fn print_banner() {
  let version = env!("CARGO_PKG_VERSION");
  println!("Woxi {version} — interactive Wolfram Language REPL");
  println!("Type Quit (or press Ctrl-D) to exit.\n");
}

/// Per-session line numbering and `In[]`/`Out[]` bookkeeping.
struct Session {
  line: u64,
}

impl Session {
  fn new() -> Self {
    Session { line: 1 }
  }

  /// Evaluate one complete logical input and print its result. Returns
  /// `false` when the input requests that the session terminate.
  fn eval(&mut self, input: &str) -> bool {
    let trimmed = input.trim();
    if trimmed.is_empty() {
      return true;
    }
    if is_exit_command(trimmed) {
      return false;
    }

    // Keep `$Line` in sync so `In[]` / `Out[]` indices line up with the
    // prompts the user sees.
    woxi::set_system_variable("$Line", &self.line.to_string());

    match interpret(input) {
      // "\0" is the sentinel for suppressed output (Null, a trailing
      // semicolon, or output already printed). Print no `Out[]` line, but
      // still advance the line counter — matching wolframscript.
      Ok(result) if result == "\0" => {}
      Ok(result) => {
        println!("Out[{}]= {}\n", self.line, result);
      }
      // Comment-only / whitespace-only input evaluates to Null with no
      // visible result.
      Err(InterpreterError::EmptyInput) => {}
      Err(e) => {
        eprintln!("Error: {e}");
        if let Some(trace) = woxi::take_error_trace() {
          eprintln!("{trace}");
        }
      }
    }
    let _ = io::stdout().flush();
    self.line += 1;
    true
  }

  fn prompt(&self) -> String {
    format!("In[{}]:= ", self.line)
  }
}

/// Interactive front-end backed by a line editor with history and cursor
/// editing.
fn run_interactive(session: &mut Session) {
  let mut editor = match DefaultEditor::new() {
    Ok(ed) => ed,
    Err(e) => {
      eprintln!("Error: could not start line editor: {e}");
      return;
    }
  };

  // Two consecutive Ctrl-C presses exit; a single press cancels the current
  // (possibly multi-line) input, like wolframscript / common REPLs.
  let mut pending_interrupt = false;

  loop {
    let mut buffer = String::new();
    let mut prompt = session.prompt();

    let complete = loop {
      match editor.readline(&prompt) {
        Ok(line) => {
          pending_interrupt = false;
          buffer.push_str(&line);
          if input_is_complete(&buffer) {
            break true;
          }
          // Unbalanced brackets / quotes: keep reading a continuation line.
          buffer.push('\n');
          prompt = continuation_prompt(&session.prompt());
        }
        Err(ReadlineError::Interrupted) => {
          if buffer.trim().is_empty() {
            if pending_interrupt {
              return;
            }
            pending_interrupt = true;
            println!("(To exit, type Quit or press Ctrl-C again)");
          }
          // Discard whatever was being typed and start a fresh prompt.
          break false;
        }
        Err(ReadlineError::Eof) => return,
        Err(e) => {
          eprintln!("Error: {e}");
          return;
        }
      }
    };

    if !complete {
      continue;
    }
    let _ = editor.add_history_entry(buffer.trim_end());
    if !session.eval(&buffer) {
      return;
    }
  }
}

/// Non-interactive front-end for piped stdin (scripts, tests). Reads logical
/// inputs (joining continuation lines) without emitting prompts or terminal
/// escapes, so captured stdout stays clean.
fn run_piped(session: &mut Session) {
  let stdin = io::stdin();
  let mut buffer = String::new();

  for line in stdin.lock().lines() {
    let line = match line {
      Ok(l) => l,
      Err(_) => break,
    };
    if !buffer.is_empty() {
      buffer.push('\n');
    }
    buffer.push_str(&line);
    if !input_is_complete(&buffer) {
      continue;
    }
    let input = std::mem::take(&mut buffer);
    if !session.eval(&input) {
      return;
    }
  }

  // Evaluate any trailing input left after EOF without a closing newline.
  if !buffer.trim().is_empty() {
    session.eval(&buffer);
  }
}

/// Whether `trimmed` (already trimmed) is a request to leave the REPL.
fn is_exit_command(trimmed: &str) -> bool {
  matches!(trimmed, "Quit" | "Exit" | "Quit[]" | "Exit[]")
}

/// Continuation prompt aligned to the width of the primary prompt, so
/// multi-line input stays visually indented under `In[n]:=`.
fn continuation_prompt(primary: &str) -> String {
  " ".repeat(primary.chars().count())
}

/// Heuristically decide whether `src` is a complete Wolfram Language input or
/// whether a continuation line is still expected. Tracks `()`, `[]`, `{}`
/// nesting while skipping over string literals (with `\` escapes) and
/// `(* … *)` comments. Unbalanced openers ⇒ incomplete.
fn input_is_complete(src: &str) -> bool {
  let mut depth: i32 = 0;
  let mut in_string = false;
  let mut escaped = false;
  let mut comment_depth: i32 = 0;

  let bytes = src.as_bytes();
  let mut i = 0;
  while i < bytes.len() {
    let c = bytes[i];

    if in_string {
      if escaped {
        escaped = false;
      } else if c == b'\\' {
        escaped = true;
      } else if c == b'"' {
        in_string = false;
      }
      i += 1;
      continue;
    }

    if comment_depth > 0 {
      if c == b'*' && i + 1 < bytes.len() && bytes[i + 1] == b')' {
        comment_depth -= 1;
        i += 2;
        continue;
      }
      if c == b'(' && i + 1 < bytes.len() && bytes[i + 1] == b'*' {
        comment_depth += 1;
        i += 2;
        continue;
      }
      i += 1;
      continue;
    }

    match c {
      b'"' => in_string = true,
      b'(' if i + 1 < bytes.len() && bytes[i + 1] == b'*' => {
        comment_depth += 1;
        i += 2;
        continue;
      }
      b'(' | b'[' | b'{' => depth += 1,
      b')' | b']' | b'}' => depth -= 1,
      _ => {}
    }
    i += 1;
  }

  // An unterminated string or comment, or unclosed brackets, means more
  // input is expected. A negative depth (too many closers) is a syntax
  // error the interpreter should report, so treat it as complete.
  depth <= 0 && !in_string && comment_depth == 0
}
