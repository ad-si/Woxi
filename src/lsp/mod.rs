//! A [Language Server Protocol] implementation for the Wolfram Language,
//! started with `woxi lsp`.
//!
//! The server speaks LSP over stdin/stdout, which is what editors expect
//! from a command they launch themselves. It provides
//!
//! - diagnostics: syntax errors from Woxi's own grammar, warnings on
//!   symbols this interpreter does not implement (so a script's
//!   unsupported parts are visible before running it), and hints on names
//!   that are one typo away from a built-in,
//! - hover documentation and completion for every `System`` symbol, drawn
//!   from the same `functions.csv` registry the interpreter uses, and for
//!   the symbols the edited file defines itself,
//! - go to definition, find references, occurrence highlighting and the
//!   document outline, computed from the file's assignments,
//! - semantic highlighting, which tells a file's own definitions and the
//!   parameters of a definition apart from the built-ins around them,
//! - formatting, which normalizes the spacing and indentation of a file
//!   without moving a single line break (see [`format`]),
//! - code actions, which correct a misspelled built-in in one place or
//!   throughout the file.
//!
//! Everything is derived from a forgiving tokeniser rather than the full
//! parser, so the answers stay useful while a file is mid-edit and does
//! not parse.
//!
//! [Language Server Protocol]: https://microsoft.github.io/language-server-protocol/

pub mod analysis;
pub mod format;
pub mod protocol;
mod server;

pub use server::{Document, Server};

/// Run the language server on stdin/stdout until the client disconnects.
///
/// Returns the process exit code the specification prescribes: `0` when
/// the client shut the server down properly, `1` when it sent `exit`
/// without a preceding `shutdown` or dropped the connection.
#[cfg(not(target_arch = "wasm32"))]
pub fn run_stdio() -> std::io::Result<i32> {
  use std::io::{BufReader, ErrorKind};

  // Anything the interpreter or a panic would print must not end up on
  // stdout: that stream carries the protocol.
  let stdin = std::io::stdin();
  let mut reader = BufReader::new(stdin.lock());
  let stdout = std::io::stdout();
  let mut writer = stdout.lock();

  let mut server = Server::new();
  // A stream that no longer yields framed messages cannot be resynced;
  // give up rather than spin on it.
  let mut malformed_in_a_row = 0_u32;

  loop {
    let message = match protocol::read_message(&mut reader) {
      Ok(Some(message)) => message,
      // Clean end of stream: the editor exited without saying goodbye.
      Ok(None) => break,
      Err(error) if error.kind() == ErrorKind::InvalidData => {
        malformed_in_a_row += 1;
        eprintln!("woxi lsp: ignoring malformed message: {error}");
        protocol::write_message(
          &mut writer,
          &serde_json::json!({
            "jsonrpc": "2.0",
            "id": Option::<u8>::None,
            "error": {
              "code": protocol::PARSE_ERROR,
              "message": error.to_string(),
            },
          }),
        )?;
        if malformed_in_a_row >= 5 {
          eprintln!("woxi lsp: giving up on an unreadable input stream");
          break;
        }
        continue;
      }
      Err(error) => return Err(error),
    };
    malformed_in_a_row = 0;

    for response in server.handle_message(&message) {
      protocol::write_message(&mut writer, &response)?;
    }
    if server.exit_requested() {
      break;
    }
  }

  Ok(i32::from(!server.shutdown_requested()))
}
