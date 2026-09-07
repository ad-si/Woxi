//! The JSON-RPC base protocol used by the Language Server Protocol.
//!
//! Messages are framed with HTTP-style headers:
//!
//! ```text
//! Content-Length: 42\r\n
//! \r\n
//! {"jsonrpc":"2.0", …}
//! ```
//!
//! Only `Content-Length` carries meaning for us; every other header (e.g.
//! `Content-Type`) is accepted and ignored, as the specification requires.

use serde_json::Value;
use std::io::{self, BufRead, Write};

/// JSON-RPC error codes used by the server (see the LSP specification).
pub(crate) const PARSE_ERROR: i64 = -32700;
pub(crate) const INVALID_REQUEST: i64 = -32600;
pub(crate) const METHOD_NOT_FOUND: i64 = -32601;
pub(crate) const INVALID_PARAMS: i64 = -32602;
pub(crate) const SERVER_NOT_INITIALIZED: i64 = -32002;

/// Read a single framed message from `reader`.
///
/// Returns `Ok(None)` when the stream ends cleanly between messages, which
/// is how an editor that dies without sending `exit` terminates the server.
pub fn read_message<R: BufRead>(reader: &mut R) -> io::Result<Option<Value>> {
  let mut content_length: Option<usize> = None;
  let mut line = String::new();
  loop {
    line.clear();
    if reader.read_line(&mut line)? == 0 {
      // EOF. Headers seen so far were incomplete, so there is no message.
      return Ok(None);
    }
    let header = line.trim_end_matches(['\r', '\n']);
    if header.is_empty() {
      break;
    }
    if let Some((name, value)) = header.split_once(':')
      && name.trim().eq_ignore_ascii_case("content-length")
    {
      content_length = value.trim().parse::<usize>().ok();
    }
  }

  let Some(length) = content_length else {
    return Err(io::Error::new(
      io::ErrorKind::InvalidData,
      "message is missing a valid Content-Length header",
    ));
  };

  let mut body = vec![0_u8; length];
  reader.read_exact(&mut body)?;
  serde_json::from_slice(&body)
    .map(Some)
    .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
}

/// Frame `message` for transmission.
pub fn encode(message: &Value) -> String {
  let body = message.to_string();
  format!("Content-Length: {}\r\n\r\n{body}", body.len())
}

/// Write a single framed message to `writer` and flush it.
///
/// Flushing on every message matters: an editor blocks on the response to
/// its request, so a buffered reply would deadlock the session.
pub fn write_message<W: Write>(
  writer: &mut W,
  message: &Value,
) -> io::Result<()> {
  writer.write_all(encode(message).as_bytes())?;
  writer.flush()
}
