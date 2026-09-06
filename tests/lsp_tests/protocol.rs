//! Framing of the JSON-RPC base protocol.

use serde_json::json;
use std::io::Cursor;
use woxi::lsp::protocol::{encode, read_message, write_message};

#[test]
fn encodes_with_a_content_length_header() {
  let framed = encode(&json!({ "jsonrpc": "2.0", "id": 1 }));
  let body = r#"{"jsonrpc":"2.0","id":1}"#;
  assert_eq!(
    framed,
    format!("Content-Length: {}\r\n\r\n{body}", body.len())
  );
}

#[test]
fn reads_a_framed_message() {
  let message = json!({ "jsonrpc": "2.0", "method": "shutdown", "id": 7 });
  let mut input = Cursor::new(encode(&message).into_bytes());
  assert_eq!(read_message(&mut input).unwrap(), Some(message));
}

#[test]
fn reads_several_messages_from_one_stream() {
  let first = json!({ "jsonrpc": "2.0", "method": "a", "id": 1 });
  let second = json!({ "jsonrpc": "2.0", "method": "b", "id": 2 });
  let stream = format!("{}{}", encode(&first), encode(&second));
  let mut input = Cursor::new(stream.into_bytes());
  assert_eq!(read_message(&mut input).unwrap(), Some(first));
  assert_eq!(read_message(&mut input).unwrap(), Some(second));
  assert_eq!(read_message(&mut input).unwrap(), None);
}

#[test]
fn ignores_other_headers_and_header_case() {
  let body = r#"{"jsonrpc":"2.0","method":"exit"}"#;
  let stream = format!(
    "content-length: {}\r\n\
     Content-Type: application/vscode-jsonrpc; charset=utf-8\r\n\r\n{body}",
    body.len()
  );
  let mut input = Cursor::new(stream.into_bytes());
  assert_eq!(
    read_message(&mut input).unwrap(),
    Some(json!({ "jsonrpc": "2.0", "method": "exit" }))
  );
}

#[test]
fn a_message_body_is_measured_in_bytes_not_characters() {
  // "√" is three bytes but one character; a length in characters would
  // truncate the body and fail to parse.
  let message = json!({ "jsonrpc": "2.0", "method": "√" });
  let mut input = Cursor::new(encode(&message).into_bytes());
  assert_eq!(read_message(&mut input).unwrap(), Some(message));
}

#[test]
fn an_empty_stream_ends_cleanly() {
  let mut input = Cursor::new(Vec::new());
  assert_eq!(read_message(&mut input).unwrap(), None);
}

#[test]
fn a_missing_content_length_is_an_error() {
  let mut input = Cursor::new(b"Content-Type: text/plain\r\n\r\n{}".to_vec());
  assert!(read_message(&mut input).is_err());
}

#[test]
fn an_invalid_body_is_an_error() {
  let mut input = Cursor::new(b"Content-Length: 3\r\n\r\nnot".to_vec());
  assert!(read_message(&mut input).is_err());
}

#[test]
fn round_trips_through_a_writer() {
  let message = json!({ "jsonrpc": "2.0", "result": { "ok": true }, "id": 3 });
  let mut buffer = Vec::new();
  write_message(&mut buffer, &message).unwrap();
  let mut input = Cursor::new(buffer);
  assert_eq!(read_message(&mut input).unwrap(), Some(message));
}
