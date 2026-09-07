//! End-to-end request handling of the language server.

use serde_json::{Value, json};
use woxi::lsp::Server;

/// A server that has completed the `initialize` handshake.
fn initialized_server() -> Server {
  let mut server = Server::new();
  let responses = server.handle_message(&json!({
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {
      "capabilities": {
        "textDocument": {
          "documentSymbol": { "hierarchicalDocumentSymbolSupport": true }
        }
      }
    },
  }));
  assert_eq!(responses.len(), 1);
  assert!(responses[0].get("result").is_some());
  server
}

/// Open `source` as `file:///test.wls` and return the notifications.
fn open(server: &mut Server, source: &str) -> Vec<Value> {
  server.handle_message(&json!({
    "jsonrpc": "2.0",
    "method": "textDocument/didOpen",
    "params": {
      "textDocument": {
        "uri": "file:///test.wls",
        "languageId": "wolfram",
        "version": 1,
        "text": source,
      }
    },
  }))
}

/// Send a request and return its `result`, failing on an error response.
fn request(server: &mut Server, method: &str, params: &Value) -> Value {
  let responses = server.handle_message(&json!({
    "jsonrpc": "2.0", "id": 42, "method": method, "params": params.clone(),
  }));
  assert_eq!(responses.len(), 1, "expected exactly one response");
  assert_eq!(responses[0]["id"], json!(42));
  assert!(
    responses[0].get("error").is_none(),
    "unexpected error: {}",
    responses[0]
  );
  responses[0]["result"].clone()
}

fn position_params(line: u64, character: u64) -> Value {
  json!({
    "textDocument": { "uri": "file:///test.wls" },
    "position": { "line": line, "character": character },
  })
}

/// The diagnostics of the single `publishDiagnostics` notification.
fn diagnostics_of(notifications: &[Value]) -> &Vec<Value> {
  assert_eq!(notifications.len(), 1);
  assert_eq!(
    notifications[0]["method"],
    json!("textDocument/publishDiagnostics")
  );
  notifications[0]["params"]["diagnostics"]
    .as_array()
    .unwrap()
}

#[test]
fn initialize_advertises_the_supported_features() {
  let mut server = Server::new();
  let responses = server.handle_message(&json!({
    "jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {},
  }));
  let capabilities = &responses[0]["result"]["capabilities"];
  assert_eq!(capabilities["textDocumentSync"]["change"], json!(1));
  assert_eq!(capabilities["hoverProvider"], json!(true));
  assert_eq!(capabilities["definitionProvider"], json!(true));
  assert_eq!(capabilities["referencesProvider"], json!(true));
  assert_eq!(capabilities["documentSymbolProvider"], json!(true));
  assert_eq!(capabilities["documentHighlightProvider"], json!(true));
  assert!(capabilities["completionProvider"].is_object());
  assert_eq!(responses[0]["result"]["serverInfo"]["name"], json!("woxi"));
}

#[test]
fn requests_before_initialize_are_rejected() {
  let mut server = Server::new();
  let responses = server.handle_message(&json!({
    "jsonrpc": "2.0", "id": 1, "method": "textDocument/hover", "params": {},
  }));
  assert_eq!(responses[0]["error"]["code"], json!(-32002));
}

#[test]
fn an_unknown_request_reports_method_not_found() {
  let mut server = initialized_server();
  let responses = server.handle_message(&json!({
    "jsonrpc": "2.0", "id": 2, "method": "textDocument/rename", "params": {},
  }));
  assert_eq!(responses[0]["error"]["code"], json!(-32601));
}

#[test]
fn an_unknown_notification_is_ignored() {
  let mut server = initialized_server();
  let responses = server.handle_message(&json!({
    "jsonrpc": "2.0", "method": "$/setTrace", "params": { "value": "off" },
  }));
  assert!(responses.is_empty());
}

#[test]
fn a_response_from_the_client_is_ignored() {
  let mut server = initialized_server();
  assert!(
    server
      .handle_message(&json!({ "jsonrpc": "2.0", "id": 1, "result": null }))
      .is_empty()
  );
}

#[test]
fn shutdown_and_exit_end_the_session() {
  let mut server = initialized_server();
  assert!(!server.shutdown_requested());
  let responses = server.handle_message(&json!({
    "jsonrpc": "2.0", "id": 9, "method": "shutdown",
  }));
  assert_eq!(responses[0]["result"], json!(null));
  assert!(server.shutdown_requested());
  // Requests after shutdown are invalid.
  let responses = server.handle_message(&json!({
    "jsonrpc": "2.0", "id": 10, "method": "textDocument/hover", "params": {},
  }));
  assert_eq!(responses[0]["error"]["code"], json!(-32600));
  assert!(!server.exit_requested());
  server.handle_message(&json!({ "jsonrpc": "2.0", "method": "exit" }));
  assert!(server.exit_requested());
}

#[test]
fn opening_a_document_publishes_diagnostics() {
  let mut server = initialized_server();
  let notifications = open(&mut server, "f[1, 2");
  let diagnostics = diagnostics_of(&notifications);
  assert_eq!(diagnostics.len(), 1);
  assert_eq!(diagnostics[0]["severity"], json!(1));
  assert_eq!(diagnostics[0]["source"], json!("woxi"));
  assert_eq!(diagnostics[0]["range"]["start"]["line"], json!(0));
}

#[test]
fn changing_a_document_republishes_diagnostics() {
  let mut server = initialized_server();
  assert_eq!(diagnostics_of(&open(&mut server, "f[1, 2")).len(), 1);
  let notifications = server.handle_message(&json!({
    "jsonrpc": "2.0",
    "method": "textDocument/didChange",
    "params": {
      "textDocument": { "uri": "file:///test.wls", "version": 2 },
      "contentChanges": [{ "text": "f[1, 2]" }],
    },
  }));
  assert!(diagnostics_of(&notifications).is_empty());
  assert_eq!(notifications[0]["params"]["version"], json!(2));
}

#[test]
fn closing_a_document_clears_its_diagnostics() {
  let mut server = initialized_server();
  open(&mut server, "f[1, 2");
  let notifications = server.handle_message(&json!({
    "jsonrpc": "2.0",
    "method": "textDocument/didClose",
    "params": { "textDocument": { "uri": "file:///test.wls" } },
  }));
  assert!(diagnostics_of(&notifications).is_empty());
  // The document is gone, so requests about it fail with invalid params.
  let responses = server.handle_message(&json!({
    "jsonrpc": "2.0",
    "id": 3,
    "method": "textDocument/hover",
    "params": position_params(0, 0),
  }));
  assert_eq!(responses[0]["error"]["code"], json!(-32602));
}

#[test]
fn diagnostics_report_unsupported_builtins_with_a_range() {
  let mut server = initialized_server();
  let notifications = open(&mut server, "x = 1;\nWordData[\"hello\"]\n");
  let diagnostics = diagnostics_of(&notifications);
  assert_eq!(diagnostics.len(), 1);
  assert_eq!(diagnostics[0]["severity"], json!(2));
  assert_eq!(diagnostics[0]["code"], json!("unsupported"));
  assert_eq!(
    diagnostics[0]["range"],
    json!({
      "start": { "line": 1, "character": 0 },
      "end": { "line": 1, "character": 8 },
    })
  );
}

#[test]
fn hover_documents_a_builtin() {
  let mut server = initialized_server();
  open(&mut server, "Sin[x]");
  let result =
    request(&mut server, "textDocument/hover", &position_params(0, 1));
  let value = result["contents"]["value"].as_str().unwrap();
  assert_eq!(result["contents"]["kind"], json!("markdown"));
  assert!(value.contains("### Sin"), "unexpected hover: {value}");
  assert!(
    value.contains("Implemented in Woxi"),
    "unexpected hover: {value}"
  );
  assert_eq!(
    result["range"],
    json!({
      "start": { "line": 0, "character": 0 },
      "end": { "line": 0, "character": 3 },
    })
  );
}

#[test]
fn hover_says_when_a_builtin_is_unsupported() {
  let mut server = initialized_server();
  open(&mut server, "WordData[\"a\"]");
  let result =
    request(&mut server, "textDocument/hover", &position_params(0, 2));
  let value = result["contents"]["value"].as_str().unwrap();
  assert!(
    value.contains("Not supported by Woxi"),
    "unexpected hover: {value}"
  );
}

#[test]
fn hover_shows_a_definition_from_the_file() {
  let mut server = initialized_server();
  open(&mut server, "square[x_] := x^2\nsquare[4]\n");
  let result =
    request(&mut server, "textDocument/hover", &position_params(1, 2));
  let value = result["contents"]["value"].as_str().unwrap();
  assert!(
    value.contains("square[x_] := x^2"),
    "unexpected hover: {value}"
  );
}

#[test]
fn hover_on_an_unknown_symbol_is_null() {
  let mut server = initialized_server();
  open(&mut server, "someUnknownName");
  let result =
    request(&mut server, "textDocument/hover", &position_params(0, 3));
  assert_eq!(result, json!(null));
  // So is hover on whitespace.
  open(&mut server, "  ");
  let result =
    request(&mut server, "textDocument/hover", &position_params(0, 1));
  assert_eq!(result, json!(null));
}

#[test]
fn completion_suggests_builtins_for_a_prefix() {
  let mut server = initialized_server();
  open(&mut server, "StringJo");
  let result = request(
    &mut server,
    "textDocument/completion",
    &position_params(0, 8),
  );
  let labels: Vec<&str> = result["items"]
    .as_array()
    .unwrap()
    .iter()
    .map(|item| item["label"].as_str().unwrap())
    .collect();
  assert!(labels.contains(&"StringJoin"), "got: {labels:?}");
  assert!(
    !labels.contains(&"Sin"),
    "prefix was not applied: {labels:?}"
  );
}

#[test]
fn completion_is_case_insensitive_and_lists_the_file_own_symbols_first() {
  let mut server = initialized_server();
  open(&mut server, "myHelper[x_] := x\nmyh");
  let result = request(
    &mut server,
    "textDocument/completion",
    &position_params(1, 3),
  );
  let items = result["items"].as_array().unwrap();
  assert_eq!(items[0]["label"], json!("myHelper"));
  assert_eq!(items[0]["detail"], json!("defined in this file"));
  assert!(
    items[0]["documentation"]["value"]
      .as_str()
      .unwrap()
      .contains("myHelper[x_] := x")
  );
}

#[test]
fn completion_marks_unsupported_builtins() {
  let mut server = initialized_server();
  open(&mut server, "WordDat");
  let result = request(
    &mut server,
    "textDocument/completion",
    &position_params(0, 7),
  );
  let item = result["items"]
    .as_array()
    .unwrap()
    .iter()
    .find(|item| item["label"] == json!("WordData"))
    .expect("WordData should be suggested");
  assert!(
    item["detail"]
      .as_str()
      .unwrap()
      .starts_with("not supported"),
    "unexpected detail: {}",
    item["detail"]
  );
}

#[test]
fn completion_without_a_prefix_offers_a_truncated_list() {
  let mut server = initialized_server();
  open(&mut server, "");
  let result = request(
    &mut server,
    "textDocument/completion",
    &position_params(0, 0),
  );
  assert_eq!(result["isIncomplete"], json!(true));
  let items = result["items"].as_array().unwrap();
  assert_eq!(items.len(), 1000);
  // The truncated list keeps the symbols Woxi can actually evaluate.
  assert!(
    items
      .iter()
      .all(|item| item["detail"].as_str().unwrap_or_default()
        != "not implemented — ")
  );
}

#[test]
fn go_to_definition_finds_every_clause() {
  let mut server = initialized_server();
  open(&mut server, "fib[0] = 0;\nfib[1] = 1;\nfib[3]\n");
  let result = request(
    &mut server,
    "textDocument/definition",
    &position_params(2, 1),
  );
  let locations = result.as_array().unwrap();
  assert_eq!(locations.len(), 2);
  assert_eq!(locations[0]["uri"], json!("file:///test.wls"));
  assert_eq!(locations[0]["range"]["start"]["line"], json!(0));
  assert_eq!(locations[1]["range"]["start"]["line"], json!(1));
}

#[test]
fn go_to_definition_of_an_undefined_symbol_is_null() {
  let mut server = initialized_server();
  open(&mut server, "Sin[x]");
  let result = request(
    &mut server,
    "textDocument/definition",
    &position_params(0, 1),
  );
  assert_eq!(result, json!(null));
}

#[test]
fn references_find_every_occurrence() {
  let mut server = initialized_server();
  open(&mut server, "x = 1;\ny = x + x;\n");
  let mut params = position_params(1, 4);
  params["context"] = json!({ "includeDeclaration": true });
  let result = request(&mut server, "textDocument/references", &params);
  assert_eq!(result.as_array().unwrap().len(), 3);

  let mut params = position_params(1, 4);
  params["context"] = json!({ "includeDeclaration": false });
  let result = request(&mut server, "textDocument/references", &params);
  let locations = result.as_array().unwrap();
  assert_eq!(locations.len(), 2);
  assert!(
    locations
      .iter()
      .all(|location| location["range"]["start"]["line"] == json!(1))
  );
}

#[test]
fn document_highlight_marks_the_definition_as_a_write() {
  let mut server = initialized_server();
  open(&mut server, "x = 1;\ny = x;\n");
  let result = request(
    &mut server,
    "textDocument/documentHighlight",
    &position_params(0, 0),
  );
  let highlights = result.as_array().unwrap();
  assert_eq!(highlights.len(), 2);
  assert_eq!(highlights[0]["kind"], json!(3));
  assert_eq!(highlights[1]["kind"], json!(1));
}

#[test]
fn the_outline_lists_top_level_definitions_only() {
  let mut server = initialized_server();
  open(
    &mut server,
    "value = 42;\nf[x_] := Module[{local = 1}, local + x];\n",
  );
  let result = request(
    &mut server,
    "textDocument/documentSymbol",
    &json!({ "textDocument": { "uri": "file:///test.wls" } }),
  );
  let symbols = result.as_array().unwrap();
  assert_eq!(symbols.len(), 2);
  assert_eq!(symbols[0]["name"], json!("value"));
  assert_eq!(symbols[0]["kind"], json!(13));
  assert_eq!(symbols[1]["name"], json!("f"));
  assert_eq!(symbols[1]["kind"], json!(12));
  // A hierarchy-capable client gets the nested shape.
  assert!(symbols[0]["selectionRange"].is_object());
  assert_eq!(symbols[1]["range"]["end"]["line"], json!(1));
}

#[test]
fn a_client_without_hierarchy_support_gets_symbol_information() {
  let mut server = Server::new();
  server.handle_message(&json!({
    "jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {},
  }));
  open(&mut server, "value = 42;\n");
  let result = request(
    &mut server,
    "textDocument/documentSymbol",
    &json!({ "textDocument": { "uri": "file:///test.wls" } }),
  );
  let symbols = result.as_array().unwrap();
  assert_eq!(symbols[0]["location"]["uri"], json!("file:///test.wls"));
  assert!(symbols[0].get("selectionRange").is_none());
}

#[test]
fn positions_in_lines_with_wide_characters_resolve_correctly() {
  let mut server = initialized_server();
  // "√" occupies one UTF-16 code unit, so `Sin` starts at character 12.
  open(&mut server, "x = \"√√√\"; Sin[1]\n");
  let result =
    request(&mut server, "textDocument/hover", &position_params(0, 12));
  let value = result["contents"]["value"].as_str().unwrap();
  assert!(value.contains("### Sin"), "unexpected hover: {value}");
}
