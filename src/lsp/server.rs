//! The language server itself: it turns LSP requests into answers derived
//! from [`super::analysis`] and the built-in function registry.
//!
//! [`Server::handle_message`] is a pure function of the server state — it
//! takes one incoming message and returns the messages to send back — so
//! the whole protocol surface is testable without spawning a process.
//! [`run_stdio`] is the thin loop that connects it to stdin/stdout.

use serde_json::{Value, json};
use std::collections::HashMap;

use super::analysis::{
  Definition, DefinitionKind, LineIndex, SEMANTIC_TOKEN_MODIFIERS,
  SEMANTIC_TOKEN_TYPES, Severity, Token, find_definitions, occurrences,
  spelling_suggestion, symbol_at, tokenize,
};
use super::protocol::{
  INVALID_PARAMS, INVALID_REQUEST, METHOD_NOT_FOUND, SERVER_NOT_INITIALIZED,
};
use crate::evaluator::functions::{
  ImplementationStatus, all_builtin_symbol_names, get_builtin_function_info,
  get_doc_url, implementation_status,
};

/// Largest number of completion items sent in one response. Editors filter
/// client-side as the user keeps typing, so a truncated (`isIncomplete`)
/// list is refreshed rather than wrong.
const MAX_COMPLETION_ITEMS: usize = 1000;

/// `CompletionItemKind` / `SymbolKind` values from the specification.
const COMPLETION_KIND_FUNCTION: i64 = 3;
const COMPLETION_KIND_VARIABLE: i64 = 6;
const SYMBOL_KIND_FUNCTION: i64 = 12;
const SYMBOL_KIND_VARIABLE: i64 = 13;

/// `CodeActionKind` values from the specification.
const QUICK_FIX: &str = "quickfix";
const SOURCE_FIX_ALL: &str = "source.fixAll";

/// `DocumentHighlightKind` values from the specification.
const HIGHLIGHT_KIND_TEXT: i64 = 1;
const HIGHLIGHT_KIND_WRITE: i64 = 3;

/// An open document plus everything the server derives from its text.
pub struct Document {
  pub version: i64,
  pub text: String,
  pub tokens: Vec<Token>,
  pub definitions: Vec<Definition>,
  pub line_index: LineIndex,
}

impl Document {
  pub fn new(text: String, version: i64) -> Self {
    let tokens = tokenize(&text);
    let definitions = find_definitions(&text, &tokens);
    let line_index = LineIndex::new(&text);
    Self {
      version,
      text,
      tokens,
      definitions,
      line_index,
    }
  }

  /// An LSP `Range` for a byte range of this document.
  fn range(&self, (start, end): (usize, usize)) -> Value {
    let (start_line, start_character) =
      self.line_index.position(&self.text, start);
    let (end_line, end_character) = self.line_index.position(&self.text, end);
    json!({
      "start": { "line": start_line, "character": start_character },
      "end": { "line": end_line, "character": end_character },
    })
  }

  /// The byte offset an LSP `Position` refers to.
  fn offset_of(&self, position: &Value) -> usize {
    let line = position.get("line").and_then(Value::as_u64).unwrap_or(0);
    let character = position
      .get("character")
      .and_then(Value::as_u64)
      .unwrap_or(0);
    self
      .line_index
      .offset(&self.text, line as u32, character as u32)
  }
}

/// Where a session stands in the protocol's lifecycle.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Lifecycle {
  /// Before `initialize`, which is the only request answered here.
  Uninitialized,
  /// Initialized and serving requests.
  Running,
  /// `shutdown` was received; only `exit` may still follow.
  ShuttingDown,
  /// `exit` was received; the process should stop.
  Exited,
}

/// State of one language server session.
pub struct Server {
  documents: HashMap<String, Document>,
  lifecycle: Lifecycle,
  /// Whether the client asked for a shutdown before exiting, which decides
  /// the process's exit code.
  clean_shutdown: bool,
  /// Whether the client understands the nested `DocumentSymbol` response.
  /// Clients that do not get the older flat `SymbolInformation` list.
  hierarchical_symbols: bool,
}

impl Default for Server {
  fn default() -> Self {
    Self::new()
  }
}

impl Server {
  pub fn new() -> Self {
    Self {
      documents: HashMap::new(),
      lifecycle: Lifecycle::Uninitialized,
      clean_shutdown: false,
      hierarchical_symbols: false,
    }
  }

  /// True once the client sent `exit`; the process should stop then.
  pub fn exit_requested(&self) -> bool {
    self.lifecycle == Lifecycle::Exited
  }

  /// True once the client sent `shutdown`. Exiting without it is an error
  /// the process reports through its exit code.
  pub fn shutdown_requested(&self) -> bool {
    self.clean_shutdown
  }

  /// Handle one incoming message, returning the messages to send back.
  ///
  /// Responses come first, followed by any notifications the message
  /// triggered (e.g. diagnostics after a document changed).
  pub fn handle_message(&mut self, message: &Value) -> Vec<Value> {
    let Some(method) = message.get("method").and_then(Value::as_str) else {
      // No method: this is a response to a request the server sent. The
      // server never sends requests, so there is nothing to do.
      return Vec::new();
    };
    let params = message.get("params").cloned().unwrap_or(Value::Null);

    match message.get("id") {
      Some(id) => {
        let result = self.handle_request(method, &params);
        vec![match result {
          Ok(result) => json!({
            "jsonrpc": "2.0", "id": id.clone(), "result": result,
          }),
          Err((code, message)) => json!({
            "jsonrpc": "2.0",
            "id": id.clone(),
            "error": { "code": code, "message": message },
          }),
        }]
      }
      None => self.handle_notification(method, &params),
    }
  }

  fn handle_request(
    &mut self,
    method: &str,
    params: &Value,
  ) -> Result<Value, (i64, String)> {
    match self.lifecycle {
      Lifecycle::Uninitialized if method != "initialize" => {
        return Err((
          SERVER_NOT_INITIALIZED,
          "expected an initialize request first".to_string(),
        ));
      }
      Lifecycle::Running if method == "initialize" => {
        return Err((
          INVALID_REQUEST,
          "server is already initialized".to_string(),
        ));
      }
      Lifecycle::ShuttingDown | Lifecycle::Exited => {
        return Err((INVALID_REQUEST, "server has been shut down".to_string()));
      }
      _ => {}
    }

    match method {
      "initialize" => {
        self.lifecycle = Lifecycle::Running;
        self.hierarchical_symbols = params
          .pointer("/capabilities/textDocument/documentSymbol/hierarchicalDocumentSymbolSupport")
          .and_then(Value::as_bool)
          .unwrap_or(false);
        Ok(initialize_result())
      }
      "shutdown" => {
        self.lifecycle = Lifecycle::ShuttingDown;
        self.clean_shutdown = true;
        Ok(Value::Null)
      }
      "textDocument/hover" => self.hover(params),
      "textDocument/completion" => self.completion(params),
      "textDocument/definition" => self.definition(params),
      "textDocument/references" => self.references(params),
      "textDocument/documentHighlight" => self.document_highlight(params),
      "textDocument/documentSymbol" => self.document_symbol(params),
      "textDocument/semanticTokens/full" => self.semantic_tokens(params, false),
      "textDocument/semanticTokens/range" => self.semantic_tokens(params, true),
      "textDocument/formatting" => self.formatting(params, false),
      "textDocument/rangeFormatting" => self.formatting(params, true),
      "textDocument/codeAction" => self.code_action(params),
      _ => Err((METHOD_NOT_FOUND, format!("unsupported request: {method}"))),
    }
  }

  fn handle_notification(
    &mut self,
    method: &str,
    params: &Value,
  ) -> Vec<Value> {
    match method {
      "exit" => {
        self.lifecycle = Lifecycle::Exited;
        Vec::new()
      }
      "textDocument/didOpen" => {
        let Some(uri) =
          params.pointer("/textDocument/uri").and_then(Value::as_str)
        else {
          return Vec::new();
        };
        let text = params
          .pointer("/textDocument/text")
          .and_then(Value::as_str)
          .unwrap_or_default()
          .to_string();
        let version = params
          .pointer("/textDocument/version")
          .and_then(Value::as_i64)
          .unwrap_or(0);
        let uri = uri.to_string();
        self
          .documents
          .insert(uri.clone(), Document::new(text, version));
        vec![self.diagnostics_notification(&uri)]
      }
      "textDocument/didChange" => {
        let Some(uri) =
          params.pointer("/textDocument/uri").and_then(Value::as_str)
        else {
          return Vec::new();
        };
        let uri = uri.to_string();
        // The server only advertises full document sync, so the last
        // content change carries the whole new text.
        let Some(text) = params
          .get("contentChanges")
          .and_then(Value::as_array)
          .and_then(|changes| changes.last())
          .and_then(|change| change.get("text"))
          .and_then(Value::as_str)
        else {
          return Vec::new();
        };
        let version = params
          .pointer("/textDocument/version")
          .and_then(Value::as_i64)
          .unwrap_or(0);
        self
          .documents
          .insert(uri.clone(), Document::new(text.to_string(), version));
        vec![self.diagnostics_notification(&uri)]
      }
      "textDocument/didClose" => {
        let Some(uri) =
          params.pointer("/textDocument/uri").and_then(Value::as_str)
        else {
          return Vec::new();
        };
        let uri = uri.to_string();
        self.documents.remove(&uri);
        // Clear what was published for the file while it was open.
        vec![json!({
          "jsonrpc": "2.0",
          "method": "textDocument/publishDiagnostics",
          "params": { "uri": uri, "diagnostics": [] },
        })]
      }
      // Notifications with nothing to do, listed so that a genuinely
      // unknown one is still ignored silently (the specification forbids
      // responding to notifications at all).
      _ => Vec::new(),
    }
  }

  /// The document a request's `textDocument/uri` refers to.
  fn document(&self, params: &Value) -> Result<&Document, (i64, String)> {
    let uri = params
      .pointer("/textDocument/uri")
      .and_then(Value::as_str)
      .ok_or_else(|| {
        (INVALID_PARAMS, "missing textDocument.uri".to_string())
      })?;
    self
      .documents
      .get(uri)
      .ok_or_else(|| (INVALID_PARAMS, format!("document is not open: {uri}")))
  }

  /// The document and the byte offset of the request's `position`.
  fn document_and_offset(
    &self,
    params: &Value,
  ) -> Result<(&Document, usize), (i64, String)> {
    let document = self.document(params)?;
    let position = params
      .get("position")
      .ok_or_else(|| (INVALID_PARAMS, "missing position".to_string()))?;
    Ok((document, document.offset_of(position)))
  }

  /// A `textDocument/publishDiagnostics` notification for `uri`.
  fn diagnostics_notification(&self, uri: &str) -> Value {
    let Some(document) = self.documents.get(uri) else {
      return json!({
        "jsonrpc": "2.0",
        "method": "textDocument/publishDiagnostics",
        "params": { "uri": uri, "diagnostics": [] },
      });
    };
    let diagnostics: Vec<Value> = super::analysis::diagnostics(
      &document.text,
      &document.tokens,
      &document.definitions,
    )
    .into_iter()
    .map(|diagnostic| {
      json!({
        "range": document.range(diagnostic.range),
        "severity": match diagnostic.severity {
          Severity::Error => 1,
          Severity::Warning => 2,
          Severity::Information => 3,
        },
        "code": diagnostic.code,
        "source": "woxi",
        "message": diagnostic.message,
      })
    })
    .collect();
    json!({
      "jsonrpc": "2.0",
      "method": "textDocument/publishDiagnostics",
      "params": {
        "uri": uri,
        "version": document.version,
        "diagnostics": diagnostics,
      },
    })
  }

  fn hover(&self, params: &Value) -> Result<Value, (i64, String)> {
    let (document, offset) = self.document_and_offset(params)?;
    let Some(token) = symbol_at(&document.tokens, offset) else {
      return Ok(Value::Null);
    };
    let name = token.text(&document.text);
    let Some(markdown) = symbol_documentation(document, name) else {
      return Ok(Value::Null);
    };
    Ok(json!({
      "contents": { "kind": "markdown", "value": markdown },
      "range": document.range((token.start, token.end)),
    }))
  }

  fn completion(&self, params: &Value) -> Result<Value, (i64, String)> {
    let (document, offset) = self.document_and_offset(params)?;
    // The word being typed: the part of the symbol left of the cursor.
    let prefix = symbol_at(&document.tokens, offset)
      .map(|token| &document.text[token.start..offset.max(token.start)])
      .unwrap_or_default();
    let lowercase_prefix = prefix.to_lowercase();

    let mut items: Vec<Value> = Vec::new();
    // Symbols the document itself defines come first: they are what the
    // user is most likely to mean in their own file.
    let mut seen: Vec<&str> = Vec::new();
    for definition in &document.definitions {
      if !definition
        .name
        .to_lowercase()
        .starts_with(&lowercase_prefix)
        || seen.contains(&definition.name.as_str())
      {
        continue;
      }
      seen.push(&definition.name);
      items.push(json!({
        "label": definition.name,
        "kind": match definition.kind {
          DefinitionKind::Function => COMPLETION_KIND_FUNCTION,
          DefinitionKind::Variable => COMPLETION_KIND_VARIABLE,
        },
        "detail": "defined in this file",
        "documentation": {
          "kind": "markdown",
          "value": definition_markdown(document, &definition.name),
        },
        "sortText": format!("0{}", definition.name),
      }));
    }

    let mut builtins: Vec<&'static str> = all_builtin_symbol_names()
      .into_iter()
      .filter(|name| name.to_lowercase().starts_with(&lowercase_prefix))
      .collect();
    // Implemented symbols first, then partial ones, then the rest; each
    // group alphabetically.
    builtins.sort_by_key(|name| (status_rank(name), *name));
    let truncated = builtins.len() > MAX_COMPLETION_ITEMS;
    for name in builtins.into_iter().take(MAX_COMPLETION_ITEMS) {
      if seen.contains(&name) {
        continue;
      }
      items.push(json!({
        "label": name,
        "kind": COMPLETION_KIND_FUNCTION,
        "detail": completion_detail(name),
        "documentation": {
          "kind": "markdown",
          "value": builtin_markdown(name),
        },
        "sortText": format!("{}{name}", status_rank(name) + 1),
      }));
    }

    Ok(json!({ "isIncomplete": truncated, "items": items }))
  }

  fn definition(&self, params: &Value) -> Result<Value, (i64, String)> {
    let (document, offset) = self.document_and_offset(params)?;
    let uri = params
      .pointer("/textDocument/uri")
      .and_then(Value::as_str)
      .unwrap_or_default();
    let Some(token) = symbol_at(&document.tokens, offset) else {
      return Ok(Value::Null);
    };
    let name = token.text(&document.text);
    let locations: Vec<Value> = document
      .definitions
      .iter()
      .filter(|definition| definition.name == name)
      .map(|definition| {
        json!({
          "uri": uri,
          "range": document.range(definition.name_range),
        })
      })
      .collect();
    if locations.is_empty() {
      return Ok(Value::Null);
    }
    Ok(Value::Array(locations))
  }

  fn references(&self, params: &Value) -> Result<Value, (i64, String)> {
    let (document, offset) = self.document_and_offset(params)?;
    let uri = params
      .pointer("/textDocument/uri")
      .and_then(Value::as_str)
      .unwrap_or_default();
    let include_declaration = params
      .pointer("/context/includeDeclaration")
      .and_then(Value::as_bool)
      .unwrap_or(true);
    let Some(token) = symbol_at(&document.tokens, offset) else {
      return Ok(Value::Array(Vec::new()));
    };
    let name = token.text(&document.text);
    let locations: Vec<Value> =
      occurrences(&document.text, &document.tokens, name)
        .into_iter()
        .filter(|occurrence| {
          include_declaration || !is_definition_name(document, *occurrence)
        })
        .map(|occurrence| {
          json!({
            "uri": uri,
            "range": document.range((occurrence.start, occurrence.end)),
          })
        })
        .collect();
    Ok(Value::Array(locations))
  }

  fn document_highlight(&self, params: &Value) -> Result<Value, (i64, String)> {
    let (document, offset) = self.document_and_offset(params)?;
    let Some(token) = symbol_at(&document.tokens, offset) else {
      return Ok(Value::Array(Vec::new()));
    };
    let name = token.text(&document.text);
    let highlights: Vec<Value> =
      occurrences(&document.text, &document.tokens, name)
        .into_iter()
        .map(|occurrence| {
          json!({
            "range": document.range((occurrence.start, occurrence.end)),
            "kind": if is_definition_name(document, occurrence) {
              HIGHLIGHT_KIND_WRITE
            } else {
              HIGHLIGHT_KIND_TEXT
            },
          })
        })
        .collect();
    Ok(Value::Array(highlights))
  }

  fn document_symbol(&self, params: &Value) -> Result<Value, (i64, String)> {
    let document = self.document(params)?;
    let uri = params
      .pointer("/textDocument/uri")
      .and_then(Value::as_str)
      .unwrap_or_default();
    let symbols: Vec<Value> = document
      .definitions
      .iter()
      // Only top-level definitions: a `Module[{i = 0}, …]` local is not
      // part of the file's outline.
      .filter(|definition| definition.depth == 0)
      .map(|definition| {
        let kind = match definition.kind {
          DefinitionKind::Function => SYMBOL_KIND_FUNCTION,
          DefinitionKind::Variable => SYMBOL_KIND_VARIABLE,
        };
        if self.hierarchical_symbols {
          json!({
            "name": definition.name,
            "kind": kind,
            "range": document.range(definition.full_range),
            "selectionRange": document.range(definition.name_range),
          })
        } else {
          json!({
            "name": definition.name,
            "kind": kind,
            "location": {
              "uri": uri,
              "range": document.range(definition.full_range),
            },
          })
        }
      })
      .collect();
    Ok(Value::Array(symbols))
  }

  /// `textDocument/semanticTokens/full` and `…/range`: the whole file's
  /// highlighting, or only the part of it the request asked about.
  fn semantic_tokens(
    &self,
    params: &Value,
    ranged: bool,
  ) -> Result<Value, (i64, String)> {
    let document = self.document(params)?;
    let tokens =
      super::analysis::semantic_tokens(&document.text, &document.definitions);
    let limit = if ranged {
      Some(byte_range(document, params)?)
    } else {
      None
    };
    let mut data: Vec<u32> = Vec::new();
    let (mut previous_line, mut previous_character) = (0_u32, 0_u32);
    for token in tokens {
      if limit.is_some_and(|(start, end)| {
        token.range.1 <= start || token.range.0 >= end
      }) {
        continue;
      }
      let (line, character) =
        document.line_index.position(&document.text, token.range.0);
      let (_, end_character) =
        document.line_index.position(&document.text, token.range.1);
      // The tokens come in source order, so these never wrap; saturating
      // rather than panicking is the right way for a server to be wrong.
      let delta_line = line.saturating_sub(previous_line);
      let delta_character = if delta_line == 0 {
        character.saturating_sub(previous_character)
      } else {
        character
      };
      data.extend([
        delta_line,
        delta_character,
        end_character.saturating_sub(character),
        token.token_type,
        token.modifiers,
      ]);
      (previous_line, previous_character) = (line, character);
    }
    Ok(json!({ "data": data }))
  }

  /// `textDocument/formatting` and `…/rangeFormatting`.
  ///
  /// Both reformat the whole document — indentation depends on what came
  /// before the range, so a range cannot be formatted on its own — and a
  /// range request then reports only the edits of the lines it covers.
  fn formatting(
    &self,
    params: &Value,
    ranged: bool,
  ) -> Result<Value, (i64, String)> {
    let document = self.document(params)?;
    let options = super::format::Options {
      tab_size: params
        .pointer("/options/tabSize")
        .and_then(Value::as_u64)
        .map_or(super::format::Options::default().tab_size, |size| {
          size.max(1) as usize
        }),
      insert_spaces: params
        .pointer("/options/insertSpaces")
        .and_then(Value::as_bool)
        .unwrap_or(true),
    };
    let formatted = super::format::format(&document.text, &options);
    let lines = if ranged {
      let range = params
        .get("range")
        .ok_or_else(|| (INVALID_PARAMS, "missing range".to_string()))?;
      let first = range
        .pointer("/start/line")
        .and_then(Value::as_u64)
        .unwrap_or(0) as u32;
      let last = range
        .pointer("/end/line")
        .and_then(Value::as_u64)
        .unwrap_or(u64::from(u32::MAX)) as u32;
      Some((first, last))
    } else {
      None
    };
    Ok(Value::Array(text_edits(document, &formatted, lines)))
  }

  /// `textDocument/codeAction`: the fixes offered for the requested range.
  ///
  /// The one thing a language server can fix in a language where every
  /// undefined symbol is legal is a name that was meant to be another:
  /// `Lenght[…]` for `Length[…]`.
  fn code_action(&self, params: &Value) -> Result<Value, (i64, String)> {
    let document = self.document(params)?;
    let uri = params
      .pointer("/textDocument/uri")
      .and_then(Value::as_str)
      .unwrap_or_default();
    let (start, end) = byte_range(document, params)?;
    let only: Vec<&str> = params
      .pointer("/context/only")
      .and_then(Value::as_array)
      .map(|kinds| kinds.iter().filter_map(Value::as_str).collect())
      .unwrap_or_default();

    // Every misspelling in the file, and the correction for it.
    let corrections: Vec<(super::analysis::Diagnostic, &'static str)> =
      super::analysis::diagnostics(
        &document.text,
        &document.tokens,
        &document.definitions,
      )
      .into_iter()
      .filter(|diagnostic| diagnostic.code == "spelling")
      .filter_map(|diagnostic| {
        let name = &document.text[diagnostic.range.0..diagnostic.range.1];
        let suggestion = spelling_suggestion(name)?;
        Some((diagnostic, suggestion))
      })
      .collect();

    let mut actions = Vec::new();
    if kind_requested(&only, QUICK_FIX) {
      for (diagnostic, suggestion) in &corrections {
        // Only what the cursor or selection actually touches.
        if diagnostic.range.1 < start || diagnostic.range.0 > end {
          continue;
        }
        let name = &document.text[diagnostic.range.0..diagnostic.range.1];
        actions.push(json!({
          "title": format!("Change `{name}` to `{suggestion}`"),
          "kind": QUICK_FIX,
          "diagnostics": [diagnostic_json(document, diagnostic)],
          "isPreferred": true,
          "edit": {
            "changes": {
              uri: [{
                "range": document.range(diagnostic.range),
                "newText": suggestion,
              }],
            },
          },
        }));
      }
    }
    // One action for the whole file, so a file full of the same typo is
    // fixed in a single step (and by an editor's "fix all on save").
    if kind_requested(&only, SOURCE_FIX_ALL) && corrections.len() > 1 {
      let edits: Vec<Value> = corrections
        .iter()
        .map(|(diagnostic, suggestion)| {
          json!({
            "range": document.range(diagnostic.range),
            "newText": suggestion,
          })
        })
        .collect();
      actions.push(json!({
        "title": format!("Fix all {} spelling suggestions", edits.len()),
        "kind": SOURCE_FIX_ALL,
        "diagnostics": corrections
          .iter()
          .map(|(diagnostic, _)| diagnostic_json(document, diagnostic))
          .collect::<Vec<Value>>(),
        "edit": { "changes": { uri: edits } },
      }));
    }
    Ok(Value::Array(actions))
  }
}

/// The byte range a request's `range` covers.
fn byte_range(
  document: &Document,
  params: &Value,
) -> Result<(usize, usize), (i64, String)> {
  let range = params
    .get("range")
    .ok_or_else(|| (INVALID_PARAMS, "missing range".to_string()))?;
  let start = range
    .get("start")
    .map_or(0, |position| document.offset_of(position));
  let end = range
    .get("end")
    .map_or(document.text.len(), |position| document.offset_of(position));
  Ok((start.min(end), start.max(end)))
}

/// Whether an action of `kind` is one the request asked for. An empty
/// `only` list asks for everything; otherwise a kind matches when a
/// requested kind is it or a prefix of it, as the specification defines.
fn kind_requested(only: &[&str], kind: &str) -> bool {
  only.is_empty()
    || only.iter().any(|requested| {
      kind == *requested || kind.starts_with(&format!("{requested}."))
    })
}

/// The `Diagnostic` a code action refers back to, in the shape the client
/// was sent it in.
fn diagnostic_json(
  document: &Document,
  diagnostic: &super::analysis::Diagnostic,
) -> Value {
  json!({
    "range": document.range(diagnostic.range),
    "severity": match diagnostic.severity {
      Severity::Error => 1,
      Severity::Warning => 2,
      Severity::Information => 3,
    },
    "code": diagnostic.code,
    "source": "woxi",
    "message": diagnostic.message,
  })
}

/// The edits turning `document` into `formatted`, restricted to the lines
/// `first..=last` when a line range is given.
///
/// The formatter keeps a document's lines, so the two texts line up and
/// only the lines that actually changed are sent — an editor then leaves
/// the cursor and the folds of every untouched line alone. Should that
/// ever not hold, the whole document is replaced instead.
fn text_edits(
  document: &Document,
  formatted: &str,
  lines: Option<(u32, u32)>,
) -> Vec<Value> {
  if formatted == document.text {
    return Vec::new();
  }
  let old: Vec<&str> = document.text.split('\n').collect();
  let new: Vec<&str> = formatted.split('\n').collect();
  if old.len() != new.len() {
    let end = document
      .line_index
      .position(&document.text, document.text.len());
    return vec![json!({
      "range": {
        "start": { "line": 0, "character": 0 },
        "end": { "line": end.0, "character": end.1 },
      },
      "newText": formatted,
    })];
  }
  let mut edits = Vec::new();
  for (line, (before, after)) in old.iter().zip(new.iter()).enumerate() {
    if before == after {
      continue;
    }
    let line = line as u32;
    if lines.is_some_and(|(first, last)| line < first || line > last) {
      continue;
    }
    edits.push(json!({
      "range": {
        "start": { "line": line, "character": 0 },
        "end": {
          "line": line,
          "character": before.chars().map(char::len_utf16).sum::<usize>(),
        },
      },
      "newText": after,
    }));
  }
  edits
}

/// Sort key putting implemented symbols before everything Woxi cannot run.
fn status_rank(name: &str) -> u8 {
  match implementation_status(name) {
    Some(ImplementationStatus::Implemented) => 0,
    Some(ImplementationStatus::Partial) => 1,
    Some(ImplementationStatus::NotImplemented) => 2,
    _ => 3,
  }
}

/// A short line describing a built-in in a completion list.
fn completion_detail(name: &str) -> String {
  let status = match implementation_status(name) {
    Some(ImplementationStatus::Partial) => "partially supported — ",
    Some(ImplementationStatus::NotImplemented) => "not implemented — ",
    Some(ImplementationStatus::NotPlanned) => "not supported — ",
    _ => "",
  };
  let description = get_builtin_function_info(name)
    .map_or("built-in Wolfram Language symbol", |info| info.description);
  format!("{status}{description}")
}

/// Documentation for a built-in symbol, as Markdown.
fn builtin_markdown(name: &str) -> String {
  let mut markdown = format!("### {name}\n");
  if let Some(info) = get_builtin_function_info(name)
    && !info.description.is_empty()
  {
    markdown.push_str(&format!("\n{}\n", info.description));
  }
  let status = match implementation_status(name) {
    Some(ImplementationStatus::Implemented) => Some("Implemented in Woxi."),
    Some(ImplementationStatus::Partial) => {
      Some("Partially implemented in Woxi — some forms are missing.")
    }
    Some(ImplementationStatus::NotImplemented) => {
      Some("Not implemented in Woxi yet.")
    }
    Some(ImplementationStatus::NotPlanned) => Some("Not supported by Woxi."),
    None => None,
  };
  if let Some(status) = status {
    markdown.push_str(&format!("\n_{status}_\n"));
  }
  if let Some(url) = get_doc_url(name) {
    markdown.push_str(&format!("\n[Documentation]({url})\n"));
  }
  markdown
}

/// Documentation for a symbol defined in `document`, as Markdown: its
/// definitions, quoted from the source.
fn definition_markdown(document: &Document, name: &str) -> String {
  let mut markdown = format!("### {name}\n\nDefined in this file:\n");
  for definition in document
    .definitions
    .iter()
    .filter(|definition| definition.name == name)
  {
    let (start, end) = definition.full_range;
    let source = document.text[start..end].trim_end();
    markdown.push_str(&format!("\n```wolfram\n{source}\n```\n"));
  }
  markdown
}

/// Hover documentation for `name`: its definitions in the file if it has
/// any, otherwise the built-in registry's entry.
fn symbol_documentation(document: &Document, name: &str) -> Option<String> {
  if document
    .definitions
    .iter()
    .any(|definition| definition.name == name)
  {
    return Some(definition_markdown(document, name));
  }
  if implementation_status(name).is_some() {
    return Some(builtin_markdown(name));
  }
  None
}

/// Whether `token` is the symbol a definition assigns to.
fn is_definition_name(document: &Document, token: Token) -> bool {
  document
    .definitions
    .iter()
    .any(|definition| definition.name_range == (token.start, token.end))
}

/// The `InitializeResult` describing what this server can do.
fn initialize_result() -> Value {
  json!({
    "capabilities": {
      // Full sync: documents are small and re-tokenising one is far
      // cheaper than tracking incremental edits correctly.
      "textDocumentSync": { "openClose": true, "change": 1 },
      "hoverProvider": true,
      "completionProvider": { "resolveProvider": false },
      "definitionProvider": true,
      "referencesProvider": true,
      "documentHighlightProvider": true,
      "documentSymbolProvider": true,
      "semanticTokensProvider": {
        "legend": {
          "tokenTypes": SEMANTIC_TOKEN_TYPES,
          "tokenModifiers": SEMANTIC_TOKEN_MODIFIERS,
        },
        "full": true,
        "range": true,
      },
      "documentFormattingProvider": true,
      "documentRangeFormattingProvider": true,
      "codeActionProvider": {
        "codeActionKinds": [QUICK_FIX, SOURCE_FIX_ALL],
      },
    },
    "serverInfo": {
      "name": "woxi",
      "version": env!("CARGO_PKG_VERSION"),
    },
  })
}
