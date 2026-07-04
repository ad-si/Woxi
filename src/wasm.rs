use std::cell::RefCell;
use std::collections::HashMap;

use wasm_bindgen::prelude::*;

use crate::{clear_state, interpret, interpret_with_stdout};

thread_local! {
  // Host-registered in-memory files. The browser has no local filesystem,
  // so `Import["name"]` resolves against this store instead (see
  // `import_virtual` in evaluator/dispatch/image_functions.rs).
  static VIRTUAL_FILES: RefCell<HashMap<String, Vec<u8>>> =
    RefCell::new(HashMap::new());
}

/// Register (or replace) an in-memory file so `Import["name"]` can read it
/// in the browser.
#[wasm_bindgen]
pub fn set_virtual_file(name: &str, data: &[u8]) {
  VIRTUAL_FILES.with(|files| {
    files.borrow_mut().insert(name.to_string(), data.to_vec());
  });
}

/// Remove all host-registered in-memory files.
#[wasm_bindgen]
pub fn clear_virtual_files() {
  VIRTUAL_FILES.with(|files| files.borrow_mut().clear());
}

/// Look up a host-registered file by exact name, falling back to the
/// basename so `Import["/tmp/data.csv"]` still finds a file registered
/// as "data.csv".
pub fn virtual_file(path: &str) -> Option<Vec<u8>> {
  VIRTUAL_FILES.with(|files| {
    let files = files.borrow();
    if let Some(data) = files.get(path) {
      return Some(data.clone());
    }
    let base = path.rsplit('/').next().unwrap_or(path);
    files.get(base).cloned()
  })
}

// Import a JS-provided function that fetches a URL and returns its content
// as a base64-encoded string.  The host (worker.js / kernel) must supply this.
#[wasm_bindgen]
extern "C" {
  #[wasm_bindgen(js_name = "__woxi_fetch_url", catch)]
  fn woxi_fetch_url(url: &str) -> Result<String, JsValue>;

  // Host-provided panic reporter. The panic hook calls this *before* the
  // resulting `unreachable` trap fires, so the host can capture a meaningful
  // message and auto-restart the instance. `catch` makes the call a no-op
  // when the host hasn't installed `globalThis.__woxi_report_panic`.
  #[wasm_bindgen(js_name = "__woxi_report_panic", catch)]
  fn report_panic(msg: &str) -> Result<(), JsValue>;
}

/// Download a URL and decode the image bytes (WASM).
/// Returns the image as an Expr::Image.
pub fn import_image_from_url_wasm(
  url: &str,
) -> Result<crate::syntax::Expr, crate::InterpreterError> {
  let b64 = woxi_fetch_url(url).map_err(|e| {
    crate::InterpreterError::EvaluationError(format!(
      "Import: failed to fetch \"{}\": {:?}",
      url, e
    ))
  })?;
  if b64.is_empty() {
    return Err(crate::InterpreterError::EvaluationError(format!(
      "Import: empty response from \"{}\"",
      url
    )));
  }
  let bytes =
    base64::Engine::decode(&base64::engine::general_purpose::STANDARD, &b64)
      .map_err(|e| {
        crate::InterpreterError::EvaluationError(format!(
          "Import: failed to decode fetched data: {}",
          e
        ))
      })?;
  crate::functions::image_ast::import_image_from_bytes(&bytes)
}

/// Download a CSV file from a URL and parse it (WASM).
pub fn csv_import_from_url_wasm(
  url: &str,
  element: Option<&str>,
) -> Result<crate::syntax::Expr, crate::InterpreterError> {
  let b64 = woxi_fetch_url(url).map_err(|e| {
    crate::InterpreterError::EvaluationError(format!(
      "Import: failed to fetch \"{}\": {:?}",
      url, e
    ))
  })?;
  if b64.is_empty() {
    return Err(crate::InterpreterError::EvaluationError(format!(
      "Import: empty response from \"{}\"",
      url
    )));
  }
  let bytes =
    base64::Engine::decode(&base64::engine::general_purpose::STANDARD, &b64)
      .map_err(|e| {
        crate::InterpreterError::EvaluationError(format!(
          "Import: failed to decode fetched data: {}",
          e
        ))
      })?;
  let content = String::from_utf8(bytes).map_err(|e| {
    crate::InterpreterError::EvaluationError(format!(
      "Import: downloaded CSV is not valid UTF-8: {}",
      e
    ))
  })?;
  let rows = crate::functions::csv_ast::parse_csv(&content);
  Ok(crate::functions::csv_ast::csv_import_element(
    &rows, element,
  ))
}

#[wasm_bindgen(start)]
pub fn init() {
  // Install a panic hook that both logs the full panic (with a JS stack) to
  // the developer console and forwards the message to the host. A Rust panic
  // in WASM compiles to the `unreachable` trap: it aborts the current call
  // and leaves the module's mutable globals (notably the shadow-stack
  // pointer) in a half-updated state, so every subsequent call into the same
  // instance re-traps with a bare "unreachable". The host uses the forwarded
  // message to show a real error and reinstantiate a fresh instance.
  std::panic::set_hook(Box::new(|info| {
    console_error_panic_hook::hook(info);
    let _ = report_panic(&info.to_string());
  }));
}

/// Synchronously block the current (worker) thread for `secs` seconds.
/// Implemented as a busy-wait against `Date.now()` because
/// `std::thread::sleep` is unavailable on `wasm32-unknown-unknown`.
/// Burns one core for the duration but does not block the UI thread
/// when called from a Web Worker.
pub fn sleep_seconds(secs: f64) {
  if secs <= 0.0 {
    return;
  }
  let end = js_sys::Date::now() + secs * 1000.0;
  while js_sys::Date::now() < end {
    // busy-wait
  }
}

/// Evaluate a Wolfram Language expression and return the result.
/// If the expression produces Print output, it is prepended to the result.
#[wasm_bindgen]
pub fn evaluate(input: &str) -> String {
  match interpret_with_stdout(input) {
    Ok(result) => {
      let mut output = result.stdout;
      if result.result != "\0" {
        output.push_str(&result.result);
      }
      output
    }
    Err(e) => format!("Error: {e}"),
  }
}

/// Return the captured SVG graphics from the last `evaluate()` call, if any.
/// Returns an empty string when there is no graphics output.
#[wasm_bindgen]
pub fn get_graphics() -> String {
  crate::get_captured_graphics().unwrap_or_default()
}

/// Return the captured GraphicsBox expression from the last `evaluate()` call.
/// Returns an empty string when there is no graphics output.
#[wasm_bindgen]
pub fn get_graphicsbox() -> String {
  crate::get_captured_graphicsbox().unwrap_or_default()
}

/// Return the playable audio (base64-encoded) from the last `evaluate()`
/// call, if any. Returns an empty string when there is no sound.
#[wasm_bindgen]
pub fn get_sound() -> String {
  crate::get_captured_sound()
    .map(|audio| audio.base64)
    .unwrap_or_default()
}

/// Return warnings from the last `evaluate()` call as newline-separated text.
/// Returns an empty string when there are no warnings.
#[wasm_bindgen]
pub fn get_warnings() -> String {
  crate::get_captured_warnings().join("\n")
}

/// Return SVG rendering of the last text result (with superscripts etc.).
/// Returns an empty string when there is no output SVG (e.g. for Graphics results).
#[wasm_bindgen]
pub fn get_output_svg() -> String {
  crate::get_captured_output_svg().unwrap_or_default()
}

/// Clear all interpreter state (variables and function definitions).
#[wasm_bindgen]
pub fn clear() {
  clear_state();
}

/// Enable or disable dark mode for SVG output colors.
#[wasm_bindgen]
pub fn set_dark_mode(enabled: bool) {
  crate::set_dark_mode(enabled);
}

/// Evaluate all top-level statements and return a JSON array of output items.
/// Each item has a "type" field ("text", "graphics", "print", "warning", "error",
/// "manipulate") and corresponding content fields.
#[wasm_bindgen]
pub fn evaluate_all(input: &str) -> String {
  let statements = crate::split_into_statements(input);
  let mut items = Vec::new();
  for stmt in &statements {
    items.extend(evaluate_statement_items(stmt));
  }
  format!("[{}]", items.join(","))
}

/// Split `input` into top-level statements and return a JSON array of strings.
/// Lets the front-end loop one statement at a time so output (and side
/// effects like `Pause[n]`) appear progressively rather than batched.
#[wasm_bindgen]
pub fn split_statements(input: &str) -> String {
  let statements = crate::split_into_statements(input);
  let parts: Vec<String> = statements
    .iter()
    .map(|s| format!("\"{}\"", json_escape(s)))
    .collect();
  format!("[{}]", parts.join(","))
}

/// Evaluate a single statement and return a JSON array of output items
/// (same shape as `evaluate_all`'s elements). Use together with
/// `split_statements` for progressive output.
#[wasm_bindgen]
pub fn evaluate_statement(stmt: &str) -> String {
  let items = evaluate_statement_items(stmt);
  format!("[{}]", items.join(","))
}

/// Residual text of a graphics-carrying result once the graphics
/// placeholders are stripped. A result that is one `Legended[-Graphics-, …]`
/// wrapper (e.g. `PeriodicTablePlot["Phase"]`) displays entirely as its SVG
/// — the legend is baked into the rendered graphic — so it leaves no
/// residual text.
fn residual_text(result: &str) -> String {
  if result.starts_with("Legended[-Graphics") && result.ends_with(']') {
    return String::new();
  }
  result
    .replace("-Graphics-", "")
    .replace("-Graphics3D-", "")
    .replace("-Image-", "")
    .trim()
    .to_string()
}

/// Build the JSON output items produced by a single statement.
fn evaluate_statement_items(stmt: &str) -> Vec<String> {
  let mut items = Vec::new();
  match interpret_with_stdout(stmt) {
    Ok(result) => {
      // Print output
      let trimmed_stdout = result.stdout.trim_end();
      if !trimmed_stdout.is_empty() {
        items.push(json_output_item("print", trimmed_stdout, None));
      }

      // Warnings
      for w in &result.warnings {
        items.push(json_output_item("warning", w, None));
      }

      // If the statement is a top-level Manipulate[…] or a standalone
      // Control[…] call, emit a dedicated "manipulate" item so the frontend
      // can render interactive controls instead of the plain text echo. We
      // re-parse the source so we can inspect the held expression shape.
      if result.result != "\0"
        && let Some(item) = try_build_manipulate_item(stmt)
      {
        items.push(item);
        return items;
      }

      // Playable audio (Play[…] / Sound[…] / Audio[…]) — emit a dedicated
      // "sound" item carrying the base64 data so the frontend can render a
      // graphical audio player. The textual echo is suppressed in favor of it.
      if let Some(ref audio) = result.sound
        && result.result != "\0"
      {
        items.push(json_sound_item(audio));
        return items;
      }

      // Main result
      if let Some(ref svg) = result.graphics {
        // Only display graphics if output wasn't suppressed by trailing semicolon
        if result.result != "\0" {
          items.push(json_output_item("graphics", svg, None));
          // Check for non-graphics text mixed in
          let cleaned = residual_text(&result.result);
          if !cleaned.is_empty() && cleaned != "\0" {
            items.push(json_output_item("text", &cleaned, None));
          }
        }
      } else if result.result != "\0" {
        items.push(json_output_item(
          "text",
          &result.result,
          result.output_svg.as_deref(),
        ));
      }
    }
    Err(crate::InterpreterError::EmptyInput) => {
      // Function definitions etc. produce no output
    }
    Err(e) => {
      items.push(json_output_item("error", &format!("{e}"), None));
    }
  }
  items
}

/// Try to detect whether `stmt` is a top-level `Manipulate[…]` call and
/// build a JSON "manipulate" output item (spec + initial rendering).
/// Returns `None` if the statement isn't a well-formed Manipulate.
fn try_build_manipulate_item(stmt: &str) -> Option<String> {
  // Re-interpret the statement so we get the evaluated (but held)
  // Manipulate FunctionCall back as an Expr we can inspect. Manipulate
  // is HoldAll-ish (see functions::graphics::manipulate_ast), so its body
  // and variable symbols remain intact.
  let expr = crate::interpret_to_expr(stmt).ok()?;
  // A top-level Manipulate[…] or a standalone Control[…] both render as an
  // interactive control widget backed by a ManipulateSpec.
  let spec = crate::functions::graphics::extract_manipulate_spec(&expr)
    .or_else(|| crate::functions::graphics::extract_control_spec(&expr))?;

  // Produce an initial rendering by substituting initial values via Block.
  let bindings = crate::functions::graphics::manipulate_initial_bindings(&spec);
  let block_code = crate::functions::graphics::manipulate_block_code(
    &spec.body_code,
    &bindings,
  );

  let initial = match crate::interpret_with_stdout(&block_code) {
    Ok(r) => r,
    Err(_) => crate::InterpretResult {
      stdout: String::new(),
      result: String::new(),
      graphics: None,
      output_svg: None,
      sound: None,
      warnings: vec![],
    },
  };

  let spec_json = crate::functions::graphics::manipulate_spec_to_json(&spec);

  // Build initial-rendering JSON.
  let mut initial_parts: Vec<String> = Vec::new();
  if let Some(ref svg) = initial.graphics {
    initial_parts.push(format!(r#""svg":"{}""#, json_escape(svg)));
  }
  let cleaned_text = residual_text(&initial.result);
  if !cleaned_text.is_empty() && initial.graphics.is_none() {
    initial_parts.push(format!(r#""text":"{}""#, json_escape(&cleaned_text)));
    if let Some(ref output_svg) = initial.output_svg {
      initial_parts.push(format!(r#""textSvg":"{}""#, json_escape(output_svg)));
    }
  }
  let initial_json = format!("{{{}}}", initial_parts.join(","));

  Some(format!(
    r#"{{"type":"manipulate",{},"initial":{}}}"#,
    spec_json, initial_json
  ))
}

/// Evaluate a Manipulate body with a specific set of variable bindings
/// and return a single JSON output item representing the result.
///
/// `body` must be the body expression in InputForm (as produced by
/// `evaluate_all`'s manipulate item). `bindings_json` must be a JSON
/// object mapping variable names to InputForm-serialized values, e.g.
/// `{"a": "1.5", "b": "\"foo\""}`.
///
/// The result is a JSON object of the same shape as `initial` from
/// the evaluate_all manipulate item: `{"svg": "...", "text": "..."}`.
#[wasm_bindgen]
pub fn evaluate_manipulate(body: &str, bindings_json: &str) -> String {
  let bindings =
    crate::functions::graphics::parse_manipulate_bindings(bindings_json);
  let code = crate::functions::graphics::manipulate_block_code(body, &bindings);

  let result = match crate::interpret_with_stdout(&code) {
    Ok(r) => r,
    Err(e) => {
      return format!(r#"{{"error":"{}"}}"#, json_escape(&format!("{e}")));
    }
  };

  let mut parts: Vec<String> = Vec::new();
  if let Some(ref svg) = result.graphics {
    parts.push(format!(r#""svg":"{}""#, json_escape(svg)));
  }
  let cleaned_text = residual_text(&result.result);
  if !cleaned_text.is_empty() && result.graphics.is_none() {
    parts.push(format!(r#""text":"{}""#, json_escape(&cleaned_text)));
    if let Some(ref output_svg) = result.output_svg {
      parts.push(format!(r#""textSvg":"{}""#, json_escape(output_svg)));
    }
  }
  format!("{{{}}}", parts.join(","))
}

/// Build a single JSON object string for an output item.
fn json_output_item(kind: &str, content: &str, svg: Option<&str>) -> String {
  let escaped_content = json_escape(content);
  if let Some(svg_str) = svg {
    let escaped_svg = json_escape(svg_str);
    format!(
      r#"{{"type":"{}","text":"{}","svg":"{}"}}"#,
      kind, escaped_content, escaped_svg
    )
  } else if kind == "graphics" {
    format!(r#"{{"type":"graphics","svg":"{}"}}"#, escaped_content)
  } else {
    format!(r#"{{"type":"{}","text":"{}"}}"#, kind, escaped_content)
  }
}

/// Build a "sound" output item carrying base64-encoded audio data. The
/// frontend turns the `audio` + `mime` fields into an `<audio controls>`
/// data URI; `label` (the source file name, when present) is shown next to
/// the player. An empty `audio` field means the data is unavailable (e.g. a
/// local file the browser cannot read) — the player chrome is still shown.
fn json_sound_item(audio: &crate::AudioOutput) -> String {
  let mut fields = vec![
    r#""type":"sound""#.to_string(),
    format!(r#""audio":"{}""#, json_escape(&audio.base64)),
    format!(r#""mime":"{}""#, json_escape(&audio.mime)),
  ];
  if let Some(ref label) = audio.label {
    fields.push(format!(r#""label":"{}""#, json_escape(label)));
  }
  format!("{{{}}}", fields.join(","))
}

/// Escape a string for safe inclusion in JSON.
fn json_escape(s: &str) -> String {
  let mut out = String::with_capacity(s.len() + 16);
  for ch in s.chars() {
    match ch {
      '"' => out.push_str(r#"\""#),
      '\\' => out.push_str(r"\\"),
      '\n' => out.push_str(r"\n"),
      '\r' => out.push_str(r"\r"),
      '\t' => out.push_str(r"\t"),
      c if (c as u32) < 0x20 => {
        out.push_str(&format!(r"\u{:04x}", c as u32));
      }
      c => out.push(c),
    }
  }
  out
}

/// Evaluate, returning only the final expression result (no Print output).
#[wasm_bindgen]
pub fn evaluate_expr(input: &str) -> String {
  match interpret(input) {
    Ok(result) => {
      if result == "\0" {
        String::new()
      } else {
        result
      }
    }
    Err(e) => format!("Error: {e}"),
  }
}
