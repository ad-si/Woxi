use wasm_bindgen::prelude::*;

use crate::{clear_state, interpret, interpret_with_stdout};

// Import a JS-provided function that fetches a URL and returns its content
// as a base64-encoded string.  The host (worker.js / kernel) must supply this.
#[wasm_bindgen]
extern "C" {
  #[wasm_bindgen(js_name = "__woxi_fetch_url", catch)]
  fn woxi_fetch_url(url: &str) -> Result<String, JsValue>;
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
  console_error_panic_hook::set_once();
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

        // If the statement is a top-level Manipulate[…] call, emit a
        // dedicated "manipulate" item so the frontend can render
        // interactive controls instead of the plain text echo. We re-parse
        // the source so we can inspect the held expression shape.
        if result.result != "\0"
          && let Some(item) = try_build_manipulate_item(stmt)
        {
          items.push(item);
          continue;
        }

        // Main result
        if let Some(ref svg) = result.graphics {
          // Only display graphics if output wasn't suppressed by trailing semicolon
          if result.result != "\0" {
            items.push(json_output_item("graphics", svg, None));
            // Check for non-graphics text mixed in
            let cleaned = result
              .result
              .replace("-Graphics-", "")
              .replace("-Graphics3D-", "")
              .replace("-Image-", "");
            let cleaned = cleaned.trim();
            if !cleaned.is_empty() && cleaned != "\0" {
              items.push(json_output_item("text", cleaned, None));
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
  }

  format!("[{}]", items.join(","))
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
  let spec = crate::functions::graphics::extract_manipulate_spec(&expr)?;

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
      warnings: vec![],
    },
  };

  let spec_json = crate::functions::graphics::manipulate_spec_to_json(&spec);

  // Build initial-rendering JSON.
  let mut initial_parts: Vec<String> = Vec::new();
  if let Some(ref svg) = initial.graphics {
    initial_parts.push(format!(r#""svg":"{}""#, json_escape(svg)));
  }
  let cleaned_text = initial
    .result
    .replace("-Graphics-", "")
    .replace("-Graphics3D-", "")
    .replace("-Image-", "");
  let cleaned_text = cleaned_text.trim();
  if !cleaned_text.is_empty() && initial.graphics.is_none() {
    initial_parts.push(format!(r#""text":"{}""#, json_escape(cleaned_text)));
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
  let cleaned_text = result
    .result
    .replace("-Graphics-", "")
    .replace("-Graphics3D-", "")
    .replace("-Image-", "");
  let cleaned_text = cleaned_text.trim();
  if !cleaned_text.is_empty() && result.graphics.is_none() {
    parts.push(format!(r#""text":"{}""#, json_escape(cleaned_text)));
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
