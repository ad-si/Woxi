use wasm_bindgen::prelude::*;

use crate::{clear_state, interpret, interpret_with_stdout};

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

/// Clear all interpreter state (variables and function definitions).
#[wasm_bindgen]
pub fn clear() {
  clear_state();
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
