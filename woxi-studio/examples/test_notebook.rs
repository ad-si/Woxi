//! Test all input cells in a notebook file by evaluating them through
//! the woxi interpreter and reporting which ones succeed or fail.
//!
//! Usage: cargo run --example test_notebook -- path/to/notebook.nb

use std::path::PathBuf;

// Pull in the studio's notebook parser.
#[path = "../src/notebook.rs"]
mod notebook;

use notebook::{CellEntry, CellStyle, parse_notebook};

fn main() {
  let path: PathBuf = std::env::args()
    .nth(1)
    .expect("notebook path required")
    .into();
  let src = std::fs::read_to_string(&path).expect("read file");
  let nb = parse_notebook(&src).expect("parse notebook");

  // Flatten all cells in order.
  let mut all_cells = Vec::new();
  for entry in &nb.cells {
    match entry {
      CellEntry::Single(c) => all_cells.push(c.clone()),
      CellEntry::Group(g) => {
        for c in &g.cells {
          all_cells.push(c.clone());
        }
      }
    }
  }

  let mut total_input = 0;
  let mut ok_count = 0;
  let mut err_count = 0;
  let mut empty_count = 0;

  // Walk cells: when we see an Input, evaluate it; if the *next* cell is
  // an Output, compare.
  let mut i = 0;
  while i < all_cells.len() {
    let cell = &all_cells[i];
    if matches!(cell.style, CellStyle::Input | CellStyle::Code) {
      total_input += 1;
      let code = cell.content.trim();
      let expected = if i + 1 < all_cells.len()
        && all_cells[i + 1].style == CellStyle::Output
      {
        Some(all_cells[i + 1].content.trim().to_string())
      } else {
        None
      };

      println!("--- Input #{total_input} ---");
      println!("CODE:");
      for line in code.lines() {
        println!("  {line}");
      }
      // Detect expected output that is a graphics expression (huge box dumps).
      let expected_is_graphics = expected
        .as_ref()
        .map(|e| {
          e.starts_with("GraphicsBox")
            || e.starts_with("Graphics3DBox")
            || e.contains("CompressedData[")
        })
        .unwrap_or(false);

      // Split into statements and evaluate.
      let statements = woxi::split_into_statements(code);
      let mut all_outputs = Vec::new();
      let mut had_err = false;
      let mut last_warnings: Vec<String> = Vec::new();
      for stmt in &statements {
        match woxi::interpret_with_stdout(stmt) {
          Ok(res) => {
            last_warnings.extend(res.warnings);
            if res.result != "\0" {
              all_outputs.push(res.result);
            }
          }
          Err(woxi::InterpreterError::EmptyInput) => {}
          Err(e) => {
            all_outputs.push(format!("ERROR: {e}"));
            had_err = true;
          }
        }
      }
      let actual = all_outputs.join("\n");

      if !last_warnings.is_empty() {
        println!("WARN: {last_warnings:?}");
      }

      if had_err {
        err_count += 1;
        println!("STATUS: ERROR");
      } else if actual.is_empty() {
        empty_count += 1;
        println!("STATUS: OK (no output)");
      } else {
        ok_count += 1;
        println!("STATUS: OK");
      }
      println!("ACTUAL: {actual}");
      if let Some(exp) = expected {
        if expected_is_graphics {
          // Just show a snippet; full graphics box output is huge.
          let snippet: String = exp.chars().take(80).collect();
          println!("EXPECTED: {snippet}... (graphics output)");
        } else {
          println!("EXPECTED: {exp}");
        }
      }
      println!();
    }
    i += 1;
  }

  println!("===========================================");
  println!("Total Input cells: {total_input}");
  println!("Successful evaluations: {ok_count}");
  println!("Successful (no output): {empty_count}");
  println!("Errors: {err_count}");
}
