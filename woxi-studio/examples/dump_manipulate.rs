//! Run every Input cell of a notebook through the exact pipeline Woxi
//! Studio uses for interactive cells (evaluate, detect a Manipulate,
//! build the widget state, re-evaluate the body once) and dump the
//! resulting widget structure. This makes "does this notebook's
//! Manipulate work in the Studio?" checkable without launching the GUI.
//!
//! Usage: cargo run --example dump_manipulate -- path/to/notebook.nb [svg-out-dir]
//!
//! When `svg-out-dir` is given, each widget's rendered SVG is written to
//! `<svg-out-dir>/widget-<n>.svg` so the visual output can be inspected.

use std::path::PathBuf;

// The full ControlState API is only partially exercised here; the unused
// remainder is fine for a diagnostic example.
#[allow(dead_code)]
#[path = "../src/manipulate.rs"]
mod manipulate;

use woxi::notebook::{CellEntry, CellStyle, parse_notebook};

fn main() {
  let path: PathBuf = std::env::args()
    .nth(1)
    .expect("notebook path required")
    .into();
  let svg_out_dir: Option<PathBuf> = std::env::args().nth(2).map(Into::into);
  let src = std::fs::read_to_string(&path).expect("read file");
  let nb = parse_notebook(&src).expect("parse notebook");

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

  let mut widget_count = 0;
  for cell in &all_cells {
    if !matches!(cell.style, CellStyle::Input | CellStyle::Code) {
      continue;
    }
    let code = cell.content.trim();
    for stmt in woxi::split_into_statements(code) {
      // Evaluate for side effects (definitions) exactly like the studio.
      let eval = woxi::interpret_with_stdout(&stmt);
      let Ok(expr) = woxi::interpret_to_expr(&stmt) else {
        continue;
      };
      let Some(state) = manipulate::ManipulateState::from_expr(&expr) else {
        if let Ok(res) = &eval
          && res.result.starts_with("Manipulate[")
        {
          println!("!! Manipulate did NOT build a widget:");
          let snippet: String = stmt.chars().take(160).collect();
          println!("   {snippet}");
        }
        continue;
      };
      widget_count += 1;
      println!("=== Interactive widget #{widget_count} ===");
      println!("animated: {}", state.animated);
      println!("controls:");
      for c in &state.controls {
        println!("  {c:?}");
      }
      if !state.state.is_empty() {
        println!("state vars:");
        for (n, v) in &state.state {
          let snippet: String = v.chars().take(100).collect();
          println!("  {n} = {snippet}");
        }
      }
      println!("body:");
      for line in state.body.lines() {
        println!("  {line}");
      }
      println!(
        "render: graphics={} text={:?} error={:?}",
        state.graphics_handle.is_some(),
        state.text_output,
        state.error
      );
      if let Some(dir) = &svg_out_dir
        && state.graphics_handle.is_some()
      {
        // Re-render through the same bindings the widget uses so the SVG
        // bytes can be captured (the iced handle doesn't expose them).
        let mut bindings: Vec<(String, String)> = state
          .controls
          .iter()
          .filter(|c| c.binds_variable())
          .map(|c| (c.name().to_string(), c.current_code()))
          .collect();
        bindings.extend(state.state.iter().cloned());
        let code = match state.initialization.as_deref() {
          Some(init) => format!("{init}; {}", state.body),
          None => state.body.clone(),
        };
        let render = woxi::with_scoped_globals(&bindings, || {
          woxi::interpret_with_stdout(&code)
        });
        if let Ok(res) = render
          && let Some(svg) = res.graphics
        {
          std::fs::create_dir_all(dir).expect("create svg out dir");
          let out = dir.join(format!("widget-{widget_count}.svg"));
          std::fs::write(&out, svg).expect("write svg");
          println!("wrote {}", out.display());
        }
      }
      println!();
    }
  }
  println!("Interactive widgets built: {widget_count}");
}
