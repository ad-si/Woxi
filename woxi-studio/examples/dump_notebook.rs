//! Dump every parsed cell of a notebook file (style + content snippet) so
//! rendering problems can be spotted without launching the GUI.
//!
//! Usage: cargo run --example dump_notebook -- path/to/notebook.nb [max-chars]

use std::path::PathBuf;

use woxi::notebook::{CellEntry, parse_notebook};

fn main() {
  let path: PathBuf = std::env::args()
    .nth(1)
    .expect("notebook path required")
    .into();
  let max_chars: usize = std::env::args()
    .nth(2)
    .map(|s| s.parse().expect("max-chars must be a number"))
    .unwrap_or(200);
  let src = std::fs::read_to_string(&path).expect("read file");
  let nb = parse_notebook(&src).expect("parse notebook");

  let mut idx = 0;
  for entry in &nb.cells {
    let cells: Vec<_> = match entry {
      CellEntry::Single(c) => vec![c],
      CellEntry::Group(g) => g.cells.iter().collect(),
    };
    for cell in cells {
      idx += 1;
      let content = cell.content.replace('\n', "\\n");
      let snippet: String = content.chars().take(max_chars).collect();
      let len = cell.content.chars().count();
      println!("#{idx:3} [{:12}] ({len:6} chars) {snippet}", cell.style);
    }
  }
}
