//! Extract full content of each cell to numbered files.
use std::path::PathBuf;
use woxi::notebook::{CellEntry, parse_notebook};

fn main() {
  let path: PathBuf = std::env::args().nth(1).expect("notebook path").into();
  let outdir: PathBuf = std::env::args().nth(2).expect("out dir").into();
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
      let f = outdir.join(format!("{:03}_{}.txt", idx, cell.style));
      std::fs::write(f, &cell.content).unwrap();
    }
  }
}
