//! Decode the stored Output cells of a notebook (RasterBox snapshots and
//! CheckboxBox grids) so their display support can be checked headlessly.
//!
//! Usage: cargo run --example dump_stored_outputs -- notebook.nb [out-dir]

use std::path::PathBuf;

use woxi::notebook::{
  CellEntry, CellStyle, parse_notebook, stored_output_checkbox_text,
  stored_output_image_svg,
};

fn main() {
  let path: PathBuf = std::env::args()
    .nth(1)
    .expect("notebook path required")
    .into();
  let out_dir: Option<PathBuf> = std::env::args().nth(2).map(Into::into);
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
      if cell.style != CellStyle::Output {
        continue;
      }
      if let Some(svg) = stored_output_image_svg(&cell.content) {
        let head: String = svg.chars().take(100).collect();
        println!("#{idx}: raster snapshot -> {head}");
        if let Some(dir) = &out_dir {
          std::fs::create_dir_all(dir).unwrap();
          let f = dir.join(format!("snapshot-{idx}.svg"));
          std::fs::write(&f, &svg).unwrap();
          println!("  wrote {}", f.display());
        }
      } else if let Some(text) = stored_output_checkbox_text(&cell.content) {
        println!("#{idx}: checkboxes:\n{text}");
      }
    }
  }
}
