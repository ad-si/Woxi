//! The metadata table behind the generated example data.
//!
//! `resources/example_data_meta.tsv.gz` is produced by
//! `scripts/example_data_meta.wls` from a wolframscript installation. It
//! records the *structure* of every entry of every collection — the
//! property list, exact values of metadata properties (sizes, counts,
//! names, types), a skeleton of every data property, and summary
//! statistics — but never the data itself.
//!
//! Layout: one `@@Collections` line, then per collection a header line
//! `@Collection<TAB>column…` and one line per entry `name<TAB>cell…`. A cell
//! starting with `=` holds the InputForm of an exact value, any other
//! non-empty cell the InputForm of a skeleton (`Sk[…]`, `Arr[…]`, `Rep[…]`,
//! a bare head such as `String`), and `@`-columns hold exact InputForm
//! values without the `=`.

use std::collections::HashMap;
use std::sync::LazyLock;

use crate::syntax::Expr;

/// One collection's table.
pub struct Table {
  columns: Vec<String>,
  column_index: HashMap<String, usize>,
  rows: Vec<Row>,
  row_index: HashMap<String, usize>,
}

/// One entry's row of raw cells, aligned with [`Table::columns`].
pub struct Row {
  pub name: String,
  cells: Vec<String>,
}

pub struct Meta {
  /// `ExampleData[]`, as InputForm.
  pub collections: String,
  tables: HashMap<String, Table>,
}

pub static META: LazyLock<Meta> = LazyLock::new(|| {
  use flate2::read::GzDecoder;
  use std::io::Read;

  let compressed =
    include_bytes!("../../../resources/example_data_meta.tsv.gz");
  let mut text = String::new();
  GzDecoder::new(&compressed[..])
    .read_to_string(&mut text)
    .expect("failed to decompress the example data metadata");
  parse(&text)
});

fn parse(text: &str) -> Meta {
  let mut collections = String::new();
  let mut tables: HashMap<String, Table> = HashMap::new();
  let mut current: Option<String> = None;
  for line in text.lines() {
    if let Some(rest) = line.strip_prefix("@@Collections\t") {
      collections = rest.to_string();
    } else if let Some(header) = line.strip_prefix('@') {
      let mut fields = header.split('\t');
      let name = fields.next().unwrap_or_default().to_string();
      let columns: Vec<String> = fields.map(str::to_string).collect();
      let column_index = columns
        .iter()
        .enumerate()
        .map(|(i, c)| (c.clone(), i))
        .collect();
      tables.insert(
        name.clone(),
        Table {
          columns,
          column_index,
          rows: Vec::new(),
          row_index: HashMap::new(),
        },
      );
      current = Some(name);
    } else if let Some(table) = current
      .as_ref()
      .filter(|_| line.contains('\t'))
      .and_then(|c| tables.get_mut(c))
    {
      let mut fields = line.split('\t');
      let name = fields.next().unwrap_or_default().to_string();
      let cells = fields.map(str::to_string).collect();
      table.row_index.insert(name.clone(), table.rows.len());
      table.rows.push(Row { name, cells });
    }
  }
  Meta {
    collections,
    tables,
  }
}

impl Meta {
  pub fn table(&self, collection: &str) -> Option<&Table> {
    self.tables.get(collection)
  }
}

impl Table {
  pub fn rows(&self) -> &[Row] {
    &self.rows
  }

  pub fn row(&self, name: &str) -> Option<&Row> {
    self.row_index.get(name).map(|&i| &self.rows[i])
  }

  pub fn columns(&self) -> &[String] {
    &self.columns
  }

  /// The raw cell of `row` in `column`, if the table has that column and
  /// the cell is not empty.
  pub fn raw<'a>(&self, row: &'a Row, column: &str) -> Option<&'a str> {
    let &i = self.column_index.get(column)?;
    row
      .cells
      .get(i)
      .map(String::as_str)
      .filter(|c| !c.is_empty())
  }
}

/// A parsed cell.
pub enum Cell {
  /// An exact value, returned as it is.
  Exact(Expr),
  /// A skeleton describing a value Woxi has to generate.
  Skeleton(Expr),
}

/// Parse InputForm text from the table. The table is generated, so a parse
/// failure is a bug in the table, not in the user's input.
pub fn parse_value(text: &str) -> Expr {
  crate::syntax::string_to_expr(text)
    .unwrap_or_else(|_| panic!("unparsable example data metadata: {text}"))
}

impl Cell {
  pub fn from_raw(raw: &str) -> Cell {
    match raw.strip_prefix('=') {
      Some(exact) => Cell::Exact(parse_value(exact)),
      None => Cell::Skeleton(parse_value(raw)),
    }
  }
}
