use crate::InterpreterError;
use crate::syntax::Expr;
use calamine::{Data, Reader, open_workbook_auto, open_workbook_auto_from_rs};
use std::io::Cursor;

/// Convert a single calamine cell value to an `Expr`.
fn cell_to_expr(cell: &Data) -> Expr {
  match cell {
    // Excel stores all numbers as f64. Wolfram's Import consistently returns
    // them as Reals (e.g. `1` → `1.`), so we do the same regardless of
    // whether calamine surfaces the cell as Int or Float.
    Data::Int(n) => Expr::Real(*n as f64),
    Data::Float(f) => Expr::Real(*f),
    Data::String(s) => Expr::String(s.clone()),
    Data::Bool(b) => {
      Expr::Identifier(if *b { "True" } else { "False" }.to_string())
    }
    Data::DateTime(dt) => Expr::Real(dt.as_f64()),
    Data::DateTimeIso(s) => Expr::String(s.clone()),
    Data::DurationIso(s) => Expr::String(s.clone()),
    Data::Error(_) => Expr::Identifier("$Failed".to_string()),
    Data::Empty => Expr::String(String::new()),
  }
}

/// Convert one worksheet (a 2-D range) into a list-of-rows `Expr`.
///
/// Trailing empty rows and trailing empty cells in each row are dropped so
/// the output matches how Wolfram's Import trims sparse sheets.
fn sheet_to_expr(range: &calamine::Range<Data>) -> Expr {
  let mut rows: Vec<Expr> = Vec::new();
  for row in range.rows() {
    let cells: Vec<Expr> = row.iter().map(cell_to_expr).collect();
    rows.push(Expr::List(cells));
  }
  // Drop trailing all-empty rows.
  while let Some(Expr::List(cells)) = rows.last() {
    let all_empty = cells
      .iter()
      .all(|c| matches!(c, Expr::String(s) if s.is_empty()));
    if all_empty {
      rows.pop();
    } else {
      break;
    }
  }
  Expr::List(rows)
}

/// Open an xls / xlsx / xlsb / ods workbook and return every sheet as a list
/// of rows. The outer list is indexed by sheet in workbook order.
fn load_sheets(
  path: &str,
) -> Result<(Vec<String>, Vec<Expr>), InterpreterError> {
  let workbook = open_workbook_auto(path).map_err(|e| {
    InterpreterError::EvaluationError(format!(
      "Import: cannot open \"{}\": {}",
      path, e
    ))
  })?;
  collect_sheets(workbook)
}

fn load_sheets_from_bytes(
  bytes: Vec<u8>,
  source: &str,
) -> Result<(Vec<String>, Vec<Expr>), InterpreterError> {
  let workbook =
    open_workbook_auto_from_rs(Cursor::new(bytes)).map_err(|e| {
      InterpreterError::EvaluationError(format!(
        "Import: cannot open \"{}\": {}",
        source, e
      ))
    })?;
  collect_sheets(workbook)
}

fn collect_sheets<RS: std::io::Read + std::io::Seek>(
  mut workbook: calamine::Sheets<RS>,
) -> Result<(Vec<String>, Vec<Expr>), InterpreterError> {
  let names = workbook.sheet_names();
  let mut sheets: Vec<Expr> = Vec::with_capacity(names.len());
  for name in &names {
    let range = workbook.worksheet_range(name).map_err(|e| {
      InterpreterError::EvaluationError(format!(
        "Import: failed to read sheet \"{}\": {:?}",
        name, e
      ))
    })?;
    sheets.push(sheet_to_expr(&range));
  }
  Ok((names, sheets))
}

/// Fetch the raw bytes at a URL via `curl`. Shared by the txt and xlsx
/// URL-import paths so both failure modes and the command-line flags stay
/// consistent.
pub fn download_url(url: &str) -> Result<Vec<u8>, InterpreterError> {
  let output = std::process::Command::new("curl")
    .args(["-fsSL", "--max-time", "30", url])
    .output()
    .map_err(|e| {
      InterpreterError::EvaluationError(format!(
        "Import: failed to run curl: {}",
        e
      ))
    })?;
  if !output.status.success() {
    let stderr = String::from_utf8_lossy(&output.stderr);
    return Err(InterpreterError::EvaluationError(format!(
      "Import: failed to download \"{}\": {}",
      url,
      stderr.trim()
    )));
  }
  Ok(output.stdout)
}

/// Import an xlsx/xls/xlsb/ods file with an optional element selector.
///
/// Supported element forms:
/// - `None`               → default: list of all sheets
/// - `Some("Data")`       → same as default
/// - `Some("Sheets")`     → list of sheet names
/// - `Some(("Data", n))`  → the n-th sheet (1-based) unwrapped
pub fn xlsx_import_file(
  path: &str,
  element: Option<&Expr>,
) -> Result<Expr, InterpreterError> {
  let (names, sheets) = load_sheets(path)?;
  select_element(names, sheets, element)
}

/// Fetch and import an xlsx/xls/xlsb/ods workbook from a URL.
pub fn xlsx_import_from_url(
  url: &str,
  element: Option<&Expr>,
) -> Result<Expr, InterpreterError> {
  let bytes = download_url(url)?;
  let (names, sheets) = load_sheets_from_bytes(bytes, url)?;
  select_element(names, sheets, element)
}

fn select_element(
  names: Vec<String>,
  sheets: Vec<Expr>,
  element: Option<&Expr>,
) -> Result<Expr, InterpreterError> {
  // Default: entire workbook as list of sheets.
  let Some(elem) = element else {
    return Ok(Expr::List(sheets));
  };

  match elem {
    Expr::String(s) if s == "Data" => Ok(Expr::List(sheets)),
    Expr::String(s) if s == "Sheets" => {
      Ok(Expr::List(names.into_iter().map(Expr::String).collect()))
    }
    Expr::List(parts) => match parts.as_slice() {
      [Expr::String(tag), Expr::Integer(n)] if tag == "Data" => {
        let idx = *n;
        if idx < 1 || (idx as usize) > sheets.len() {
          return Err(InterpreterError::EvaluationError(format!(
            "Import: sheet index {} out of range (1..{})",
            idx,
            sheets.len()
          )));
        }
        Ok(sheets.into_iter().nth((idx - 1) as usize).unwrap())
      }
      _ => Err(InterpreterError::EvaluationError(format!(
        "Import: unsupported element {:?} for xlsx file",
        elem
      ))),
    },
    _ => Err(InterpreterError::EvaluationError(format!(
      "Import: unsupported element {:?} for xlsx file",
      elem
    ))),
  }
}
