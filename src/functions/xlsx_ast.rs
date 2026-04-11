use crate::InterpreterError;
use crate::functions::math_ast::try_eval_to_f64;
use crate::syntax::Expr;
use calamine::{Data, Reader, open_workbook_auto, open_workbook_auto_from_rs};
use rust_xlsxwriter::Workbook;
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

/// Write a single cell value. We mirror how Wolfram's `Export[_, _, "XLSX"]`
/// behaves: numeric expressions (including symbolic constants like `Pi` and
/// rationals like `1/7`) are written as IEEE-754 doubles, booleans stay
/// boolean, strings stay string, and anything else is rendered to its
/// canonical textual form so no data silently disappears.
fn write_cell(
  worksheet: &mut rust_xlsxwriter::Worksheet,
  row: u32,
  col: u16,
  value: &Expr,
) -> Result<(), InterpreterError> {
  let to_err = |e: rust_xlsxwriter::XlsxError| {
    InterpreterError::EvaluationError(format!("Export: xlsx write error: {e}"))
  };
  match value {
    Expr::String(s) => {
      worksheet.write_string(row, col, s).map_err(to_err)?;
    }
    Expr::Identifier(name) if name == "True" => {
      worksheet.write_boolean(row, col, true).map_err(to_err)?;
    }
    Expr::Identifier(name) if name == "False" => {
      worksheet.write_boolean(row, col, false).map_err(to_err)?;
    }
    Expr::Identifier(name) if name == "Null" => {
      // Write nothing – leave the cell empty.
    }
    _ => {
      if let Some(f) = try_eval_to_f64(value) {
        worksheet.write_number(row, col, f).map_err(to_err)?;
      } else {
        // Fall back to the canonical textual form (e.g. unbound symbols,
        // complex numbers, nested expressions).
        let text = crate::syntax::expr_to_string(value);
        worksheet.write_string(row, col, &text).map_err(to_err)?;
      }
    }
  }
  Ok(())
}

/// Return `Some(rows)` if `expr` is a 2-D list (list of lists of atoms),
/// `None` otherwise.
fn as_sheet_rows(expr: &Expr) -> Option<Vec<&[Expr]>> {
  let Expr::List(rows) = expr else {
    return None;
  };
  let mut out: Vec<&[Expr]> = Vec::with_capacity(rows.len());
  for row in rows {
    let Expr::List(cells) = row else {
      return None;
    };
    out.push(cells.as_slice());
  }
  Some(out)
}

/// Export `data` to an xlsx file at `path`.
///
/// Supported shapes for `data`:
/// - A 2-D list `{{...}, {...}}` — written as a single worksheet `Sheet1`.
/// - A 3-D list `{{{...}}, {{...}}}` — each outer element becomes one
///   worksheet in workbook order (`Sheet1`, `Sheet2`, ...).
/// - A flat 1-D list `{a, b, c}` — written as a single-row worksheet.
pub fn xlsx_export_file(
  path: &str,
  data: &Expr,
) -> Result<(), InterpreterError> {
  let mut workbook = Workbook::new();

  // Decide whether `data` is a list of sheets or a single sheet.
  let sheets: Vec<Vec<&[Expr]>> = if let Expr::List(outer) = data {
    // A 3-D list: every element itself decomposes into rows.
    let three_d: Option<Vec<Vec<&[Expr]>>> =
      outer.iter().map(as_sheet_rows).collect();
    if let Some(all_sheets) = three_d.filter(|v| !v.is_empty()) {
      all_sheets
    } else if let Some(rows) = as_sheet_rows(data) {
      // A 2-D list: one sheet.
      vec![rows]
    } else {
      // A 1-D list of atoms: treat as a single row in a single sheet.
      vec![vec![outer.as_slice()]]
    }
  } else {
    return Err(InterpreterError::EvaluationError(format!(
      "Export: xlsx data must be a list, got {}",
      crate::syntax::expr_to_string(data)
    )));
  };

  for sheet_rows in &sheets {
    let worksheet = workbook.add_worksheet();
    for (r, row_cells) in sheet_rows.iter().enumerate() {
      for (c, cell) in row_cells.iter().enumerate() {
        write_cell(worksheet, r as u32, c as u16, cell)?;
      }
    }
  }

  workbook.save(path).map_err(|e| {
    InterpreterError::EvaluationError(format!(
      "Export: cannot write \"{}\": {}",
      path, e
    ))
  })?;
  Ok(())
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
