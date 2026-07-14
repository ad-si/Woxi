use crate::syntax::Expr;

/// Parse CSV input (RFC 4180) into a vector of rows, each row a vector of field strings.
pub fn parse_csv(input: &str) -> Vec<Vec<String>> {
  let mut rows: Vec<Vec<String>> = Vec::new();
  let mut row: Vec<String> = Vec::new();
  let mut field = String::new();
  let mut in_quotes = false;
  let mut chars = input.chars().peekable();

  while let Some(c) = chars.next() {
    if in_quotes {
      if c == '"' {
        if chars.peek() == Some(&'"') {
          // Escaped quote
          chars.next();
          field.push('"');
        } else {
          // End of quoted field
          in_quotes = false;
        }
      } else {
        field.push(c);
      }
    } else {
      match c {
        '"' => {
          in_quotes = true;
        }
        ',' => {
          row.push(std::mem::take(&mut field));
        }
        '\r' => {
          if chars.peek() == Some(&'\n') {
            chars.next();
          }
          row.push(std::mem::take(&mut field));
          rows.push(std::mem::take(&mut row));
        }
        '\n' => {
          row.push(std::mem::take(&mut field));
          rows.push(std::mem::take(&mut row));
        }
        _ => {
          field.push(c);
        }
      }
    }
  }

  // Handle last field/row (if no trailing newline)
  if !field.is_empty() || !row.is_empty() {
    row.push(field);
    rows.push(row);
  }

  rows
}

/// Split TSV content into rows. TSV has no quoting rules in Import — each
/// line is split on tabs verbatim.
pub fn parse_tsv(content: &str) -> Vec<Vec<String>> {
  let trimmed = content.strip_suffix('\n').unwrap_or(content);
  if trimmed.is_empty() {
    Vec::new()
  } else {
    trimmed
      .split('\n')
      .map(|line| line.split('\t').map(|s| s.to_string()).collect())
      .collect()
  }
}

/// Auto-convert a string to Integer, Real, or keep as String.
fn auto_convert(s: &str) -> Expr {
  let trimmed = s.trim();
  if trimmed.is_empty() {
    return Expr::String(s.to_string());
  }

  // Try integer
  if let Ok(n) = trimmed.parse::<i128>() {
    return Expr::Integer(n);
  }

  // Try real
  if let Ok(f) = trimmed.parse::<f64>()
    && (trimmed.contains('.') || trimmed.contains('e') || trimmed.contains('E'))
  {
    return Expr::Real(f);
  }

  Expr::String(s.to_string())
}

/// Available CSV element names.
const ELEMENTS: &[&str] = &[
  "ColumnCount",
  "ColumnLabels",
  "ColumnTypes",
  "Data",
  "Dataset",
  "Dimensions",
  "RawData",
  "RowCount",
  "Schema",
  "Summary",
  "Tabular",
];

/// Extract a specific element from parsed CSV rows.
/// If `element` is None, returns "Data" by default.
pub fn csv_import_element(rows: &[Vec<String>], element: Option<&str>) -> Expr {
  let element = element.unwrap_or("Data");

  if rows.is_empty() {
    return match element {
      "Elements" => Expr::List(
        ELEMENTS
          .iter()
          .map(|s| Expr::String(s.to_string()))
          .collect(),
      ),
      "Data" | "RawData" => Expr::List(vec![].into()),
      "ColumnLabels" => Expr::List(vec![].into()),
      "ColumnCount" => Expr::Integer(0),
      "RowCount" => Expr::Integer(0),
      "Dimensions" => {
        Expr::List(vec![Expr::Integer(0), Expr::Integer(0)].into())
      }
      "ColumnTypes" => Expr::List(vec![].into()),
      "Schema" => super::tabular_ast::build_schema(&[], &[], 0),
      "Summary" => Expr::Association(vec![
        (Expr::String("Columns".to_string()), Expr::Integer(0)),
        (Expr::String("Rows".to_string()), Expr::Integer(0)),
        (
          Expr::String("ColumnLabels".to_string()),
          Expr::List(vec![].into()),
        ),
      ]),
      "Dataset" => {
        super::dataset_ast::dataset_ast(&[Expr::List(vec![].into())])
      }
      "Tabular" => {
        super::tabular_ast::tabular_ast(&[Expr::List(vec![].into())])
      }
      _ => Expr::FunctionCall {
        name: "$Failed".to_string(),
        args: vec![].into(),
      },
    };
  }

  let header = &rows[0];
  let data_rows = if rows.len() > 1 { &rows[1..] } else { &[] };
  let num_cols = header.len();
  let num_data_rows = data_rows.len();

  match element {
    "Elements" => Expr::List(
      ELEMENTS
        .iter()
        .map(|s| Expr::String(s.to_string()))
        .collect(),
    ),

    "Data" => {
      let mut all_rows = Vec::with_capacity(rows.len());
      // Include header row with auto-conversion
      all_rows
        .push(Expr::List(header.iter().map(|s| auto_convert(s)).collect()));
      for row in data_rows {
        all_rows
          .push(Expr::List(row.iter().map(|s| auto_convert(s)).collect()));
      }
      Expr::List(all_rows.into())
    }

    "RawData" => {
      let all_rows: Vec<Expr> = rows
        .iter()
        .map(|row| {
          Expr::List(row.iter().map(|s| Expr::String(s.clone())).collect())
        })
        .collect();
      Expr::List(all_rows.into())
    }

    "ColumnLabels" => {
      Expr::List(header.iter().map(|s| Expr::String(s.clone())).collect())
    }

    "ColumnCount" => Expr::Integer(num_cols as i128),

    "RowCount" => Expr::Integer(num_data_rows as i128),

    "Dimensions" => Expr::List(
      vec![
        Expr::Integer(num_data_rows as i128),
        Expr::Integer(num_cols as i128),
      ]
      .into(),
    ),

    "ColumnTypes" => {
      let types: Vec<Expr> = (0..num_cols)
        .map(|col| {
          let values: Vec<Expr> = data_rows
            .iter()
            .filter_map(|row| row.get(col).map(|s| auto_convert(s)))
            .collect();
          let refs: Vec<&Expr> = values.iter().collect();
          let t = super::tabular_ast::infer_column_type(&refs);
          Expr::Identifier(t)
        })
        .collect();
      Expr::List(types.into())
    }

    "Dataset" => {
      // Build list of associations: each data row becomes <|col1 -> val1, ...|>
      let assocs: Vec<Expr> = data_rows
        .iter()
        .map(|row| {
          let pairs: Vec<(Expr, Expr)> = header
            .iter()
            .zip(row.iter())
            .map(|(h, v)| (Expr::String(h.clone()), auto_convert(v)))
            .collect();
          Expr::Association(pairs)
        })
        .collect();
      super::dataset_ast::dataset_ast(&[Expr::List(assocs.into())])
    }

    "Tabular" => {
      // Build list of lists for data rows (without header)
      let data_lists: Vec<Expr> = data_rows
        .iter()
        .map(|row| Expr::List(row.iter().map(|s| auto_convert(s)).collect()))
        .collect();
      let col_names: Vec<Expr> =
        header.iter().map(|s| Expr::String(s.clone())).collect();
      super::tabular_ast::tabular_ast(&[
        Expr::List(data_lists.into()),
        Expr::List(col_names.into()),
      ])
    }

    "Schema" => {
      let col_keys: Vec<Expr> =
        header.iter().map(|s| Expr::String(s.clone())).collect();
      let col_types: Vec<String> = (0..num_cols)
        .map(|col| {
          let values: Vec<Expr> = data_rows
            .iter()
            .filter_map(|row| row.get(col).map(|s| auto_convert(s)))
            .collect();
          let refs: Vec<&Expr> = values.iter().collect();
          super::tabular_ast::infer_column_type(&refs)
        })
        .collect();
      super::tabular_ast::build_schema(&col_keys, &col_types, num_data_rows)
    }

    "Summary" => Expr::Association(vec![
      (
        Expr::String("Columns".to_string()),
        Expr::Integer(num_cols as i128),
      ),
      (
        Expr::String("Rows".to_string()),
        Expr::Integer(num_data_rows as i128),
      ),
      (
        Expr::String("ColumnLabels".to_string()),
        Expr::List(header.iter().map(|s| Expr::String(s.clone())).collect()),
      ),
    ]),

    _ => Expr::FunctionCall {
      name: "$Failed".to_string(),
      args: vec![].into(),
    },
  }
}

/// Resolve an Import position spec — `Integer` (1-based, negative counts
/// from the end), `Span[a, b]` / `Span[a, b, step]` (`a ;; b ;; step`, with
/// `All` endpoints), or `All` — against `len` items. Returns the selected
/// 0-based indices, or `None` when the spec has another shape or points
/// outside the data.
fn resolve_position_spec(spec: &Expr, len: usize) -> Option<Vec<usize>> {
  let ilen = len as i64;
  let normalize = |pos: i64| -> Option<usize> {
    let p = if pos < 0 { ilen + pos + 1 } else { pos };
    if (1..=ilen).contains(&p) {
      Some((p - 1) as usize)
    } else {
      None
    }
  };
  match spec {
    Expr::Integer(n) => Some(vec![normalize(*n as i64)?]),
    Expr::Identifier(s) if s == "All" => Some((0..len).collect()),
    Expr::FunctionCall { name, args }
      if name == "Span" && (args.len() == 2 || args.len() == 3) =>
    {
      let bound = |e: &Expr, default: i64| -> Option<i64> {
        match e {
          Expr::Identifier(s) if s == "All" => Some(default),
          Expr::Integer(n) => Some(*n as i64),
          _ => None,
        }
      };
      let step = match args.get(2) {
        Some(Expr::Integer(n)) if *n != 0 => *n as i64,
        Some(_) => return None,
        None => 1,
      };
      let start = bound(&args[0], if step < 0 { ilen } else { 1 })?;
      let end = bound(&args[1], if step < 0 { 1 } else { ilen })?;
      let start = normalize(start)? as i64;
      let end = normalize(end)? as i64;
      let mut out = Vec::new();
      let mut i = start;
      while (step > 0 && i <= end) || (step < 0 && i >= end) {
        out.push(i as usize);
        i += step;
      }
      Some(out)
    }
    _ => None,
  }
}

/// Whether `spec` is a shape `resolve_position_spec` understands. Used by
/// the Import dispatcher to decide between handling `{"Data", …}` here and
/// leaving the call unevaluated.
pub fn is_position_spec(spec: &Expr) -> bool {
  matches!(spec, Expr::Integer(_))
    || matches!(spec, Expr::Identifier(s) if s == "All")
    || matches!(spec, Expr::FunctionCall { name, args }
        if name == "Span" && (args.len() == 2 || args.len() == 3))
}

/// `Import[…, {"Data", rowspec}]` / `{"Data", rowspec, colspec}` — extract a
/// subset of rows (and optionally columns) without materializing the whole
/// table. Rows are 1-based over all rows of the file (the header line is row
/// 1, matching wolframscript). A bare integer spec yields the row itself
/// (or the cell when both specs are integers); spans and `All` yield lists.
pub fn csv_import_data_spec(
  rows: &[Vec<String>],
  row_spec: &Expr,
  col_spec: Option<&Expr>,
) -> Result<Expr, crate::InterpreterError> {
  let noelem = || {
    crate::emit_message(
      "Import::noelem: The Import element is not present when importing as CSV.",
    );
    Expr::Identifier("$Failed".to_string())
  };
  let Some(row_idx) = resolve_position_spec(row_spec, rows.len()) else {
    return Ok(noelem());
  };
  let scalar_row = matches!(row_spec, Expr::Integer(_));
  let scalar_col = matches!(col_spec, Some(Expr::Integer(_)));

  let mut out_rows: Vec<Expr> = Vec::with_capacity(row_idx.len());
  for ri in row_idx {
    let row = &rows[ri];
    let converted: Expr = match col_spec {
      None => Expr::List(row.iter().map(|s| auto_convert(s)).collect()),
      Some(cs) => {
        let Some(col_idx) = resolve_position_spec(cs, row.len()) else {
          return Ok(noelem());
        };
        if scalar_col {
          auto_convert(&row[col_idx[0]])
        } else {
          Expr::List(col_idx.iter().map(|&ci| auto_convert(&row[ci])).collect())
        }
      }
    };
    out_rows.push(converted);
  }
  if scalar_row {
    Ok(out_rows.pop().expect("scalar row spec resolves to one row"))
  } else {
    Ok(Expr::List(out_rows.into()))
  }
}

/// Import a CSV file and optionally extract a specific element.
#[cfg(not(target_arch = "wasm32"))]
pub fn csv_import_file(
  path: &str,
  element: Option<&str>,
) -> Result<Expr, crate::InterpreterError> {
  let content = std::fs::read_to_string(path).map_err(|e| {
    crate::InterpreterError::EvaluationError(format!(
      "Import: cannot open \"{}\": {}",
      path, e
    ))
  })?;
  let rows = parse_csv(&content);
  Ok(csv_import_element(&rows, element))
}

/// Import a row/column subset of a CSV file:
/// `Import["f.csv", {"Data", rowspec}]` / `{"Data", rowspec, colspec}`.
#[cfg(not(target_arch = "wasm32"))]
pub fn csv_import_file_data_spec(
  path: &str,
  row_spec: &Expr,
  col_spec: Option<&Expr>,
) -> Result<Expr, crate::InterpreterError> {
  let content = std::fs::read_to_string(path).map_err(|e| {
    crate::InterpreterError::EvaluationError(format!(
      "Import: cannot open \"{}\": {}",
      path, e
    ))
  })?;
  let rows = parse_csv(&content);
  csv_import_data_spec(&rows, row_spec, col_spec)
}

/// Fetch a CSV file from a URL and optionally extract a specific element.
#[cfg(not(target_arch = "wasm32"))]
pub fn csv_import_from_url(
  url: &str,
  element: Option<&str>,
) -> Result<Expr, crate::InterpreterError> {
  let output = std::process::Command::new("curl")
    .args(["-fsSL", "--max-time", "15", url])
    .output()
    .map_err(|e| {
      crate::InterpreterError::EvaluationError(format!(
        "Import: failed to run curl: {}",
        e
      ))
    })?;
  if !output.status.success() {
    let stderr = String::from_utf8_lossy(&output.stderr);
    return Err(crate::InterpreterError::EvaluationError(format!(
      "Import: failed to download \"{}\": {}",
      url,
      stderr.trim()
    )));
  }
  let content = String::from_utf8(output.stdout).map_err(|e| {
    crate::InterpreterError::EvaluationError(format!(
      "Import: downloaded CSV is not valid UTF-8: {}",
      e
    ))
  })?;
  let rows = parse_csv(&content);
  Ok(csv_import_element(&rows, element))
}
