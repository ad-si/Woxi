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

/// Auto-convert a string to Integer, Real, or keep as String.
pub fn auto_convert(s: &str) -> Expr {
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
      "Data" | "RawData" => Expr::List(vec![]),
      "ColumnLabels" => Expr::List(vec![]),
      "ColumnCount" => Expr::Integer(0),
      "RowCount" => Expr::Integer(0),
      "Dimensions" => Expr::List(vec![Expr::Integer(0), Expr::Integer(0)]),
      "ColumnTypes" => Expr::List(vec![]),
      "Schema" => super::tabular_ast::build_schema(&[], &[], 0),
      "Summary" => Expr::Association(vec![
        (Expr::String("Columns".to_string()), Expr::Integer(0)),
        (Expr::String("Rows".to_string()), Expr::Integer(0)),
        (Expr::String("ColumnLabels".to_string()), Expr::List(vec![])),
      ]),
      "Dataset" => super::dataset_ast::dataset_ast(&[Expr::List(vec![])]),
      "Tabular" => super::tabular_ast::tabular_ast(&[Expr::List(vec![])]),
      _ => Expr::FunctionCall {
        name: "$Failed".to_string(),
        args: vec![],
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
      Expr::List(all_rows)
    }

    "RawData" => {
      let all_rows: Vec<Expr> = rows
        .iter()
        .map(|row| {
          Expr::List(row.iter().map(|s| Expr::String(s.clone())).collect())
        })
        .collect();
      Expr::List(all_rows)
    }

    "ColumnLabels" => {
      Expr::List(header.iter().map(|s| Expr::String(s.clone())).collect())
    }

    "ColumnCount" => Expr::Integer(num_cols as i128),

    "RowCount" => Expr::Integer(num_data_rows as i128),

    "Dimensions" => Expr::List(vec![
      Expr::Integer(num_data_rows as i128),
      Expr::Integer(num_cols as i128),
    ]),

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
      Expr::List(types)
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
      super::dataset_ast::dataset_ast(&[Expr::List(assocs)])
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
        Expr::List(data_lists),
        Expr::List(col_names),
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
      args: vec![],
    },
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
