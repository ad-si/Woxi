use crate::syntax::Expr;

/// Infer the element type for a column of values.
fn infer_column_type(values: &[&Expr]) -> String {
  if values.is_empty() {
    return "Expression".to_string();
  }
  let mut seen_int = false;
  let mut seen_real = false;
  let mut seen_string = false;
  let mut seen_other = false;
  for v in values {
    match v {
      Expr::Integer(_) | Expr::BigInteger(_) => seen_int = true,
      Expr::Real(_) | Expr::BigFloat(_, _) => seen_real = true,
      Expr::String(_) => seen_string = true,
      Expr::Identifier(s) if s == "Missing" || s == "Null" => {
        // Missing/Null values don't affect column type
      }
      Expr::FunctionCall { name, .. } if name == "Missing" => {
        // Missing[] values don't affect column type
      }
      _ => seen_other = true,
    }
  }
  if seen_other || (seen_string && (seen_int || seen_real)) {
    "Expression".to_string()
  } else if seen_string {
    "String".to_string()
  } else if seen_real {
    "Real".to_string()
  } else if seen_int {
    "Integer".to_string()
  } else {
    "Expression".to_string()
  }
}

/// Build a TabularSchema expression from column keys and types.
fn build_schema(
  col_keys: &[Expr],
  col_types: &[String],
  num_rows: usize,
) -> Expr {
  // Build ColumnKeys -> {key1, key2, ...}
  let column_keys_rule = (
    Expr::String("ColumnKeys".to_string()),
    Expr::List(col_keys.to_vec()),
  );

  // Build ColumnTypes -> <|key1 -> type1, key2 -> type2, ...|>
  let type_pairs: Vec<(Expr, Expr)> = col_keys
    .iter()
    .zip(col_types.iter())
    .map(|(k, t)| (k.clone(), Expr::Identifier(t.clone())))
    .collect();
  let column_types_rule = (
    Expr::String("ColumnTypes".to_string()),
    Expr::Association(type_pairs),
  );

  // Build RowCount -> n
  let row_count_rule = (
    Expr::String("RowCount".to_string()),
    Expr::Integer(num_rows as i128),
  );

  Expr::FunctionCall {
    name: "TabularSchema".to_string(),
    args: vec![Expr::Association(vec![
      column_keys_rule,
      column_types_rule,
      row_count_rule,
    ])],
  }
}

/// Tabular[data] — constructs a Tabular object from rectangular data.
/// Tabular[data, {col1, col2, ...}] — constructs with column names.
/// Tabular[data, schema] — constructs with explicit schema.
pub fn tabular_ast(args: &[Expr]) -> Expr {
  if args.is_empty() {
    return Expr::FunctionCall {
      name: "Tabular".to_string(),
      args: vec![],
    };
  }

  // If already has schema (2+ args where the second is a TabularSchema), return as-is
  if args.len() >= 2
    && let Expr::FunctionCall { name, .. } = &args[1]
    && name == "TabularSchema"
  {
    return Expr::FunctionCall {
      name: "Tabular".to_string(),
      args: args.to_vec(),
    };
  }

  let data = &args[0];

  match data {
    // Tabular[{{row1...}, {row2...}, ...}] - list of lists (rows)
    Expr::List(rows) if !rows.is_empty() => {
      // Check if it's a list of associations -> column-keyed data
      if rows.iter().all(|r| matches!(r, Expr::Association(_))) {
        return tabular_from_list_of_associations(rows, args);
      }

      // Check if it's a list of lists -> matrix-style data
      if rows.iter().all(|r| matches!(r, Expr::List(_))) {
        return tabular_from_list_of_lists(rows, args);
      }

      // Single flat list — treat as single-column data
      tabular_from_flat_list(rows, args)
    }

    // Tabular[<|key1 -> val, ...|>] — column-oriented association
    Expr::Association(pairs) if !pairs.is_empty() => {
      tabular_from_column_association(pairs, args)
    }

    _ => {
      // Return unevaluated for unsupported forms
      Expr::FunctionCall {
        name: "Tabular".to_string(),
        args: args.to_vec(),
      }
    }
  }
}

/// Construct Tabular from a list of associations (row-oriented data).
/// Each association is a row: {<|"a"->1,"b"->2|>, <|"a"->3,"b"->4|>}
fn tabular_from_list_of_associations(rows: &[Expr], args: &[Expr]) -> Expr {
  // Collect column keys in order of first appearance
  let mut col_keys: Vec<Expr> = Vec::new();
  let mut col_key_strs: Vec<String> = Vec::new();
  let mut key_set = std::collections::HashSet::new();

  for row in rows {
    if let Expr::Association(pairs) = row {
      for (k, _) in pairs {
        let key_str = crate::syntax::expr_to_string(k);
        if key_set.insert(key_str.clone()) {
          col_keys.push(k.clone());
          col_key_strs.push(key_str);
        }
      }
    }
  }

  // Override with explicit column names if provided
  if args.len() >= 2
    && let Expr::List(names) = &args[1]
  {
    col_keys = names.clone();
    col_key_strs = names.iter().map(crate::syntax::expr_to_string).collect();
  }

  // Infer column types
  let num_cols = col_keys.len();
  let mut col_types = Vec::with_capacity(num_cols);
  for (j, key_str) in col_key_strs.iter().enumerate() {
    let values: Vec<&Expr> = rows
      .iter()
      .filter_map(|r| {
        if let Expr::Association(pairs) = r {
          pairs
            .iter()
            .find(|(k, _)| crate::syntax::expr_to_string(k) == *key_str)
            .map(|(_, v)| v)
        } else {
          None
        }
      })
      .collect();
    let _ = j;
    col_types.push(infer_column_type(&values));
  }

  let schema = build_schema(&col_keys, &col_types, rows.len());

  Expr::FunctionCall {
    name: "Tabular".to_string(),
    args: vec![args[0].clone(), schema],
  }
}

/// Construct Tabular from a list of lists (matrix-style data).
/// {{1,2,3}, {4,5,6}} with optional column names.
fn tabular_from_list_of_lists(rows: &[Expr], args: &[Expr]) -> Expr {
  // Determine number of columns from the widest row
  let num_cols = rows
    .iter()
    .map(|r| {
      if let Expr::List(items) = r {
        items.len()
      } else {
        0
      }
    })
    .max()
    .unwrap_or(0);

  // Column keys: from second argument or generate as integers 1..n
  let col_keys: Vec<Expr> = if args.len() >= 2 {
    if let Expr::List(names) = &args[1] {
      names.clone()
    } else {
      (1..=num_cols).map(|i| Expr::Integer(i as i128)).collect()
    }
  } else {
    (1..=num_cols).map(|i| Expr::Integer(i as i128)).collect()
  };

  // Infer column types
  let mut col_types = Vec::with_capacity(num_cols);
  for j in 0..num_cols {
    let values: Vec<&Expr> = rows
      .iter()
      .filter_map(|r| {
        if let Expr::List(items) = r {
          items.get(j)
        } else {
          None
        }
      })
      .collect();
    col_types.push(infer_column_type(&values));
  }

  let schema = build_schema(&col_keys, &col_types, rows.len());

  Expr::FunctionCall {
    name: "Tabular".to_string(),
    args: vec![args[0].clone(), schema],
  }
}

/// Construct Tabular from a flat list (single column).
fn tabular_from_flat_list(items: &[Expr], args: &[Expr]) -> Expr {
  let col_keys: Vec<Expr> = if args.len() >= 2 {
    if let Expr::List(names) = &args[1] {
      names.clone()
    } else {
      vec![Expr::Integer(1)]
    }
  } else {
    vec![Expr::Integer(1)]
  };

  let values: Vec<&Expr> = items.iter().collect();
  let col_types = vec![infer_column_type(&values)];

  let schema = build_schema(&col_keys, &col_types, items.len());

  Expr::FunctionCall {
    name: "Tabular".to_string(),
    args: vec![args[0].clone(), schema],
  }
}

/// ToTabular[data, "Columns"] — converts a list of rules to a column-oriented Tabular.
/// {"a" -> {1,2}, "b" -> {3,4}} => Tabular[<|"a" -> {1,2}, "b" -> {3,4}|>]
pub fn to_tabular_ast(args: &[Expr]) -> Expr {
  if args.len() < 2 {
    return Expr::FunctionCall {
      name: "ToTabular".to_string(),
      args: args.to_vec(),
    };
  }

  let orientation = &args[1];

  // Only "Columns" orientation is currently supported
  let is_columns = matches!(orientation, Expr::String(s) if s == "Columns");

  if !is_columns {
    return Expr::FunctionCall {
      name: "ToTabular".to_string(),
      args: args.to_vec(),
    };
  }

  match &args[0] {
    // List of rules: {"a" -> {1, 4}, "b" -> {2, 5}}
    Expr::List(items)
      if !items.is_empty()
        && items.iter().all(|item| matches!(item, Expr::Rule { .. })) =>
    {
      let pairs: Vec<(Expr, Expr)> = items
        .iter()
        .map(|item| {
          if let Expr::Rule {
            pattern,
            replacement,
          } = item
          {
            (*pattern.clone(), *replacement.clone())
          } else {
            unreachable!()
          }
        })
        .collect();
      let assoc_data = Expr::Association(pairs.clone());
      tabular_from_column_association(&pairs, &[assoc_data])
    }
    // Already an association: <|"a" -> {1, 4}, "b" -> {2, 5}|>
    Expr::Association(pairs) if !pairs.is_empty() => {
      tabular_from_column_association(pairs, &[args[0].clone()])
    }
    _ => Expr::FunctionCall {
      name: "ToTabular".to_string(),
      args: args.to_vec(),
    },
  }
}

/// Construct Tabular from a column-oriented association.
/// <|"a" -> {1,2,3}, "b" -> {4,5,6}|>
fn tabular_from_column_association(
  pairs: &[(Expr, Expr)],
  args: &[Expr],
) -> Expr {
  let col_keys: Vec<Expr> = pairs.iter().map(|(k, _)| k.clone()).collect();

  // Determine number of rows from the longest column
  let num_rows = pairs
    .iter()
    .map(|(_, v)| {
      if let Expr::List(items) = v {
        items.len()
      } else {
        1
      }
    })
    .max()
    .unwrap_or(0);

  // Infer column types
  let mut col_types = Vec::with_capacity(pairs.len());
  for (_, v) in pairs {
    let values: Vec<&Expr> = if let Expr::List(items) = v {
      items.iter().collect()
    } else {
      vec![v]
    };
    col_types.push(infer_column_type(&values));
  }

  let schema = build_schema(&col_keys, &col_types, num_rows);

  Expr::FunctionCall {
    name: "Tabular".to_string(),
    args: vec![args[0].clone(), schema],
  }
}
