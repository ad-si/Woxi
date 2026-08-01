#[allow(unused_imports)]
use super::*;

/// Infer the TypeSystem type for an expression.
fn infer_type(expr: &Expr) -> Expr {
  match expr {
    Expr::Integer(_) | Expr::BigInteger(_) => ts_atom("Integer"),
    Expr::Real(_) | Expr::BigFloat(_, _) => ts_atom("Real"),
    Expr::String(_) => ts_atom("String"),
    Expr::Identifier(name) if name == "True" || name == "False" => {
      // TypeSystem`Atom[TypeSystem`Boolean]
      Expr::FunctionCall {
        name: "TypeSystem`Atom".to_string(),
        args: vec![Expr::Identifier("TypeSystem`Boolean".to_string())].into(),
      }
    }
    Expr::Identifier(name) if name == "Null" => ts_atom("Null"),
    Expr::Identifier(_) => ts_atom("String"),
    Expr::List(items) => infer_list_type(items),
    Expr::Association(pairs) => infer_assoc_type(pairs, true),
    // For any other expression (FunctionCall, etc.), treat as opaque
    _ => ts_atom("Expression"),
  }
}

/// Create TypeSystem`Atom[<name>]
fn ts_atom(name: &str) -> Expr {
  Expr::FunctionCall {
    name: "TypeSystem`Atom".to_string(),
    args: vec![Expr::Identifier(name.to_string())].into(),
  }
}

/// Infer the type for a list of items.
fn infer_list_type(items: &[Expr]) -> Expr {
  if items.is_empty() {
    // Empty list: Vector[Atom[Expression], 0]
    return Expr::FunctionCall {
      name: "TypeSystem`Vector".to_string(),
      args: vec![ts_atom("Expression"), Expr::Integer(0)].into(),
    };
  }

  let types: Vec<Expr> = items.iter().map(infer_type).collect();

  // Check if all types are the same
  let first_str = expr_to_string(&types[0]);
  let all_same = types.iter().all(|t| expr_to_string(t) == first_str);

  if all_same {
    // Vector[type, count]
    Expr::FunctionCall {
      name: "TypeSystem`Vector".to_string(),
      args: vec![types[0].clone(), Expr::Integer(items.len() as i128)].into(),
    }
  } else {
    // Tuple[{type1, type2, ...}]
    Expr::FunctionCall {
      name: "TypeSystem`Tuple".to_string(),
      args: vec![Expr::List(types.into())].into(),
    }
  }
}

/// Infer the type for an association.
/// `top_level` indicates whether this association is the top-level data
/// (affects whether we use Assoc vs Struct for homogeneous values).
fn infer_assoc_type(pairs: &[(Expr, Expr)], top_level: bool) -> Expr {
  if pairs.is_empty() {
    return Expr::FunctionCall {
      name: "TypeSystem`Struct".to_string(),
      args: vec![Expr::List(vec![].into()), Expr::List(vec![].into())].into(),
    };
  }

  let keys: Vec<Expr> = pairs.iter().map(|(k, _)| k.clone()).collect();
  let value_types: Vec<Expr> = pairs
    .iter()
    .map(|(_, v)| {
      // When inferring types for values that are associations nested inside
      // another association, always produce Struct (not Assoc)
      if !top_level {
        infer_type(v)
      } else {
        match v {
          Expr::Association(inner_pairs) => {
            infer_assoc_type(inner_pairs, false)
          }
          _ => infer_type(v),
        }
      }
    })
    .collect();

  // Check if all value types are the same
  let first_str = expr_to_string(&value_types[0]);
  let all_same = value_types.iter().all(|t| expr_to_string(t) == first_str);

  if top_level && all_same {
    // Assoc[keyType, valueType, count]
    let values_are_assoc = matches!(&pairs[0].1, Expr::Association(_));
    let key_type = if values_are_assoc {
      // Keys mapping to associations use Atom[String]
      ts_atom("String")
    } else {
      // Check if keys are integers
      let keys_are_int = keys.iter().all(|k| matches!(k, Expr::Integer(_)));
      if keys_are_int {
        ts_atom("Integer")
      } else {
        // Use Enumeration for string keys
        let key_names: Vec<Expr> = keys
          .iter()
          .map(|k| match k {
            Expr::String(s) => Expr::Identifier(s.clone()),
            Expr::Identifier(s) => Expr::Identifier(s.clone()),
            other => other.clone(),
          })
          .collect();
        Expr::FunctionCall {
          name: "TypeSystem`Atom".to_string(),
          args: vec![Expr::FunctionCall {
            name: "TypeSystem`Enumeration".to_string(),
            args: key_names.into(),
          }]
          .into(),
        }
      }
    };

    Expr::FunctionCall {
      name: "TypeSystem`Assoc".to_string(),
      args: vec![
        key_type,
        value_types[0].clone(),
        Expr::Integer(pairs.len() as i128),
      ]
      .into(),
    }
  } else {
    // Struct[{key1, key2, ...}, {type1, type2, ...}]
    let key_names: Vec<Expr> = keys
      .iter()
      .map(|k| match k {
        Expr::String(s) => Expr::Identifier(s.clone()),
        Expr::Identifier(s) => Expr::Identifier(s.clone()),
        other => other.clone(),
      })
      .collect();

    Expr::FunctionCall {
      name: "TypeSystem`Struct".to_string(),
      args: vec![Expr::List(key_names.into()), Expr::List(value_types.into())]
        .into(),
    }
  }
}

/// Dataset[data, type, meta][op1, op2, …] — query a dataset.
///
/// A dataset query is the same successive-level operator spec `Query` uses,
/// applied to the wrapped data; a collection result is re-wrapped as a
/// Dataset and an atomic one is returned bare.
/// Whether `expr` is a constructed `Dataset[…]`.
pub fn is_dataset(expr: &Expr) -> bool {
  matches!(expr, Expr::FunctionCall { name, args }
    if name == "Dataset" && !args.is_empty())
}

/// The data a Dataset wraps.
pub fn dataset_contents(expr: &Expr) -> Option<Expr> {
  match expr {
    Expr::FunctionCall { name, args }
      if name == "Dataset" && !args.is_empty() =>
    {
      Some(args[0].clone())
    }
    _ => None,
  }
}

/// Heads that a Dataset argument passes straight through to its data. The
/// second field says whether the answer is wrapped back up as a Dataset:
/// `Sort[ds]` is a Dataset, `Total[ds]` a plain value.
pub fn dataset_passthrough_head(name: &str) -> bool {
  dataset_passthrough_kind(name).is_some()
}

fn dataset_passthrough_kind(name: &str) -> Option<bool> {
  match name {
    // Reducing: answer a plain value.
    "Total" | "Mean" | "Max" | "Min" | "Median" | "Variance"
    | "StandardDeviation" | "Norm" | "Commonest" | "Quantile" | "Length"
    | "Count" | "Position" => Some(false),
    // Structure-preserving: answer another Dataset when the result is itself a
    // collection. `First[Dataset[{{1, 2}, {3}}]]` is `Dataset[{1, 2}]` while
    // `First[Dataset[{1, 2, 3}]]` is the plain `1`.
    "Sort" | "SortBy" | "Reverse" | "Take" | "Drop" | "Rest" | "Most"
    | "Accumulate" | "Select" | "Map" | "Counts" | "GroupBy"
    | "DeleteDuplicates" | "First" | "Last" | "Part" => Some(true),
    _ => None,
  }
}

/// Apply `name` to the data of every Dataset argument.
pub fn dataset_passthrough(
  name: &str,
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  let unwrapped: Vec<Expr> = args
    .iter()
    .map(|a| dataset_contents(a).unwrap_or_else(|| a.clone()))
    .collect();
  let result = crate::evaluator::evaluate_function_call_ast(name, &unwrapped)?;
  let keeps_wrapper = dataset_passthrough_kind(name).unwrap_or(false);
  if keeps_wrapper && matches!(&result, Expr::List(_) | Expr::Association(_)) {
    return Ok(dataset_ast(std::slice::from_ref(&result)));
  }
  Ok(result)
}

pub fn dataset_query(
  func_args: &[Expr],
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  let data = &func_args[0];

  let unsupported = || {
    Ok(Expr::CurriedCall {
      func: Box::new(unevaluated("Dataset", func_args)),
      args: args.to_vec(),
    })
  };

  // A key spec addresses an association, not the rows of a list — Wolfram
  // reports `Dataset::partnotapplicable` for `Dataset[{…}]["key"]`. Woxi does
  // not build that Failure object, so the query stays unevaluated rather than
  // silently looking the key up in every row.
  if matches!(args.first(), Some(Expr::String(_)))
    && matches!(data, Expr::List(_))
  {
    return unsupported();
  }

  let result = crate::functions::query_ast::apply_query(args, data)?;

  // The query engine echoes an operator it cannot apply as a curried call;
  // report that as an unevaluated dataset query instead.
  if matches!(&result, Expr::CurriedCall { .. }) {
    return unsupported();
  }

  Ok(match &result {
    Expr::List(_) | Expr::Association(_) => {
      dataset_ast(std::slice::from_ref(&result))
    }
    _ => result,
  })
}

/// Dataset[data] — wraps data with type information.
/// Dataset[data, type, metadata] — already constructed, return as-is.
pub fn dataset_ast(args: &[Expr]) -> Expr {
  if args.len() == 1 {
    let data = &args[0];
    let type_expr = infer_type(data);
    let metadata = Expr::Association(vec![]);

    Expr::FunctionCall {
      name: "Dataset".to_string(),
      args: vec![data.clone(), type_expr, metadata].into(),
    }
  } else {
    // Already has type info or other args — return as-is
    unevaluated("Dataset", args)
  }
}
