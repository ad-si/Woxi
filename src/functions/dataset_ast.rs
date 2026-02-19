use crate::syntax::Expr;

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
        args: vec![Expr::Identifier("TypeSystem`Boolean".to_string())],
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
    args: vec![Expr::Identifier(name.to_string())],
  }
}

/// Infer the type for a list of items.
fn infer_list_type(items: &[Expr]) -> Expr {
  if items.is_empty() {
    // Empty list: Vector[Atom[Expression], 0]
    return Expr::FunctionCall {
      name: "TypeSystem`Vector".to_string(),
      args: vec![ts_atom("Expression"), Expr::Integer(0)],
    };
  }

  let types: Vec<Expr> = items.iter().map(|e| infer_type(e)).collect();

  // Check if all types are the same
  let first_str = crate::syntax::expr_to_string(&types[0]);
  let all_same = types
    .iter()
    .all(|t| crate::syntax::expr_to_string(t) == first_str);

  if all_same {
    // Vector[type, count]
    Expr::FunctionCall {
      name: "TypeSystem`Vector".to_string(),
      args: vec![types[0].clone(), Expr::Integer(items.len() as i128)],
    }
  } else {
    // Tuple[{type1, type2, ...}]
    Expr::FunctionCall {
      name: "TypeSystem`Tuple".to_string(),
      args: vec![Expr::List(types)],
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
      args: vec![Expr::List(vec![]), Expr::List(vec![])],
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
          Expr::Association(inner_pairs) => infer_assoc_type(inner_pairs, false),
          _ => infer_type(v),
        }
      }
    })
    .collect();

  // Check if all value types are the same
  let first_str = crate::syntax::expr_to_string(&value_types[0]);
  let all_same = value_types
    .iter()
    .all(|t| crate::syntax::expr_to_string(t) == first_str);

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
            args: key_names,
          }],
        }
      }
    };

    Expr::FunctionCall {
      name: "TypeSystem`Assoc".to_string(),
      args: vec![
        key_type,
        value_types[0].clone(),
        Expr::Integer(pairs.len() as i128),
      ],
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
      args: vec![Expr::List(key_names), Expr::List(value_types)],
    }
  }
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
      args: vec![data.clone(), type_expr, metadata],
    }
  } else {
    // Already has type info or other args — return as-is
    Expr::FunctionCall {
      name: "Dataset".to_string(),
      args: args.to_vec(),
    }
  }
}
