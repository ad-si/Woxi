#[allow(unused_imports)]
use super::*;

pub fn dispatch_association_functions(
  name: &str,
  args: &[Expr],
) -> Option<Result<Expr, InterpreterError>> {
  match name {
    "Keys" if args.len() == 1 => {
      return Some(crate::functions::association_ast::keys_ast(args));
    }
    "Values" if args.len() == 1 => {
      return Some(crate::functions::association_ast::values_ast(args));
    }
    "KeyDropFrom" if args.len() == 2 => {
      return Some(crate::functions::association_ast::key_drop_from_ast(args));
    }
    "KeyExistsQ" if args.len() == 2 => {
      return Some(crate::functions::association_ast::key_exists_q_ast(args));
    }
    "Lookup" if args.len() >= 2 => {
      return Some(crate::functions::association_ast::lookup_ast(args));
    }
    "KeySort" if args.len() == 1 => {
      return Some(crate::functions::association_ast::key_sort_ast(args));
    }
    "KeyUnion" if !args.is_empty() && args.len() <= 2 => {
      return Some(crate::functions::association_ast::key_union_ast(args));
    }
    "KeyValueMap" if args.len() == 2 => {
      return Some(crate::functions::association_ast::key_value_map_ast(args));
    }
    "FilterRules" if args.len() == 2 => {
      return Some(filter_rules_ast(&args[0], &args[1]));
    }
    "Dataset" if !args.is_empty() => {
      return Some(Ok(crate::functions::dataset_ast::dataset_ast(args)));
    }
    "Tabular" if !args.is_empty() => {
      return Some(Ok(crate::functions::tabular_ast::tabular_ast(args)));
    }
    "ToTabular" if !args.is_empty() => {
      return Some(Ok(crate::functions::tabular_ast::to_tabular_ast(args)));
    }
    "AssociationMap" if args.len() == 2 => {
      return Some(crate::functions::association_ast::association_map_ast(
        args,
      ));
    }
    "AssociationThread" if args.len() == 1 || args.len() == 2 => {
      return Some(
        match crate::functions::association_ast::association_thread_ast(args) {
          Err(InterpreterError::EvaluationError(msg))
            if msg.contains("same length") =>
          {
            let keys_str = if !args.is_empty() {
              crate::syntax::expr_to_string(&args[0])
            } else {
              String::new()
            };
            let vals_str = if args.len() >= 2 {
              crate::syntax::expr_to_string(&args[1])
            } else {
              String::new()
            };
            crate::emit_message(&format!(
              "AssociationThread::idim: {} and {} must have the same length.",
              keys_str, vals_str,
            ));
            Ok(Expr::FunctionCall {
              name: "AssociationThread".to_string(),
              args: args.to_vec(),
            })
          }
          other => other,
        },
      );
    }
    "Merge" if args.len() == 2 => {
      return Some(crate::functions::association_ast::merge_ast(args));
    }
    "KeyMap" if args.len() == 2 => {
      return Some(crate::functions::association_ast::key_map_ast(args));
    }
    "KeySelect" if args.len() == 2 => {
      return Some(crate::functions::association_ast::key_select_ast(args));
    }
    "KeyTake" if args.len() == 2 => {
      return Some(crate::functions::association_ast::key_take_ast(args));
    }
    "KeyDrop" if args.len() == 2 => {
      return Some(crate::functions::association_ast::key_drop_ast(args));
    }
    // Association[{a -> 1, b -> 2}] or Association[a -> 1, b -> 2]
    "Association" => {
      return Some(association_constructor(args));
    }
    _ => {}
  }
  None
}

/// Insert a key-value pair into an association, deduplicating by key (last value wins).
fn assoc_insert_dedup(pairs: &mut Vec<(Expr, Expr)>, key: Expr, val: Expr) {
  if let Some(pos) = pairs
    .iter()
    .position(|(k, _)| crate::evaluator::pattern_matching::expr_equal(k, &key))
  {
    pairs[pos].1 = val;
  } else {
    pairs.push((key, val));
  }
}

/// Convert Association[...] function call to Expr::Association
fn association_constructor(args: &[Expr]) -> Result<Expr, InterpreterError> {
  // Association[{a -> 1, b -> 2}] - single list argument containing rules
  if args.len() == 1
    && let Expr::List(items) = &args[0]
  {
    let mut pairs = Vec::new();
    for item in items {
      match item {
        Expr::Rule {
          pattern,
          replacement,
        }
        | Expr::RuleDelayed {
          pattern,
          replacement,
        } => {
          assoc_insert_dedup(
            &mut pairs,
            *pattern.clone(),
            *replacement.clone(),
          );
        }
        _ => {
          return Ok(Expr::FunctionCall {
            name: "Association".to_string(),
            args: args.to_vec(),
          });
        }
      }
    }
    return Ok(Expr::Association(pairs));
  }

  // Association[a -> 1, b -> 2] - direct rule arguments
  let mut pairs = Vec::new();
  for arg in args {
    match arg {
      Expr::Rule {
        pattern,
        replacement,
      }
      | Expr::RuleDelayed {
        pattern,
        replacement,
      } => {
        assoc_insert_dedup(&mut pairs, *pattern.clone(), *replacement.clone());
      }
      _ => {
        return Ok(Expr::FunctionCall {
          name: "Association".to_string(),
          args: args.to_vec(),
        });
      }
    }
  }
  Ok(Expr::Association(pairs))
}
