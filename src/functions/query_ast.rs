//! Query[op1, op2, ...][data] — successive-level data queries over
//! lists and associations. Each operator works at its level:
//! descending operators (All, integer parts, string keys, and list
//! operators like Select/SortBy/Take/Drop) transform on the way down,
//! everything else (Total, Mean, arbitrary functions) applies on the
//! way up to the results of the deeper levels.

use crate::InterpreterError;
use crate::syntax::Expr;

pub fn apply_query(
  ops: &[Expr],
  data: &Expr,
) -> Result<Expr, InterpreterError> {
  let Some((op, rest)) = ops.split_first() else {
    return Ok(data.clone());
  };
  let eval = crate::evaluator::evaluate_expr_to_expr;
  match op {
    Expr::Identifier(s) if s == "All" => map_rest(rest, data),
    Expr::Integer(_) => {
      let part = eval(&Expr::FunctionCall {
        name: "Part".to_string(),
        args: vec![data.clone(), op.clone()].into(),
      })?;
      apply_query(rest, &part)
    }
    Expr::String(_) => {
      let value = eval(&Expr::FunctionCall {
        name: "Lookup".to_string(),
        args: vec![data.clone(), op.clone()].into(),
      })?;
      apply_query(rest, &value)
    }
    Expr::FunctionCall { name, .. }
      if matches!(
        name.as_str(),
        "Select"
          | "SortBy"
          | "TakeLargest"
          | "TakeSmallest"
          | "TakeLargestBy"
          | "TakeSmallestBy"
          | "DeleteDuplicates"
          | "DeleteMissing"
      ) =>
    {
      // Descending list operator: apply it here (data prepended to
      // its held arguments, sidestepping per-function curried support),
      // then run the rest of the spec over the surviving elements
      let (op_name, op_args) = match op {
        Expr::FunctionCall { name, args } => (name.clone(), args.to_vec()),
        _ => unreachable!("guarded by the match arm"),
      };
      let mut full_args = vec![data.clone()];
      full_args.extend(op_args);
      let transformed = eval(&Expr::FunctionCall {
        name: op_name,
        args: full_args.into(),
      })?;
      map_rest(rest, &transformed)
    }
    _ => {
      // Ascending: deeper levels first, then apply the operator
      let deeper = map_rest(rest, data)?;
      eval(&Expr::CurriedCall {
        func: Box::new(op.clone()),
        args: vec![deeper],
      })
    }
  }
}

/// Apply the remaining spec one level down: mapped over list elements
/// and association values, or directly when the data is atomic.
fn map_rest(rest: &[Expr], data: &Expr) -> Result<Expr, InterpreterError> {
  if rest.is_empty() {
    return Ok(data.clone());
  }
  match data {
    Expr::List(items) => {
      let mapped: Result<Vec<Expr>, InterpreterError> =
        items.iter().map(|e| apply_query(rest, e)).collect();
      Ok(Expr::List(mapped?.into()))
    }
    // Association literal `<|k -> v, …|>`: map the rest of the spec over the
    // values, keeping the keys and the association structure.
    Expr::Association(pairs) => {
      let mapped: Result<Vec<(Expr, Expr)>, InterpreterError> = pairs
        .iter()
        .map(|(k, v)| Ok((k.clone(), apply_query(rest, v)?)))
        .collect();
      Ok(Expr::Association(mapped?))
    }
    Expr::FunctionCall { name, args } if name == "Association" => {
      let mapped: Result<Vec<Expr>, InterpreterError> = args
        .iter()
        .map(|rule| match rule {
          Expr::Rule {
            pattern,
            replacement,
          } => Ok(Expr::Rule {
            pattern: pattern.clone(),
            replacement: Box::new(apply_query(rest, replacement)?),
          }),
          other => Ok(other.clone()),
        })
        .collect();
      Ok(Expr::FunctionCall {
        name: "Association".to_string(),
        args: mapped?.into(),
      })
    }
    other => apply_query(rest, other),
  }
}
