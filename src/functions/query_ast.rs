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
    // `Query[{s1, s2, …}]` picks several parts at this level, keeping the
    // container type, then queries on inside each of them.
    Expr::List(specs) => match query_take_parts(data, specs) {
      Some(picked) => map_rest(rest, &picked),
      // Not a list or association: treat the spec as an ordinary operator.
      None => {
        let deeper = map_rest(rest, data)?;
        eval(&Expr::CurriedCall {
          func: Box::new(op.clone()),
          args: vec![deeper],
        })
      }
    },
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

/// `Missing[tag, detail]`.
fn missing(tag: &str, detail: &Expr) -> Expr {
  Expr::FunctionCall {
    name: "Missing".to_string(),
    args: vec![Expr::String(tag.to_string()), detail.clone()].into(),
  }
}

/// Resolve a 1-based (possibly negative) index against a container of `len`
/// elements, returning the 0-based offset.
fn resolve_index(n: i128, len: usize) -> Option<usize> {
  let len = len as i128;
  let i = if n < 0 { len + n } else { n - 1 };
  (n != 0 && i >= 0 && i < len).then_some(i as usize)
}

/// `Query[{s1, s2, …}][data]` gathers the parts named by the sub-specs, in
/// the order given, keeping the container type: a list of elements for list
/// data, a sub-association for association data. Where `Part` would report
/// `Part::partw` / `Part::pspec1`, Query is lenient and leaves a `Missing[…]`
/// in place instead. Returns `None` for data that is neither a list nor an
/// association, so the caller can treat the spec as an ordinary operator.
fn query_take_parts(data: &Expr, specs: &[Expr]) -> Option<Expr> {
  match data {
    Expr::List(items) => {
      let picked = specs
        .iter()
        .map(|s| match s {
          Expr::Integer(n) => match resolve_index(*n, items.len()) {
            Some(i) => items[i].clone(),
            None => missing("PartAbsent", s),
          },
          _ => missing("PartInvalid", s),
        })
        .collect::<Vec<_>>();
      Some(Expr::List(picked.into()))
    }
    Expr::Association(pairs) => {
      let mut picked: Vec<(Expr, Expr)> = Vec::with_capacity(specs.len());
      for s in specs {
        // An integer names a position; anything else names a key.
        let (key, value) = if let Expr::Integer(n) = s {
          // An out-of-range position fails the whole query here, unlike the
          // per-element Missing a list spec produces.
          match resolve_index(*n, pairs.len()) {
            Some(i) => pairs[i].clone(),
            None => {
              return Some(missing(
                "PartAbsent",
                &Expr::List(specs.to_vec().into()),
              ));
            }
          }
        } else {
          let found = pairs
            .iter()
            .find(|(k, _)| {
              crate::syntax::expr_to_string(k)
                == crate::syntax::expr_to_string(s)
            })
            .map(|(_, v)| v.clone());
          (s.clone(), found.unwrap_or_else(|| missing("KeyAbsent", s)))
        };
        // Repeating a key does not repeat the entry.
        if !picked.iter().any(|(k, _)| {
          crate::syntax::expr_to_string(k)
            == crate::syntax::expr_to_string(&key)
        }) {
          picked.push((key, value));
        }
      }
      Some(Expr::Association(picked))
    }
    _ => None,
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
