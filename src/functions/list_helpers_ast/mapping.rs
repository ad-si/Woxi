#[allow(unused_imports)]
use super::utilities::*;
#[allow(unused_imports)]
use super::*;
use crate::syntax::ComparisonOp;

/// AST-based Map: apply function to each element of a list or association.
/// Map[f, {a, b, c}] -> {f[a], f[b], f[c]}
/// Map[f, <|a -> 1, b -> 2|>] -> <|a -> f[1], b -> f[2]|>
pub fn map_ast(func: &Expr, list: &Expr) -> Result<Expr, InterpreterError> {
  match list {
    Expr::List(items) => {
      let results: Result<Vec<Expr>, _> = items
        .iter()
        .map(|item| apply_func_ast(func, item))
        .collect();
      let results: Vec<Expr> =
        results?.into_iter().filter(|e| !is_nothing(e)).collect();
      Ok(Expr::List(results.into()))
    }
    Expr::Association(items) => {
      // Map over association applies function to values only. When the value
      // is a RuleDelayed marker (`{pattern == key, replacement}`) — denoting
      // the original entry was `key :> v` — apply func to the replacement and
      // re-wrap the marker so the `:>` form survives.
      let results: Result<Vec<(Expr, Expr)>, InterpreterError> = items
        .iter()
        .map(|(key, val)| {
          if let Expr::RuleDelayed {
            pattern,
            replacement,
          } = val
            && crate::syntax::assoc_marker_matches(key, pattern)
          {
            let new_repl = apply_func_ast(func, replacement)?;
            return Ok((
              key.clone(),
              Expr::RuleDelayed {
                pattern: pattern.clone(),
                replacement: Box::new(new_repl),
              },
            ));
          }
          let new_val = apply_func_ast(func, val)?;
          Ok((key.clone(), new_val))
        })
        .collect();
      Ok(Expr::Association(results?))
    }
    // Map over a rank-1 SparseArray applies f to each element while keeping the
    // sparse structure: f is applied to every stored value and to the default
    // value (Map[f, SparseArray[..]] -> SparseArray[.., f[default], {.., {f[v1],
    // ..}}]), matching Wolfram. Without this, the generic head branch below would
    // map f over the internal representation (Automatic, dims, ...) and corrupt
    // it. Higher-rank arrays map over their (sub-array) rows, so we densify and
    // map as a list — value-correct via Normal.
    Expr::FunctionCall {
      name,
      args: sa_args,
    } if name == "SparseArray" => {
      let canonical =
        crate::functions::list_helpers_ast::sparse_array_normalize_ast(
          sa_args,
        )?;
      if let Expr::FunctionCall { name: cn, args: ca } = &canonical
        && cn == "SparseArray"
        && ca.len() == 4
        && matches!(&ca[0], Expr::Identifier(s) if s == "Automatic")
        && let Expr::List(dims) = &ca[1]
        && dims.len() == 1
        && let Expr::List(structure) = &ca[3]
        && structure.len() == 3
        && let Expr::List(values) = &structure[2]
      {
        let new_default = apply_func_ast(func, &ca[2])?;
        let new_values: Result<Vec<Expr>, _> =
          values.iter().map(|v| apply_func_ast(func, v)).collect();
        let new_structure = Expr::List(
          vec![
            structure[0].clone(),
            structure[1].clone(),
            Expr::List(new_values?.into()),
          ]
          .into(),
        );
        return Ok(Expr::FunctionCall {
          name: "SparseArray".to_string(),
          args: vec![ca[0].clone(), ca[1].clone(), new_default, new_structure]
            .into(),
        });
      }
      // Rank >= 2 (or an unrecognized structure): densify and map as a list.
      let dense =
        crate::functions::list_helpers_ast::sparse_array_ast(sa_args)?;
      map_ast(func, &dense)
    }
    // Rational / Complex are atoms: Map at level 1 returns them unchanged.
    _ if crate::functions::predicate_ast::is_atomic_number(list) => {
      Ok(list.clone())
    }
    _ => {
      // For any compound expression, decompose into head + children,
      // apply func to each child, and reconstruct.
      // E.g. Map[f, Power[x, 2]] -> Power[f[x], f[2]]
      use crate::functions::expr_form::{ExprForm, decompose_expr};
      match decompose_expr(list) {
        ExprForm::Composite { head, children } => {
          let mapped: Result<Vec<Expr>, _> = children
            .iter()
            .map(|child| apply_func_ast(func, child))
            .collect();
          crate::evaluator::evaluate_function_call_ast(&head, &mapped?)
        }
        ExprForm::Atom(_) => {
          // Atomic expression: Map[f, atom] returns atom unchanged
          Ok(list.clone())
        }
      }
    }
  }
}

/// Map[f, expr, levelspec] - apply f at specified levels
pub fn map_with_level_ast(
  func: &Expr,
  expr: &Expr,
  level_spec: &Expr,
) -> Result<Expr, InterpreterError> {
  // Check for Infinity identifier
  let is_infinity = |e: &Expr| -> bool {
    matches!(e, Expr::Identifier(s) if s == "Infinity" || s == "DirectedInfinity")
      || matches!(e, Expr::FunctionCall { name, args }
        if name == "DirectedInfinity" && args.len() == 1
        && matches!(&args[0], Expr::Integer(1)))
  };

  // Parse level spec: {n} = exactly level n, {min, max} = range, n = {1, n}
  // Infinity means all levels, negative levels count from leaves
  // Also handle Heads -> True option
  let (min_level, max_level, _heads) = match level_spec {
    Expr::Integer(n) => (1i64, *n as i64, false),
    _ if is_infinity(level_spec) => (1i64, i64::MAX, false),
    Expr::List(items) if items.len() == 1 => {
      if is_infinity(&items[0]) {
        (1i64, i64::MAX, false)
      } else if let Some(n) = expr_to_i128(&items[0]) {
        (n as i64, n as i64, false)
      } else {
        return Ok(Expr::FunctionCall {
          name: "Map".to_string(),
          args: vec![func.clone(), expr.clone(), level_spec.clone()].into(),
        });
      }
    }
    Expr::List(items) if items.len() == 2 => {
      let min = expr_to_i128(&items[0]).unwrap_or(0) as i64;
      let max = if is_infinity(&items[1]) {
        i64::MAX
      } else {
        expr_to_i128(&items[1]).unwrap_or(0) as i64
      };
      (min, max, false)
    }
    // Heads -> True option (Expr::Rule variant from -> syntax)
    Expr::Rule {
      pattern,
      replacement,
    } => {
      if matches!(pattern.as_ref(), Expr::Identifier(s) if s == "Heads")
        && matches!(replacement.as_ref(), Expr::Identifier(v) if v == "True")
      {
        return map_with_heads(func, expr);
      }
      return Ok(Expr::FunctionCall {
        name: "Map".to_string(),
        args: vec![func.clone(), expr.clone(), level_spec.clone()].into(),
      });
    }
    // Also handle FunctionCall("Rule", ...) form
    Expr::FunctionCall { name, args } if name == "Rule" && args.len() == 2 => {
      if let Expr::Identifier(s) = &args[0]
        && s == "Heads"
        && matches!(&args[1], Expr::Identifier(v) if v == "True")
      {
        return map_with_heads(func, expr);
      }
      return Ok(Expr::FunctionCall {
        name: "Map".to_string(),
        args: vec![func.clone(), expr.clone(), level_spec.clone()].into(),
      });
    }
    _ => {
      return Ok(Expr::FunctionCall {
        name: "Map".to_string(),
        args: vec![func.clone(), expr.clone(), level_spec.clone()].into(),
      });
    }
  };

  // If any level bound is negative, we need depth-aware traversal
  if min_level < 0 || max_level < 0 {
    map_at_depth_negative(func, expr, 0, min_level, max_level)
  } else {
    map_at_depth(func, expr, 0, min_level, max_level)
  }
}

/// Canonical (head, children) for Map's level traversal.
///
/// Returns `None` for atoms — including Rational/Complex (which are atomic)
/// and Associations (whose keys are not child elements, handled separately).
/// Everything else is decomposed to its FullForm so operator and special
/// forms (`x^2` = Power[x, 2], `a == b` = Equal[a, b], `a -> b` = Rule[a, b],
/// `!a` = Not[a], …) are descended into, just like Level and Depth.
fn map_decompose(expr: &Expr) -> Option<(String, Vec<Expr>)> {
  use crate::functions::expr_form::{ExprForm, decompose_expr};
  if crate::functions::predicate_ast::is_atomic_number(expr)
    || matches!(expr, Expr::Association(_))
  {
    return None;
  }
  match decompose_expr(expr) {
    ExprForm::Atom(_) => None,
    ExprForm::Composite { head, children } => Some((head, children)),
  }
}

/// Rebuild a mapped composite from its canonical head and new children.
fn rewrap(head: &str, children: Vec<Expr>) -> Expr {
  // Relational heads must be rebuilt as `Expr::Comparison` so they render
  // infix (`f[a] == f[b]`), not as `Equal[f[a], f[b]]`.
  let cmp_op = match head {
    "Equal" => Some(ComparisonOp::Equal),
    "Unequal" => Some(ComparisonOp::NotEqual),
    "Less" => Some(ComparisonOp::Less),
    "LessEqual" => Some(ComparisonOp::LessEqual),
    "Greater" => Some(ComparisonOp::Greater),
    "GreaterEqual" => Some(ComparisonOp::GreaterEqual),
    "SameQ" => Some(ComparisonOp::SameQ),
    "UnsameQ" => Some(ComparisonOp::UnsameQ),
    _ => None,
  };
  if let Some(op) = cmp_op
    && children.len() >= 2
  {
    let operators = vec![op; children.len() - 1];
    return Expr::Comparison {
      operands: children,
      operators,
    };
  }
  if head == "List" {
    Expr::List(children.into())
  } else {
    Expr::FunctionCall {
      name: head.to_string(),
      args: children.into(),
    }
  }
}

/// Compute the Wolfram-style depth of an expression.
/// Atoms have depth 1, compound expressions have depth 1 + max child depth.
fn expr_depth(expr: &Expr) -> i64 {
  match map_decompose(expr) {
    None => 1,
    Some((_, children)) => {
      1 + children.iter().map(expr_depth).max().unwrap_or(0)
    }
  }
}

/// Map with negative level support. Negative levels count from leaves:
/// level -1 = atoms, level -2 = expressions whose max child depth is 1, etc.
/// The negative level of a node is -depth(node).
fn map_at_depth_negative(
  func: &Expr,
  expr: &Expr,
  current_depth: i64,
  min_level: i64,
  max_level: i64,
) -> Result<Expr, InterpreterError> {
  let neg_level = -(expr_depth(expr));

  // Atoms (incl. Rational/Complex) and Associations are not descended into;
  // operator/special forms are decomposed to their canonical FullForm.
  let result = if let Some((head, kids)) = map_decompose(expr) {
    let mapped: Result<Vec<Expr>, _> = kids
      .iter()
      .map(|item| {
        map_at_depth_negative(
          func,
          item,
          current_depth + 1,
          min_level,
          max_level,
        )
      })
      .collect();
    rewrap(&head, mapped?)
  } else {
    expr.clone()
  };

  // Check if this node should have f applied.
  // Each node has both a positive level (current_depth from root) and
  // a negative level (neg_level = -depth, counting from leaves).
  // A node matches if its level falls within [min_level, max_level]
  // using the appropriate sign convention for each bound.
  let meets_min = if min_level >= 0 {
    current_depth >= min_level
  } else {
    neg_level >= min_level
  };
  let meets_max = if max_level >= 0 {
    current_depth <= max_level
  } else {
    neg_level <= max_level
  };

  if meets_min && meets_max {
    apply_func_ast(func, &result)
  } else {
    Ok(result)
  }
}

/// Recursively map at specified levels (bottom-up)
fn map_at_depth(
  func: &Expr,
  expr: &Expr,
  current_depth: i64,
  min_level: i64,
  max_level: i64,
) -> Result<Expr, InterpreterError> {
  // Recurse into composite expression children. `Rule`/`RuleDelayed` need
  // their own arms so a `Map[f, _, {2}]` can reach the pattern/replacement
  // pair (level 2 of `Q[a->1, b->2]` is `a, 1, b, 2`, not just `a->1, b->2`).
  let recurse = |child: &Expr| -> Result<Expr, InterpreterError> {
    map_at_depth(func, child, current_depth + 1, min_level, max_level)
  };
  let result = match expr {
    // Rational / Complex are atoms: do not descend into their parts.
    _ if crate::functions::predicate_ast::is_atomic_number(expr) => {
      expr.clone()
    }
    Expr::List(items) => {
      let mapped: Result<Vec<Expr>, _> = items.iter().map(recurse).collect();
      Expr::List(mapped?.into())
    }
    Expr::FunctionCall { name, args } => {
      let mapped: Result<Vec<Expr>, _> = args.iter().map(recurse).collect();
      Expr::FunctionCall {
        name: name.clone(),
        args: mapped?.into(),
      }
    }
    Expr::Rule {
      pattern,
      replacement,
    } => Expr::Rule {
      pattern: Box::new(recurse(pattern)?),
      replacement: Box::new(recurse(replacement)?),
    },
    Expr::RuleDelayed {
      pattern,
      replacement,
    } => Expr::RuleDelayed {
      pattern: Box::new(recurse(pattern)?),
      replacement: Box::new(recurse(replacement)?),
    },
    Expr::Association(items) => {
      // Recurse through values; the key is not a child element of the
      // Association (matches Wolfram's level numbering where `<|a->1|>`
      // has only `1` at level 1, not `a`). When the value is a
      // RuleDelayed marker, descend into the replacement and re-wrap.
      let mapped: Result<Vec<(Expr, Expr)>, InterpreterError> = items
        .iter()
        .map(|(k, v)| {
          if let Expr::RuleDelayed {
            pattern,
            replacement,
          } = v
            && crate::syntax::assoc_marker_matches(k, pattern)
          {
            let new_repl = recurse(replacement)?;
            return Ok((
              k.clone(),
              Expr::RuleDelayed {
                pattern: pattern.clone(),
                replacement: Box::new(new_repl),
              },
            ));
          }
          Ok((k.clone(), recurse(v)?))
        })
        .collect();
      Expr::Association(mapped?)
    }
    // Operator and special forms (Power/Sqrt, Comparison, Not, …) are
    // decomposed to their FullForm so Map descends into them.
    _ => match map_decompose(expr) {
      None => expr.clone(),
      Some((head, children)) => {
        let mapped: Result<Vec<Expr>, _> =
          children.iter().map(recurse).collect();
        rewrap(&head, mapped?)
      }
    },
  };

  // Apply f at this depth if in range
  if current_depth >= min_level && current_depth <= max_level {
    apply_func_ast(func, &result)
  } else {
    Ok(result)
  }
}

/// Map[f, expr, Heads -> True]: apply f to head and all level-1 elements
fn map_with_heads(func: &Expr, expr: &Expr) -> Result<Expr, InterpreterError> {
  match expr {
    Expr::List(items) => {
      let mapped: Result<Vec<Expr>, _> = items
        .iter()
        .map(|item| apply_func_ast(func, item))
        .collect();
      // Apply f to the head (List)
      let new_head =
        apply_func_ast(func, &Expr::Identifier("List".to_string()))?;
      Ok(Expr::CurriedCall {
        func: Box::new(new_head),
        args: mapped?,
      })
    }
    Expr::FunctionCall { name, args } => {
      let mapped: Result<Vec<Expr>, _> =
        args.iter().map(|item| apply_func_ast(func, item)).collect();
      // Apply f to the head
      let new_head = apply_func_ast(func, &Expr::Identifier(name.clone()))?;
      Ok(Expr::CurriedCall {
        func: Box::new(new_head),
        args: mapped?,
      })
    }
    _ => apply_func_ast(func, expr),
  }
}

/// MapAt[f, expr, spec] — apply f at the given positions. Supports
/// integer positions (0 wraps the head: f[List][a, b, c]), nested paths,
/// lists of paths, All, and Span; invalid positions emit MapAt::partw
/// (always the full path with braces and the original expression) and
/// non-position specs emit MapAt::psl, both staying unevaluated.
/// Extract the target key from a MapAt position specification for an
/// association: `Key[k]`, a bare key (`String`/`Identifier`), or a
/// single-element wrapper `{Key[k]}` / `{k}`. Integer positions return `None`
/// (they are handled positionally).
fn extract_assoc_key(spec: &Expr) -> Option<&Expr> {
  match spec {
    Expr::FunctionCall { name, args } if name == "Key" && args.len() == 1 => {
      Some(&args[0])
    }
    Expr::String(_) | Expr::Identifier(_) => Some(spec),
    Expr::List(items) if items.len() == 1 => extract_assoc_key(&items[0]),
    _ => None,
  }
}

pub fn map_at_ast(
  func: &Expr,
  list: &Expr,
  pos_spec: &Expr,
) -> Result<Expr, InterpreterError> {
  map_at_ast_named("MapAt", func, list, pos_spec)
}

/// Like [`map_at_ast`], but `caller` names the symbol used in emitted
/// diagnostics (`caller::partw` / `caller::psl`). Delegates such as
/// `ReplaceAt`, which are implemented as `MapAt[Replace[#, rules] &, …]`,
/// pass their own name so messages read `ReplaceAt::partw` rather than
/// `MapAt::partw`.
pub fn map_at_ast_named(
  caller: &str,
  func: &Expr,
  list: &Expr,
  pos_spec: &Expr,
) -> Result<Expr, InterpreterError> {
  let unevaluated = || Expr::FunctionCall {
    name: "MapAt".to_string(),
    args: vec![func.clone(), list.clone(), pos_spec.clone()].into(),
  };
  let partw = |path: &[i128]| {
    crate::emit_message(&format!(
      "{caller}::partw: Part {{{}}} of {} does not exist.",
      path
        .iter()
        .map(|p| p.to_string())
        .collect::<Vec<_>>()
        .join(", "),
      crate::syntax::format_expr(list, crate::syntax::ExprForm::Output)
    ));
  };
  let psl = || {
    crate::emit_message(&format!(
      "{caller}::psl: Position specification {} in {} is not a machine-sized integer or a list of machine-sized integers.",
      crate::syntax::format_expr(pos_spec, crate::syntax::ExprForm::Output),
      crate::syntax::format_expr(
        &unevaluated(),
        crate::syntax::ExprForm::Output
      )
    ));
  };

  // Association: integer-position MapAt transforms the value at that position.
  if let Expr::Association(pairs) = list
    && let Some(n) = expr_to_i128(pos_spec)
  {
    let len = pairs.len() as i128;
    let idx = if n < 0 {
      (len + n) as usize
    } else {
      (n - 1) as usize
    };
    if idx >= pairs.len() {
      return Ok(unevaluated());
    }
    let mut new_pairs = pairs.clone();
    let new_value = apply_func_ast(func, &new_pairs[idx].1)?;
    new_pairs[idx].1 = new_value;
    return Ok(Expr::Association(new_pairs));
  }

  // Association: key-based MapAt transforms the value stored at that key.
  // Accepts `Key[k]`, a bare key, and single-element wrappers `{Key[k]}` /
  // `{k}`.
  if let Expr::Association(pairs) = list
    && let Some(key) = extract_assoc_key(pos_spec)
  {
    let key_str = crate::syntax::expr_to_string(key);
    if let Some(idx) = pairs
      .iter()
      .position(|(k, _)| crate::syntax::expr_to_string(k) == key_str)
    {
      let mut new_pairs = pairs.clone();
      new_pairs[idx].1 = apply_func_ast(func, &new_pairs[idx].1)?;
      return Ok(Expr::Association(new_pairs));
    }
    // Key not present: emit partw and leave unevaluated.
    crate::emit_message(&format!(
      "{caller}::partw: Part {} of {} does not exist.",
      crate::syntax::format_expr(pos_spec, crate::syntax::ExprForm::Output),
      crate::syntax::format_expr(list, crate::syntax::ExprForm::Output)
    ));
    return Ok(unevaluated());
  }

  // Apply f along one validated path (positions refer to the current
  // result; 0 as the final index wraps the head). Err(()) = invalid.
  fn apply_path(
    func: &Expr,
    expr: &Expr,
    path: &[i128],
  ) -> Result<Result<Expr, ()>, InterpreterError> {
    let (items, head): (Vec<Expr>, Option<String>) =
      match super::element_access::parts_and_head(expr) {
        Some(p) => p,
        None => return Ok(Err(())),
      };
    let items = items.as_slice();
    let len = items.len() as i128;
    let n = path[0];
    if n == 0 && path.len() == 1 {
      // Wrap the head: f[head][args...]
      let head_expr =
        Expr::Identifier(head.clone().unwrap_or_else(|| "List".to_string()));
      let wrapped = apply_func_ast(func, &head_expr)?;
      return Ok(Ok(Expr::CurriedCall {
        func: Box::new(wrapped),
        args: items.to_vec(),
      }));
    }
    let idx = if n > 0 { n - 1 } else { len + n };
    if idx < 0 || idx >= len {
      return Ok(Err(()));
    }
    let idx = idx as usize;
    let mut new_items = items.to_vec();
    if path.len() == 1 {
      new_items[idx] = apply_func_ast(func, &items[idx])?;
    } else {
      match apply_path(func, &items[idx], &path[1..])? {
        Ok(v) => new_items[idx] = v,
        Err(()) => return Ok(Err(())),
      }
    }
    Ok(Ok(match head {
      Some(h) => Expr::FunctionCall {
        name: h,
        args: new_items.into(),
      },
      None => Expr::List(new_items.into()),
    }))
  }

  let (items, head): (Vec<Expr>, Option<String>) =
    match super::element_access::parts_and_head(list) {
      Some(p) => p,
      None => {
        // Atomic subject: partw with the spec (when it is positional)
        match pos_spec {
          Expr::Integer(n) => partw(&[*n]),
          Expr::List(parts) => {
            if let Some(path) = parts
              .iter()
              .map(expr_to_i128)
              .collect::<Option<Vec<i128>>>()
            {
              partw(&path);
            } else {
              psl();
            }
          }
          _ => psl(),
        }
        return Ok(unevaluated());
      }
    };
  let items = items.as_slice();
  let len = items.len() as i128;
  let wrap = |v: Vec<Expr>| match &head {
    Some(h) => Expr::FunctionCall {
      name: h.clone(),
      args: v.into(),
    },
    None => Expr::List(v.into()),
  };

  match pos_spec {
    // All: map every top-level element
    Expr::Identifier(s) if s == "All" => {
      let mut new_items = Vec::with_capacity(items.len());
      for item in items {
        new_items.push(apply_func_ast(func, item)?);
      }
      Ok(wrap(new_items))
    }
    Expr::Integer(_) | Expr::BigInteger(_) => {
      let n = expr_to_i128(pos_spec).unwrap();
      match apply_path(func, list, &[n])? {
        Ok(v) => Ok(v),
        Err(()) => {
          partw(&[n]);
          Ok(unevaluated())
        }
      }
    }
    Expr::List(pos_list) if !pos_list.is_empty() => {
      let all_lists = pos_list.iter().all(|p| matches!(p, Expr::List(_)));
      if all_lists {
        // Multiple paths, applied in order
        let mut paths: Vec<Vec<i128>> = Vec::with_capacity(pos_list.len());
        for p in pos_list {
          let Expr::List(inner) = p else { unreachable!() };
          match inner
            .iter()
            .map(expr_to_i128)
            .collect::<Option<Vec<i128>>>()
          {
            Some(path) if !path.is_empty() => paths.push(path),
            _ => {
              psl();
              return Ok(unevaluated());
            }
          }
        }
        let mut result = list.clone();
        for path in &paths {
          match apply_path(func, &result, path)? {
            Ok(v) => result = v,
            Err(()) => {
              partw(path);
              return Ok(unevaluated());
            }
          }
        }
        Ok(result)
      } else if let Some(path) = pos_list
        .iter()
        .map(expr_to_i128)
        .collect::<Option<Vec<i128>>>()
      {
        match apply_path(func, list, &path)? {
          Ok(v) => Ok(v),
          Err(()) => {
            partw(&path);
            Ok(unevaluated())
          }
        }
      } else {
        psl();
        Ok(unevaluated())
      }
    }
    // Span[start, end] or Span[start, end, step]
    Expr::FunctionCall {
      name,
      args: span_args,
    } if name == "Span" && (span_args.len() == 2 || span_args.len() == 3) => {
      let start_raw = expr_to_i128(&span_args[0]).unwrap_or(1);
      let step = if span_args.len() == 3 {
        expr_to_i128(&span_args[2]).unwrap_or(1)
      } else {
        1
      };
      let end_raw = match &span_args[1] {
        Expr::Identifier(s) if s == "All" => len,
        other => expr_to_i128(other).unwrap_or(len),
      };
      let start_idx = if start_raw < 0 {
        (len + start_raw) as usize
      } else {
        (start_raw - 1) as usize
      };
      let end_idx = if end_raw < 0 {
        (len + end_raw) as usize
      } else {
        (end_raw - 1) as usize
      };
      if step <= 0 {
        return Ok(unevaluated());
      }
      let step = step as usize;
      let mut new_items = items.to_vec();
      let mut i = start_idx;
      while i <= end_idx && i < new_items.len() {
        new_items[i] = apply_func_ast(func, &items[i])?;
        i += step;
      }
      Ok(wrap(new_items))
    }
    _ => {
      psl();
      Ok(unevaluated())
    }
  }
}

/// AST-based MapIndexed: apply function with index to each element.
/// MapIndexed[f, {a, b, c}] -> {f[a, {1}], f[b, {2}], f[c, {3}]}
pub fn map_indexed_ast(
  func: &Expr,
  list: &Expr,
) -> Result<Expr, InterpreterError> {
  // MapIndexed threads over any non-atomic head, keeping it: List gives a
  // List, and a general head[...] gives head[f[e1, {1}], f[e2, {2}], ...].
  let (head, items): (Option<&String>, &[Expr]) = match list {
    Expr::List(items) => (None, items),
    Expr::FunctionCall { name, args } => (Some(name), args),
    // On an association, f is applied to each value with the index {Key[key]},
    // keeping the keys: <|k -> f[v, {Key[k]}], ...|>.
    Expr::Association(pairs) => {
      let new_pairs: Result<Vec<(Expr, Expr)>, _> = pairs
        .iter()
        .map(|(k, v)| {
          let index = Expr::List(
            vec![Expr::FunctionCall {
              name: "Key".to_string(),
              args: vec![k.clone()].into(),
            }]
            .into(),
          );
          apply_func_to_two_args(func, v, &index).map(|r| (k.clone(), r))
        })
        .collect();
      return Ok(Expr::Association(new_pairs?));
    }
    _ => {
      return Ok(Expr::FunctionCall {
        name: "MapIndexed".to_string(),
        args: vec![func.clone(), list.clone()].into(),
      });
    }
  };

  let results: Result<Vec<Expr>, _> = items
    .iter()
    .enumerate()
    .map(|(i, item)| {
      let index = Expr::List(vec![Expr::Integer((i + 1) as i128)].into());
      apply_func_to_two_args(func, item, &index)
    })
    .collect();
  match head {
    // `Nothing` is auto-removed only from Lists, not general heads.
    None => {
      let results: Vec<Expr> =
        results?.into_iter().filter(|e| !is_nothing(e)).collect();
      Ok(Expr::List(results.into()))
    }
    Some(h) => Ok(Expr::FunctionCall {
      name: h.clone(),
      args: results?.into(),
    }),
  }
}

/// Detect a Heads -> True option, either as Expr::Rule or Expr::FunctionCall.
fn is_heads_true_option(e: &Expr) -> bool {
  match e {
    Expr::Rule {
      pattern,
      replacement,
    } => {
      matches!(pattern.as_ref(), Expr::Identifier(s) if s == "Heads")
        && matches!(replacement.as_ref(), Expr::Identifier(v) if v == "True")
    }
    Expr::FunctionCall { name, args } if name == "Rule" && args.len() == 2 => {
      matches!(&args[0], Expr::Identifier(s) if s == "Heads")
        && matches!(&args[1], Expr::Identifier(v) if v == "True")
    }
    _ => false,
  }
}

/// MapIndexed with level spec: MapIndexed[f, expr, levelspec]
/// Applies f to elements at the specified level, passing {position indices} as second arg.
/// Also accepts a bare 'Heads -> True' as the 3rd argument, in which case the
/// default level {1} is used but the head is included with index 0 appended.
/// As a 4th argument, 'Heads -> True' combines with the levelspec.
pub fn map_indexed_with_level_ast(
  func: &Expr,
  expr: &Expr,
  level_spec: &Expr,
) -> Result<Expr, InterpreterError> {
  // 'Heads -> True' as the 3rd arg ⇒ default level {1}, with heads.
  if is_heads_true_option(level_spec) {
    return map_indexed_at_depth_heads(func, expr, 0, 1, 1, &[]);
  }
  // `{-1}` selects every atomic leaf — handle it directly so we don't
  // need a depth-from-leaves walk in the general routine.
  if is_neg_one_levelspec(level_spec) {
    return map_indexed_atoms(func, expr, &[], false);
  }
  // Parse level spec: {n} = exactly level n
  let (min_level, max_level) = match level_spec {
    Expr::Integer(n) => (1i64, *n as i64),
    Expr::List(items) if items.len() == 1 => {
      if let Some(n) = expr_to_i128(&items[0]) {
        (n as i64, n as i64)
      } else {
        return Ok(Expr::FunctionCall {
          name: "MapIndexed".to_string(),
          args: vec![func.clone(), expr.clone(), level_spec.clone()].into(),
        });
      }
    }
    Expr::List(items) if items.len() == 2 => {
      let min = expr_to_i128(&items[0]).unwrap_or(0) as i64;
      let max = expr_to_i128(&items[1]).unwrap_or(0) as i64;
      (min, max)
    }
    _ => {
      return Ok(Expr::FunctionCall {
        name: "MapIndexed".to_string(),
        args: vec![func.clone(), expr.clone(), level_spec.clone()].into(),
      });
    }
  };

  map_indexed_at_depth(func, expr, 0, min_level, max_level, &[])
}

/// Detect the special `{-1}` levelspec (atomic leaves). Negative levels in
/// general count "depth from leaves", but `{-1}` is by far the common
/// case in the mathics doctest suite, so we special-case it without a
/// full depth-from-leaves traversal.
fn is_neg_one_levelspec(level_spec: &Expr) -> bool {
  match level_spec {
    Expr::List(items) if items.len() == 1 => {
      matches!(&items[0], Expr::Integer(n) if *n == -1)
    }
    _ => false,
  }
}

/// Apply `func` to every atomic leaf, threading the position index list.
/// When `with_heads` is true the head of every compound node is also
/// visited, with `0` appended to its position — matching `Heads -> True`.
fn map_indexed_atoms(
  func: &Expr,
  expr: &Expr,
  position: &[i128],
  with_heads: bool,
) -> Result<Expr, InterpreterError> {
  let (children, head_name_opt): (Option<&[Expr]>, Option<String>) = match expr
  {
    Expr::List(items) => (Some(items.as_slice()), Some("List".to_string())),
    Expr::FunctionCall { name, args } => {
      (Some(args.as_slice()), Some(name.clone()))
    }
    _ => (None, None),
  };
  if let (Some(items), Some(head_name)) = (children, head_name_opt) {
    let mut mapped: Vec<Expr> = Vec::with_capacity(items.len());
    for (i, item) in items.iter().enumerate() {
      let mut child_pos = position.to_vec();
      child_pos.push((i + 1) as i128);
      mapped.push(map_indexed_atoms(func, item, &child_pos, with_heads)?);
    }
    if !with_heads {
      return Ok(if head_name == "List" {
        Expr::List(mapped.into())
      } else {
        Expr::FunctionCall {
          name: head_name,
          args: mapped.into(),
        }
      });
    }
    // Heads-True: rewrite the head as `func[head, append(position, 0)]`.
    let mut head_pos = position.to_vec();
    head_pos.push(0);
    let head_index =
      Expr::List(head_pos.iter().map(|&i| Expr::Integer(i)).collect());
    let head_expr = apply_func_to_two_args(
      func,
      &Expr::Identifier(head_name.clone()),
      &head_index,
    )?;
    // If the rewritten head reduced to a bare Identifier, splice it as
    // the function head; otherwise wrap as `head_expr[mapped...]` so
    // the head expression is still visible in the result.
    return Ok(match &head_expr {
      Expr::Identifier(n) if n == "List" => Expr::List(mapped.into()),
      Expr::Identifier(n) => Expr::FunctionCall {
        name: n.clone(),
        args: mapped.into(),
      },
      _ => Expr::CurriedCall {
        func: Box::new(head_expr),
        args: mapped,
      },
    });
  }
  // Atomic leaf — apply `func` with the position index list.
  let index = Expr::List(position.iter().map(|&i| Expr::Integer(i)).collect());
  apply_func_to_two_args(func, expr, &index)
}

/// MapIndexed[f, expr, levelspec, Heads -> True].
pub fn map_indexed_with_level_heads_ast(
  func: &Expr,
  expr: &Expr,
  level_spec: &Expr,
  heads_opt: &Expr,
) -> Result<Expr, InterpreterError> {
  if !is_heads_true_option(heads_opt) {
    return Ok(Expr::FunctionCall {
      name: "MapIndexed".to_string(),
      args: vec![
        func.clone(),
        expr.clone(),
        level_spec.clone(),
        heads_opt.clone(),
      ]
      .into(),
    });
  }
  // `{-1}` selects every atomic leaf; with `Heads -> True` each compound
  // node's head is also rewritten with `0` appended to its position.
  if is_neg_one_levelspec(level_spec) {
    return map_indexed_atoms(func, expr, &[], true);
  }
  // Parse level spec the same way as the 3-arg form.
  let (min_level, max_level) = match level_spec {
    Expr::Integer(n) => (1i64, *n as i64),
    Expr::List(items) if items.len() == 1 => {
      if let Some(n) = expr_to_i128(&items[0]) {
        (n as i64, n as i64)
      } else {
        return Ok(Expr::FunctionCall {
          name: "MapIndexed".to_string(),
          args: vec![
            func.clone(),
            expr.clone(),
            level_spec.clone(),
            heads_opt.clone(),
          ]
          .into(),
        });
      }
    }
    Expr::List(items) if items.len() == 2 => {
      let min = expr_to_i128(&items[0]).unwrap_or(0) as i64;
      let max = expr_to_i128(&items[1]).unwrap_or(0) as i64;
      (min, max)
    }
    _ => {
      return Ok(Expr::FunctionCall {
        name: "MapIndexed".to_string(),
        args: vec![
          func.clone(),
          expr.clone(),
          level_spec.clone(),
          heads_opt.clone(),
        ]
        .into(),
      });
    }
  };
  map_indexed_at_depth_heads(func, expr, 0, min_level, max_level, &[])
}

/// Recursively apply MapIndexed at specified depth levels, tracking position indices.
fn map_indexed_at_depth(
  func: &Expr,
  expr: &Expr,
  current_depth: i64,
  min_level: i64,
  max_level: i64,
  position: &[i128],
) -> Result<Expr, InterpreterError> {
  let children = match expr {
    Expr::List(items) => Some((items.as_slice(), true)),
    Expr::FunctionCall { name: _, args } => Some((args.as_slice(), false)),
    _ => None,
  };

  if let Some((items, is_list)) = children {
    // Recurse into children
    let mapped: Result<Vec<Expr>, _> = items
      .iter()
      .enumerate()
      .map(|(i, item)| {
        let mut child_pos = position.to_vec();
        child_pos.push((i + 1) as i128);
        map_indexed_at_depth(
          func,
          item,
          current_depth + 1,
          min_level,
          max_level,
          &child_pos,
        )
      })
      .collect();
    let mapped = mapped?;
    let result = if is_list {
      Expr::List(mapped.into())
    } else if let Expr::FunctionCall { name, .. } = expr {
      Expr::FunctionCall {
        name: name.clone(),
        args: mapped.into(),
      }
    } else {
      unreachable!()
    };
    // Apply at this level if within range
    if current_depth >= min_level && current_depth <= max_level {
      let index =
        Expr::List(position.iter().map(|&i| Expr::Integer(i)).collect());
      apply_func_to_two_args(func, &result, &index)
    } else {
      Ok(result)
    }
  } else {
    // Atom — apply if at the right level
    if current_depth >= min_level && current_depth <= max_level {
      let index =
        Expr::List(position.iter().map(|&i| Expr::Integer(i)).collect());
      apply_func_to_two_args(func, expr, &index)
    } else {
      Ok(expr.clone())
    }
  }
}

/// MapIndexed variant that also applies f to heads, with index position 0 appended.
/// Each compound node first has its head rewritten as f[head, {position..., 0}]
/// (if within the level range), then its children are recursed with their own
/// position suffixes. The resulting head is the transformed head expression itself
/// (Mathematica style: f[List, {0}] becomes the head of the outer list).
fn map_indexed_at_depth_heads(
  func: &Expr,
  expr: &Expr,
  current_depth: i64,
  min_level: i64,
  max_level: i64,
  position: &[i128],
) -> Result<Expr, InterpreterError> {
  let (children, head_name_opt): (Option<&[Expr]>, Option<String>) = match expr
  {
    Expr::List(items) => (Some(items.as_slice()), Some("List".to_string())),
    Expr::FunctionCall { name, args } => {
      (Some(args.as_slice()), Some(name.clone()))
    }
    _ => (None, None),
  };

  if let (Some(items), Some(head_name)) = (children, head_name_opt) {
    // Recurse into children first (bottom-up).
    let mapped: Result<Vec<Expr>, _> = items
      .iter()
      .enumerate()
      .map(|(i, item)| {
        let mut child_pos = position.to_vec();
        child_pos.push((i + 1) as i128);
        map_indexed_at_depth_heads(
          func,
          item,
          current_depth + 1,
          min_level,
          max_level,
          &child_pos,
        )
      })
      .collect();
    let mapped = mapped?;

    // Transform the head at the current position with index 0 appended.
    // Heads count as level current_depth+1 (one step deeper than the node),
    // matching Mathematica's Heads->True convention.
    let head_expr = {
      let head_atom = Expr::Identifier(head_name.clone());
      let mut head_pos = position.to_vec();
      head_pos.push(0);
      let head_index =
        Expr::List(head_pos.iter().map(|&i| Expr::Integer(i)).collect());
      let head_depth = current_depth + 1;
      if head_depth >= min_level && head_depth <= max_level {
        apply_func_to_two_args(func, &head_atom, &head_index)?
      } else {
        head_atom
      }
    };

    // Re-wrap children using the transformed head as a head expression.
    // If the head is still the original symbol, use List / FunctionCall directly;
    // otherwise produce a CurriedCall so output renders as 'f[List,{0}][...]'.
    let inner_expr = match &head_expr {
      Expr::Identifier(n) if *n == "List" => Expr::List(mapped.into()),
      Expr::Identifier(n) => Expr::FunctionCall {
        name: n.clone(),
        args: mapped.into(),
      },
      _ => Expr::CurriedCall {
        func: Box::new(head_expr.clone()),
        args: mapped,
      },
    };

    // Apply f at this level to the wrapped expression if the level matches.
    if current_depth >= min_level && current_depth <= max_level {
      let index =
        Expr::List(position.iter().map(|&i| Expr::Integer(i)).collect());
      apply_func_to_two_args(func, &inner_expr, &index)
    } else {
      Ok(inner_expr)
    }
  } else {
    // Atom: apply f if current depth is in range; heads don't apply to atoms.
    if current_depth >= min_level && current_depth <= max_level {
      let index =
        Expr::List(position.iter().map(|&i| Expr::Integer(i)).collect());
      apply_func_to_two_args(func, expr, &index)
    } else {
      Ok(expr.clone())
    }
  }
}

/// AST-based MapThread: apply function to corresponding elements.
/// MapThread[f, {{a, b}, {c, d}}] -> {f[a, c], f[b, d]}
pub fn map_thread_ast(
  func: &Expr,
  lists: &Expr,
  level: Option<usize>,
) -> Result<Expr, InterpreterError> {
  let outer_items = match lists {
    Expr::List(items) => items,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "MapThread".to_string(),
        args: vec![func.clone(), lists.clone()].into(),
      });
    }
  };

  if outer_items.is_empty() {
    return Ok(Expr::List(vec![].into()));
  }

  // Get each sublist
  let mut sublists: Vec<Vec<Expr>> = Vec::new();
  for item in outer_items {
    match item {
      Expr::List(items) => sublists.push(items.to_vec()),
      _ => {
        return Err(InterpreterError::EvaluationError(
          "MapThread: second argument must be a list of lists".into(),
        ));
      }
    }
  }

  // Check all sublists have the same length
  let len = sublists[0].len();
  for sublist in &sublists {
    if sublist.len() != len {
      return Err(InterpreterError::EvaluationError(
        "MapThread: all lists must have the same length".into(),
      ));
    }
  }

  let depth = level.unwrap_or(1);

  if depth <= 1 {
    // Apply function to corresponding elements
    let mut results = Vec::new();
    for i in 0..len {
      let args: Vec<Expr> = sublists.iter().map(|sl| sl[i].clone()).collect();
      let result = apply_func_to_n_args(func, &args)?;
      if !is_nothing(&result) {
        results.push(result);
      }
    }
    Ok(Expr::List(results.into()))
  } else {
    // Recurse: thread at this level, then recurse into sublists
    let mut results = Vec::new();
    for i in 0..len {
      let inner_lists: Vec<Expr> =
        sublists.iter().map(|sl| sl[i].clone()).collect();
      let inner_arg = Expr::List(inner_lists.into());
      let result = map_thread_ast(func, &inner_arg, Some(depth - 1))?;
      results.push(result);
    }
    Ok(Expr::List(results.into()))
  }
}

/// AST-based Thread: thread a function over lists.
/// Thread[f[{a, b}, {c, d}]] -> {f[a, c], f[b, d]}
pub fn thread_ast(
  expr: &Expr,
  thread_head: Option<&str>,
) -> Result<Expr, InterpreterError> {
  thread_ast_positions(expr, thread_head, None)
}

/// Thread over the parts of `expr`. When `positions` is `Some`, only the
/// arguments at those 1-based positions are threaded over; every other
/// argument is held constant (the three-argument `Thread[expr, h, n]` form).
pub fn thread_ast_positions(
  expr: &Expr,
  thread_head: Option<&str>,
  positions: Option<&[usize]>,
) -> Result<Expr, InterpreterError> {
  match expr {
    Expr::FunctionCall { name, args } => {
      // Find which args contain the target head (List by default, or specified head)
      let mut list_indices: Vec<usize> = Vec::new();
      let mut list_len: Option<usize> = None;

      for (i, arg) in args.iter().enumerate() {
        // With an explicit position set, only thread over those positions.
        if let Some(pos) = positions
          && !pos.contains(&(i + 1))
        {
          continue;
        }
        let matching_items: Option<&crate::ExprList> = match thread_head {
          None => {
            // Default: thread over List
            if let Expr::List(items) = arg {
              Some(items)
            } else {
              None
            }
          }
          Some(head) => {
            // Thread over specified head
            if let Expr::FunctionCall {
              name: arg_name,
              args: arg_args,
            } = arg
            {
              if arg_name == head {
                Some(arg_args)
              } else {
                None
              }
            } else if head == "List" {
              if let Expr::List(items) = arg {
                Some(items)
              } else {
                None
              }
            } else {
              None
            }
          }
        };

        if let Some(items) = matching_items {
          if let Some(len) = list_len {
            if items.len() != len {
              return Err(InterpreterError::EvaluationError(
                "Thread: all lists must have the same length".into(),
              ));
            }
          } else {
            list_len = Some(items.len());
          }
          list_indices.push(i);
        }
      }

      if list_indices.is_empty() {
        return Ok(expr.clone());
      }

      let len = list_len.unwrap();
      let mut results = Vec::new();

      for j in 0..len {
        let new_args: Vec<Expr> = args
          .iter()
          .enumerate()
          .map(|(i, arg)| {
            if list_indices.contains(&i) {
              match thread_head {
                None => {
                  if let Expr::List(items) = arg {
                    items[j].clone()
                  } else {
                    arg.clone()
                  }
                }
                Some(_) => {
                  if let Expr::FunctionCall { args: arg_args, .. } = arg {
                    arg_args[j].clone()
                  } else if let Expr::List(items) = arg {
                    items[j].clone()
                  } else {
                    arg.clone()
                  }
                }
              }
            } else {
              arg.clone()
            }
          })
          .collect();
        let result =
          crate::evaluator::evaluate_function_call_ast(name, &new_args)?;
        results.push(result);
      }

      // Wrap result in the thread head
      match thread_head {
        None => Ok(Expr::List(results.into())),
        Some(head) => {
          if head == "List" {
            Ok(Expr::List(results.into()))
          } else {
            crate::evaluator::evaluate_function_call_ast(head, &results)
          }
        }
      }
    }
    Expr::Comparison {
      operands,
      operators,
    } => {
      // Thread over list operands in a comparison, e.g. Thread[{a,b} >= 0] -> {a>=0, b>=0}
      let mut list_index: Option<usize> = None;
      let mut list_len: Option<usize> = None;
      for (i, op) in operands.iter().enumerate() {
        if let Expr::List(items) = op {
          match list_len {
            None => {
              list_len = Some(items.len());
              list_index = Some(i);
            }
            Some(len) if len != items.len() => {
              return Err(InterpreterError::EvaluationError(
                "Thread: all lists must have the same length".into(),
              ));
            }
            _ => {}
          }
        }
      }
      match (list_index, list_len) {
        (Some(_), Some(len)) => {
          let results: Vec<Expr> = (0..len)
            .map(|j| {
              let new_operands: Vec<Expr> = operands
                .iter()
                .map(|op| {
                  if let Expr::List(items) = op {
                    items[j].clone()
                  } else {
                    op.clone()
                  }
                })
                .collect();
              let new_cmp = Expr::Comparison {
                operands: new_operands,
                operators: operators.clone(),
              };
              crate::evaluator::evaluate_expr_to_expr(&new_cmp)
            })
            .collect::<Result<_, _>>()?;
          Ok(Expr::List(results.into()))
        }
        _ => Ok(expr.clone()),
      }
    }
    Expr::Rule {
      pattern,
      replacement,
    }
    | Expr::RuleDelayed {
      pattern,
      replacement,
    } => {
      // Thread[{a,b} -> {c,d}] -> {a -> c, b -> d}
      let is_delayed = matches!(expr, Expr::RuleDelayed { .. });
      let lhs_items = match pattern.as_ref() {
        Expr::List(items) => Some(items),
        _ => None,
      };
      let rhs_items = match replacement.as_ref() {
        Expr::List(items) => Some(items),
        _ => None,
      };
      match (lhs_items, rhs_items) {
        (Some(lhs), Some(rhs)) => {
          if lhs.len() != rhs.len() {
            return Err(InterpreterError::EvaluationError(
              "Thread: all lists must have the same length".into(),
            ));
          }
          let results: Vec<Expr> = lhs
            .iter()
            .zip(rhs.iter())
            .map(|(l, r)| {
              if is_delayed {
                Expr::RuleDelayed {
                  pattern: Box::new(l.clone()),
                  replacement: Box::new(r.clone()),
                }
              } else {
                Expr::Rule {
                  pattern: Box::new(l.clone()),
                  replacement: Box::new(r.clone()),
                }
              }
            })
            .collect();
          Ok(Expr::List(results.into()))
        }
        (Some(lhs), None) => {
          // Thread[{a,b} -> c] -> {a -> c, b -> c}
          let results: Vec<Expr> = lhs
            .iter()
            .map(|l| {
              if is_delayed {
                Expr::RuleDelayed {
                  pattern: Box::new(l.clone()),
                  replacement: replacement.clone(),
                }
              } else {
                Expr::Rule {
                  pattern: Box::new(l.clone()),
                  replacement: replacement.clone(),
                }
              }
            })
            .collect();
          Ok(Expr::List(results.into()))
        }
        (None, Some(rhs)) => {
          // Thread[a -> {c,d}] -> {a -> c, a -> d}
          let results: Vec<Expr> = rhs
            .iter()
            .map(|r| {
              if is_delayed {
                Expr::RuleDelayed {
                  pattern: pattern.clone(),
                  replacement: Box::new(r.clone()),
                }
              } else {
                Expr::Rule {
                  pattern: pattern.clone(),
                  replacement: Box::new(r.clone()),
                }
              }
            })
            .collect();
          Ok(Expr::List(results.into()))
        }
        (None, None) => Ok(expr.clone()),
      }
    }
    _ => Ok(expr.clone()),
  }
}

/// AST-based Through: apply multiple functions.
/// Through[{f, g}[x]] -> {f[x], g[x]}
/// Through[f[g][x]] -> f[g[x]]
/// Through[Plus[f, g][x]] -> f[x] + g[x]
pub fn through_ast(
  expr: &Expr,
  head_filter: Option<&str>,
) -> Result<Expr, InterpreterError> {
  // Through operates on CurriedCall: h[f1, f2, ...][args...]
  // It threads the args through each fi, wrapping the result in h.
  match expr {
    Expr::CurriedCall { func, args } => {
      // func is the head expression, e.g. f[g], {f, g}, Plus[f, g]
      // args are the outer arguments to thread through
      let (head_name, functions) = match func.as_ref() {
        Expr::FunctionCall { name, args: fns } => {
          (name.as_str(), fns.as_slice())
        }
        Expr::List(items) => ("List", items.as_slice()),
        _ => {
          // Not a compound head - return unevaluated
          return Ok(Expr::FunctionCall {
            name: "Through".to_string(),
            args: vec![expr.clone()].into(),
          });
        }
      };

      // Check head filter if provided
      if let Some(filter) = head_filter
        && head_name != filter
      {
        // Head doesn't match filter - return the inner expression unchanged
        return Ok(expr.clone());
      }

      // Thread: apply each function to the outer args, then evaluate
      let threaded: Vec<Expr> = functions
        .iter()
        .map(|f| {
          let call = Expr::FunctionCall {
            name: crate::syntax::expr_to_string(f),
            args: args.clone().into(),
          };
          crate::evaluator::evaluate_expr_to_expr(&call).unwrap_or(call)
        })
        .collect();

      // Wrap in the head
      if head_name == "List" {
        Ok(Expr::List(threaded.into()))
      } else {
        Ok(Expr::FunctionCall {
          name: head_name.to_string(),
          args: threaded.into(),
        })
      }
    }
    _ => {
      if head_filter.is_some() {
        // With head filter and non-CurriedCall: return expression as-is
        Ok(expr.clone())
      } else {
        // Not a curried call - return unevaluated
        Ok(Expr::FunctionCall {
          name: "Through".to_string(),
          args: vec![expr.clone()].into(),
        })
      }
    }
  }
}

/// Comap[funs, x] applies each element of `funs` to `x`, preserving the
/// structure of `funs`. Equivalent to `Map[#[x]&, funs, lev]` with a default
/// level spec of `{1}`.
///
/// Examples:
///   Comap[{f, g, h}, x]            -> {f[x], g[x], h[x]}
///   Comap[<|a -> f, b -> g|>, x]   -> <|a -> f[x], b -> g[x]|>
///   Comap[h[f, g], x]             -> h[f[x], g[x]]
///   Comap[f, x]                   -> f   (atomic: unchanged)
pub fn comap_ast(
  funs: &Expr,
  arg: &Expr,
  level_spec: Option<&Expr>,
) -> Result<Expr, InterpreterError> {
  // Synthetic pure function `#1[arg]` applied to each element of `funs`.
  let applier = Expr::Function {
    body: Box::new(Expr::CurriedCall {
      func: Box::new(Expr::Slot(1)),
      args: vec![arg.clone()],
    }),
  };
  match level_spec {
    Some(lev) => map_with_level_ast(&applier, funs, lev),
    None => map_ast(&applier, funs),
  }
}

/// ComapApply[funs, args] applies each element of `funs` to the sequence of
/// `args` (like `Apply`), preserving the structure of `funs`. Equivalent to
/// `Map[Apply[#, args]&, funs]` at level `{1}`.
///
/// Examples:
///   ComapApply[{f, g}, {1, 2}]           -> {f[1, 2], g[1, 2]}
///   ComapApply[<|a -> f, b -> g|>, {1}]  -> <|a -> f[1], b -> g[1]|>
///   ComapApply[h[f, g], {1, 2}]          -> h[f[1, 2], g[1, 2]]
///   ComapApply[f, {1, 2}]               -> f   (atomic: unchanged)
pub fn comap_apply_ast(
  funs: &Expr,
  args: &Expr,
) -> Result<Expr, InterpreterError> {
  // Synthetic pure function `Apply[#1, args]` applied to each element of `funs`.
  let applier = Expr::Function {
    body: Box::new(Expr::FunctionCall {
      name: "Apply".to_string(),
      args: vec![Expr::Slot(1), args.clone()].into(),
    }),
  };
  map_ast(&applier, funs)
}

/// ComposeList[{f, g, h}, x] -> {x, f[x], g[f[x]], h[g[f[x]]]}
pub fn compose_list_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "ComposeList expects exactly 2 arguments".into(),
    ));
  }
  let funcs = match &args[0] {
    Expr::List(items) => items,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "ComposeList".to_string(),
        args: args.to_vec().into(),
      });
    }
  };

  let mut result = vec![args[1].clone()];
  let mut current = args[1].clone();
  for func in funcs {
    current = apply_func_ast(func, &current)?;
    result.push(current.clone());
  }
  Ok(Expr::List(result.into()))
}
