#[allow(unused_imports)]
use super::utilities::*;
#[allow(unused_imports)]
use super::*;

/// Shared AllTrue/AnyTrue/NoneTrue implementation: apply the predicate
/// to every element at exactly the requested level (default 1) and
/// combine via And/Or/Nor, mirroring wolframscript: non-boolean test
/// results stay symbolic (AllTrue[{a, b}, f] = f[a] && f[b]), atoms
/// have no level-1 elements and yield the vacuous result, general
/// heads and associations are traversed, and an invalid level emits
/// ::intnm.
fn quantifier_ast(
  fname: &str,
  combiner: &str,
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  let level: usize = if args.len() == 3 {
    match &args[2] {
      Expr::Integer(n) if (0..=i64::MAX as i128).contains(n) => *n as usize,
      _ => {
        let call = Expr::FunctionCall {
          name: fname.to_string(),
          args: args.to_vec().into(),
        };
        crate::emit_message(&format!(
          "{}::intnm: Non-negative machine-sized integer expected at position 3 in {}.",
          fname,
          crate::syntax::format_expr(&call, crate::syntax::ExprForm::Output)
        ));
        return Ok(call);
      }
    }
  } else {
    1
  };

  let mut elements = Vec::new();
  collect_at_level(&args[0], level, &mut elements);

  let results: Result<Vec<Expr>, InterpreterError> = elements
    .iter()
    .map(|e| apply_func_ast(&args[1], e))
    .collect();
  crate::evaluator::evaluate_function_call_ast(combiner, &results?)
}

/// AST-based AllTrue: And of the predicate over elements at a level.
pub fn all_true_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  quantifier_ast("AllTrue", "And", args)
}

/// AST-based AnyTrue: Or of the predicate over elements at a level.
pub fn any_true_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  quantifier_ast("AnyTrue", "Or", args)
}

/// AST-based NoneTrue: Nor of the predicate over elements at a level.
pub fn none_true_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  quantifier_ast("NoneTrue", "Nor", args)
}

/// Helper: collect all parts at a given level depth.
/// Level 0 = the expression itself, level 1 = direct sub-parts, etc.
fn collect_at_level(expr: &Expr, level: usize, result: &mut Vec<Expr>) {
  if level == 0 {
    result.push(expr.clone());
    return;
  }
  match expr {
    Expr::List(items) => {
      for item in items {
        collect_at_level(item, level - 1, result);
      }
    }
    Expr::FunctionCall { args, .. } => {
      for item in args {
        collect_at_level(item, level - 1, result);
      }
    }
    Expr::Association(pairs) => {
      // Association level 1 parts are the values
      for (_key, value) in pairs {
        collect_at_level(value, level - 1, result);
      }
    }
    _ => {}
  }
}

/// AllMatch[list, pattern] - True if all elements at level 1 match pattern
/// AllMatch[list, pattern, n] - True if all elements at level n match pattern
pub fn all_match_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() < 2 || args.len() > 3 {
    return Err(InterpreterError::EvaluationError(
      "AllMatch expects 2 or 3 arguments".into(),
    ));
  }
  let level = if args.len() == 3 {
    match &args[2] {
      Expr::Integer(n) if *n >= 0 => *n as usize,
      _ => {
        return Ok(Expr::FunctionCall {
          name: "AllMatch".to_string(),
          args: args.to_vec().into(),
        });
      }
    }
  } else {
    1
  };

  let mut parts = Vec::new();
  collect_at_level(&args[0], level, &mut parts);

  for part in &parts {
    if !matches_pattern_ast(part, &args[1]) {
      return Ok(bool_to_expr(false));
    }
  }
  Ok(bool_to_expr(true))
}

/// AnyMatch[list, pattern] - True if any element at level 1 matches pattern
/// AnyMatch[list, pattern, n] - True if any element at level n matches pattern
pub fn any_match_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() < 2 || args.len() > 3 {
    return Err(InterpreterError::EvaluationError(
      "AnyMatch expects 2 or 3 arguments".into(),
    ));
  }
  let level = if args.len() == 3 {
    match &args[2] {
      Expr::Integer(n) if *n >= 0 => *n as usize,
      _ => {
        return Ok(Expr::FunctionCall {
          name: "AnyMatch".to_string(),
          args: args.to_vec().into(),
        });
      }
    }
  } else {
    1
  };

  let mut parts = Vec::new();
  collect_at_level(&args[0], level, &mut parts);

  for part in &parts {
    if matches_pattern_ast(part, &args[1]) {
      return Ok(bool_to_expr(true));
    }
  }
  Ok(bool_to_expr(false))
}

/// AST-based GroupBy: group elements by the value of a function.
/// GroupBy[{a, b, c}, f] -> association of f[x] -> {elements with that f value}
pub fn group_by_ast(
  list: &Expr,
  func: &Expr,
) -> Result<Expr, InterpreterError> {
  // On an association, group its values by `f` (or `f -> g`). Each group is a
  // sub-association preserving the original keys, so e.g.
  // GroupBy[<|a -> 1, b -> 2, c -> 3|>, EvenQ] gives
  // <|False -> <|a -> 1, c -> 3|>, True -> <|b -> 2|>|>.
  if let Expr::Association(assoc_pairs) = list {
    if matches!(func, Expr::List(_)) {
      // The nested {f1, f2, …} form on associations is not supported.
      return Ok(Expr::FunctionCall {
        name: "GroupBy".to_string(),
        args: vec![list.clone(), func.clone()].into(),
      });
    }
    let (key_func, val_func): (&Expr, Option<&Expr>) = match func {
      Expr::Rule {
        pattern,
        replacement,
      }
      | Expr::RuleDelayed {
        pattern,
        replacement,
      } => (pattern.as_ref(), Some(replacement.as_ref())),
      _ => (func, None),
    };
    use std::collections::HashMap;
    let mut groups: HashMap<String, Vec<(Expr, Expr)>> = HashMap::new();
    let mut order: Vec<String> = Vec::new();
    for (k, v) in assoc_pairs {
      let key = apply_func_ast(key_func, v)?;
      let key_str = crate::syntax::expr_to_string(&key);
      let stored = match val_func {
        Some(g) => apply_func_ast(g, v)?,
        None => v.clone(),
      };
      if let Some(group) = groups.get_mut(&key_str) {
        group.push((k.clone(), stored));
      } else {
        order.push(key_str.clone());
        groups.insert(key_str, vec![(k.clone(), stored)]);
      }
    }
    let pairs: Vec<(Expr, Expr)> = order
      .into_iter()
      .map(|ks| {
        let entries = groups.remove(&ks).unwrap();
        let key_expr =
          crate::syntax::string_to_expr(&ks).unwrap_or(Expr::Raw(ks));
        (key_expr, Expr::Association(entries))
      })
      .collect();
    return Ok(Expr::Association(pairs));
  }

  let items = match list {
    Expr::List(items) => items,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "GroupBy".to_string(),
        args: vec![list.clone(), func.clone()].into(),
      });
    }
  };

  // `GroupBy[list, {f1, f2, ...}]` groups by `f1`, then sub-groups each result
  // by `f2`, and so on, producing nested associations.
  if let Expr::List(funcs) = func {
    return group_by_nested(items, funcs);
  }

  // `GroupBy[list, f -> g]` groups by `f` but stores `g[element]` in each
  // group instead of the element itself.
  let (key_func, val_func): (&Expr, Option<&Expr>) = match func {
    Expr::Rule {
      pattern,
      replacement,
    }
    | Expr::RuleDelayed {
      pattern,
      replacement,
    } => (pattern.as_ref(), Some(replacement.as_ref())),
    _ => (func, None),
  };

  use std::collections::HashMap;
  let mut groups: HashMap<String, Vec<Expr>> = HashMap::new();
  let mut order: Vec<String> = Vec::new();

  for item in items {
    let key = apply_func_ast(key_func, item)?;
    let key_str = crate::syntax::expr_to_string(&key);
    let value = match val_func {
      Some(g) => apply_func_ast(g, item)?,
      None => item.clone(),
    };
    if let Some(group) = groups.get_mut(&key_str) {
      group.push(value);
    } else {
      order.push(key_str.clone());
      groups.insert(key_str, vec![value]);
    }
  }

  // Build association preserving order
  let pairs: Vec<(Expr, Expr)> = order
    .into_iter()
    .map(|k| {
      let items = groups.remove(&k).unwrap();
      let key_expr = crate::syntax::string_to_expr(&k).unwrap_or(Expr::Raw(k));
      (key_expr, Expr::List(items.into()))
    })
    .collect();

  Ok(Expr::Association(pairs))
}

/// Nested GroupBy: group `items` by `funcs[0]`, then recursively group each
/// result by the remaining functions. With a single function this is the
/// ordinary flat grouping; an empty list leaves the items ungrouped.
fn group_by_nested(
  items: &[Expr],
  funcs: &[Expr],
) -> Result<Expr, InterpreterError> {
  if funcs.is_empty() {
    return Ok(Expr::List(items.to_vec().into()));
  }

  use std::collections::HashMap;
  let mut groups: HashMap<String, Vec<Expr>> = HashMap::new();
  let mut order: Vec<String> = Vec::new();
  for item in items {
    let key = apply_func_ast(&funcs[0], item)?;
    let key_str = crate::syntax::expr_to_string(&key);
    if let Some(group) = groups.get_mut(&key_str) {
      group.push(item.clone());
    } else {
      order.push(key_str.clone());
      groups.insert(key_str, vec![item.clone()]);
    }
  }

  let rest = &funcs[1..];
  let mut pairs: Vec<(Expr, Expr)> = Vec::with_capacity(order.len());
  for k in order {
    let group_items = groups.remove(&k).unwrap();
    let key_expr = crate::syntax::string_to_expr(&k).unwrap_or(Expr::Raw(k));
    let value = if rest.is_empty() {
      Expr::List(group_items.into())
    } else {
      group_by_nested(&group_items, rest)?
    };
    pairs.push((key_expr, value));
  }
  Ok(Expr::Association(pairs))
}

/// Closed-form Median for known distributions. Returns None for
/// distributions whose median Woxi cannot express symbolically.
fn distribution_median(name: &str, dargs: &[Expr]) -> Option<Expr> {
  match name {
    "PowerDistribution" if dargs.len() == 2 => {
      // Median[PowerDistribution[k, a]] = 1/(2^(1/a) k).
      let (k, a) = (dargs[0].clone(), dargs[1].clone());
      let med = Expr::BinaryOp {
        op: crate::syntax::BinaryOperator::Divide,
        left: Box::new(Expr::Integer(1)),
        right: Box::new(Expr::FunctionCall {
          name: "Times".to_string(),
          args: vec![
            Expr::BinaryOp {
              op: crate::syntax::BinaryOperator::Power,
              left: Box::new(Expr::Integer(2)),
              right: Box::new(Expr::BinaryOp {
                op: crate::syntax::BinaryOperator::Power,
                left: Box::new(a),
                right: Box::new(Expr::Integer(-1)),
              }),
            },
            k,
          ]
          .into(),
        }),
      };
      crate::evaluator::evaluate_expr_to_expr(&med).ok()
    }
    "LogisticDistribution" => match dargs.len() {
      0 => Some(Expr::Integer(0)),
      2 => Some(dargs[0].clone()),
      _ => None,
    },
    "HalfNormalDistribution" if dargs.len() == 1 => {
      let theta = dargs[0].clone();
      // Median = Sqrt[Pi] * InverseErf[1/2] / theta
      let sqrt_pi = Expr::FunctionCall {
        name: "Sqrt".to_string(),
        args: vec![Expr::Identifier("Pi".to_string())].into(),
      };
      let half = Expr::FunctionCall {
        name: "Rational".to_string(),
        args: vec![Expr::Integer(1), Expr::Integer(2)].into(),
      };
      let inverse_erf_half = Expr::FunctionCall {
        name: "InverseErf".to_string(),
        args: vec![half].into(),
      };
      let numer = Expr::FunctionCall {
        name: "Times".to_string(),
        args: vec![sqrt_pi, inverse_erf_half].into(),
      };
      let med = Expr::BinaryOp {
        op: crate::syntax::BinaryOperator::Divide,
        left: Box::new(numer),
        right: Box::new(theta),
      };
      crate::evaluator::evaluate_expr_to_expr(&med).ok()
    }
    "FrechetDistribution" if dargs.len() == 2 || dargs.len() == 3 => {
      let a = dargs[0].clone();
      let b = dargs[1].clone();
      let mu = if dargs.len() == 3 {
        Some(dargs[2].clone())
      } else {
        None
      };
      // Median = μ + b * Log[2]^(-1/a) ≡ μ + b / Log[2]^(1/a)
      let log2 = Expr::FunctionCall {
        name: "Log".to_string(),
        args: vec![Expr::Integer(2)].into(),
      };
      let inv_a = Expr::BinaryOp {
        op: crate::syntax::BinaryOperator::Divide,
        left: Box::new(Expr::Integer(1)),
        right: Box::new(a),
      };
      let denom = Expr::FunctionCall {
        name: "Power".to_string(),
        args: vec![log2, inv_a].into(),
      };
      let b_over = Expr::BinaryOp {
        op: crate::syntax::BinaryOperator::Divide,
        left: Box::new(b),
        right: Box::new(denom),
      };
      let med = match mu {
        Some(m) => Expr::BinaryOp {
          op: crate::syntax::BinaryOperator::Plus,
          left: Box::new(m),
          right: Box::new(b_over),
        },
        None => b_over,
      };
      crate::evaluator::evaluate_expr_to_expr(&med).ok()
    }
    "ExtremeValueDistribution" => {
      let (a, b) = match dargs.len() {
        0 => (Expr::Integer(0), Expr::Integer(1)),
        2 => (dargs[0].clone(), dargs[1].clone()),
        _ => return None,
      };
      // Median = a - b * Log[Log[2]]
      let log2 = Expr::FunctionCall {
        name: "Log".to_string(),
        args: vec![Expr::Integer(2)].into(),
      };
      let log_log2 = Expr::FunctionCall {
        name: "Log".to_string(),
        args: vec![log2].into(),
      };
      let med = Expr::BinaryOp {
        op: crate::syntax::BinaryOperator::Minus,
        left: Box::new(a),
        right: Box::new(Expr::BinaryOp {
          op: crate::syntax::BinaryOperator::Times,
          left: Box::new(b),
          right: Box::new(log_log2),
        }),
      };
      crate::evaluator::evaluate_expr_to_expr(&med).ok()
    }
    "LaplaceDistribution" if dargs.len() == 2 => {
      // Median[LaplaceDistribution[mu, beta]] = mu
      Some(dargs[0].clone())
    }
    "HypoexponentialDistribution" if dargs.len() == 1 => {
      hypoexponential_median(&dargs[0])
    }
    "NormalDistribution" if dargs.len() == 2 => {
      // Median[NormalDistribution[mu, sigma]] = mu (symmetric about mu)
      Some(dargs[0].clone())
    }
    "CauchyDistribution" if dargs.len() == 2 => {
      // Median[CauchyDistribution[a, b]] = a (symmetric about a)
      Some(dargs[0].clone())
    }
    "ChiDistribution" if dargs.len() == 1 => {
      // Median[ChiDistribution[v]] =
      //   Sqrt[2] * Sqrt[InverseGammaRegularized[v/2, 0, 1/2]].
      let v = dargs[0].clone();
      let half = Expr::FunctionCall {
        name: "Rational".to_string(),
        args: vec![Expr::Integer(1), Expr::Integer(2)].into(),
      };
      let v_half = Expr::BinaryOp {
        op: crate::syntax::BinaryOperator::Divide,
        left: Box::new(v),
        right: Box::new(Expr::Integer(2)),
      };
      let inv_gamma = Expr::FunctionCall {
        name: "InverseGammaRegularized".to_string(),
        args: vec![v_half, Expr::Integer(0), half].into(),
      };
      let sqrt_inner = Expr::FunctionCall {
        name: "Sqrt".to_string(),
        args: vec![inv_gamma].into(),
      };
      let sqrt2 = Expr::FunctionCall {
        name: "Sqrt".to_string(),
        args: vec![Expr::Integer(2)].into(),
      };
      let med = Expr::BinaryOp {
        op: crate::syntax::BinaryOperator::Times,
        left: Box::new(sqrt2),
        right: Box::new(sqrt_inner),
      };
      crate::evaluator::evaluate_expr_to_expr(&med).ok()
    }
    "BernoulliDistribution" if dargs.len() == 1 => {
      // Median[BernoulliDistribution[p]] = Piecewise[{{1, p > 1/2}}, 0].
      let p = dargs[0].clone();
      let half = Expr::FunctionCall {
        name: "Rational".to_string(),
        args: vec![Expr::Integer(1), Expr::Integer(2)].into(),
      };
      let cond = Expr::Comparison {
        operands: vec![p, half],
        operators: vec![crate::syntax::ComparisonOp::Greater],
      };
      let pair = Expr::List(vec![Expr::Integer(1), cond].into());
      let cases = Expr::List(vec![pair].into());
      let med = Expr::FunctionCall {
        name: "Piecewise".to_string(),
        args: vec![cases, Expr::Integer(0)].into(),
      };
      crate::evaluator::evaluate_expr_to_expr(&med).ok()
    }
    "DagumDistribution" if dargs.len() == 3 => {
      // Median[DagumDistribution[p, a, b]] = b / (-1 + 2^(1/p))^(1/a),
      // from inverting the CDF (1 + (b/x)^a)^(-p) = 1/2.
      let p = dargs[0].clone();
      let a = dargs[1].clone();
      let b = dargs[2].clone();
      let inv_p = Expr::BinaryOp {
        op: crate::syntax::BinaryOperator::Divide,
        left: Box::new(Expr::Integer(1)),
        right: Box::new(p),
      };
      let two_pow = Expr::BinaryOp {
        op: crate::syntax::BinaryOperator::Power,
        left: Box::new(Expr::Integer(2)),
        right: Box::new(inv_p),
      };
      let base = Expr::BinaryOp {
        op: crate::syntax::BinaryOperator::Plus,
        left: Box::new(Expr::Integer(-1)),
        right: Box::new(two_pow),
      };
      let inv_a = Expr::BinaryOp {
        op: crate::syntax::BinaryOperator::Divide,
        left: Box::new(Expr::Integer(1)),
        right: Box::new(a),
      };
      let denom = Expr::BinaryOp {
        op: crate::syntax::BinaryOperator::Power,
        left: Box::new(base),
        right: Box::new(inv_a),
      };
      let med = Expr::BinaryOp {
        op: crate::syntax::BinaryOperator::Divide,
        left: Box::new(b),
        right: Box::new(denom),
      };
      crate::evaluator::evaluate_expr_to_expr(&med).ok()
    }
    "BetaDistribution" if dargs.len() == 2 => {
      // Median[BetaDistribution[a, b]] = InverseBetaRegularized[1/2, a, b].
      let half = Expr::FunctionCall {
        name: "Rational".to_string(),
        args: vec![Expr::Integer(1), Expr::Integer(2)].into(),
      };
      Some(Expr::FunctionCall {
        name: "InverseBetaRegularized".to_string(),
        args: vec![half, dargs[0].clone(), dargs[1].clone()].into(),
      })
    }
    "ParetoDistribution" if dargs.len() == 2 => {
      // Median[ParetoDistribution[k, a]] = k * 2^(1/a)
      let k = dargs[0].clone();
      let a = dargs[1].clone();
      let inv_a = Expr::BinaryOp {
        op: crate::syntax::BinaryOperator::Divide,
        left: Box::new(Expr::Integer(1)),
        right: Box::new(a),
      };
      let pow_term = Expr::BinaryOp {
        op: crate::syntax::BinaryOperator::Power,
        left: Box::new(Expr::Integer(2)),
        right: Box::new(inv_a),
      };
      let med = Expr::BinaryOp {
        op: crate::syntax::BinaryOperator::Times,
        left: Box::new(k),
        right: Box::new(pow_term),
      };
      crate::evaluator::evaluate_expr_to_expr(&med).ok()
    }
    "WeibullDistribution" if dargs.len() == 2 => {
      let a = dargs[0].clone();
      let b = dargs[1].clone();
      // Median = b * Log[2]^(1/a)
      let log2 = Expr::FunctionCall {
        name: "Log".to_string(),
        args: vec![Expr::Integer(2)].into(),
      };
      let inv_a = Expr::BinaryOp {
        op: crate::syntax::BinaryOperator::Divide,
        left: Box::new(Expr::Integer(1)),
        right: Box::new(a),
      };
      let pow_term = Expr::BinaryOp {
        op: crate::syntax::BinaryOperator::Power,
        left: Box::new(log2),
        right: Box::new(inv_a),
      };
      let med = Expr::BinaryOp {
        op: crate::syntax::BinaryOperator::Times,
        left: Box::new(b),
        right: Box::new(pow_term),
      };
      crate::evaluator::evaluate_expr_to_expr(&med).ok()
    }
    "LogNormalDistribution" if dargs.len() == 2 => {
      // Median[LogNormalDistribution[mu, sigma]] = E^mu
      let med = Expr::BinaryOp {
        op: crate::syntax::BinaryOperator::Power,
        left: Box::new(Expr::Constant("E".to_string())),
        right: Box::new(dargs[0].clone()),
      };
      crate::evaluator::evaluate_expr_to_expr(&med).ok()
    }
    "ExponentialDistribution" if dargs.len() == 1 => {
      // Median[ExponentialDistribution[lambda]] = Log[2]/lambda. Reuses
      // the closed-form Quantile path which already returns -Log[1-p]/lambda.
      let half = Expr::FunctionCall {
        name: "Rational".to_string(),
        args: vec![Expr::Integer(1), Expr::Integer(2)].into(),
      };
      let quantile_call = Expr::FunctionCall {
        name: "Quantile".to_string(),
        args: vec![
          Expr::FunctionCall {
            name: "ExponentialDistribution".to_string(),
            args: dargs.to_vec().into(),
          },
          half,
        ]
        .into(),
      };
      crate::evaluator::evaluate_expr_to_expr(&quantile_call).ok()
    }
    "UniformDistribution" | "ArcSinDistribution" if dargs.len() == 1 => {
      let Expr::List(bounds) = &dargs[0] else {
        return None;
      };
      if bounds.len() != 2 {
        return None;
      }
      // Median = (a + b)/2. ArcSinDistribution is also symmetric about
      // the midpoint of its support, so the same formula applies.
      let sum = Expr::BinaryOp {
        op: crate::syntax::BinaryOperator::Plus,
        left: Box::new(bounds[0].clone()),
        right: Box::new(bounds[1].clone()),
      };
      let med = Expr::BinaryOp {
        op: crate::syntax::BinaryOperator::Divide,
        left: Box::new(sum),
        right: Box::new(Expr::Integer(2)),
      };
      crate::evaluator::evaluate_expr_to_expr(&med).ok()
    }
    "StudentTDistribution" if dargs.len() == 1 => {
      // Median[StudentTDistribution[v]] = 0 (symmetric about 0 for every v).
      Some(Expr::Integer(0))
    }
    "RayleighDistribution" if dargs.len() == 1 => {
      let sigma = dargs[0].clone();
      // Median = sigma * Sqrt[Log[4]]
      let log4 = Expr::FunctionCall {
        name: "Log".to_string(),
        args: vec![Expr::Integer(4)].into(),
      };
      let sqrt_log4 = Expr::FunctionCall {
        name: "Sqrt".to_string(),
        args: vec![log4].into(),
      };
      let med = Expr::BinaryOp {
        op: crate::syntax::BinaryOperator::Times,
        left: Box::new(sigma),
        right: Box::new(sqrt_log4),
      };
      crate::evaluator::evaluate_expr_to_expr(&med).ok()
    }
    "DiscreteUniformDistribution" if dargs.len() == 1 => {
      let Expr::List(bounds) = &dargs[0] else {
        return None;
      };
      if bounds.len() != 2 {
        return None;
      }
      let imin = bounds[0].clone();
      let imax = bounds[1].clone();
      // Median = -1 + min + Max[1, Ceiling[(1 + max - min)/2]]
      let diff = Expr::BinaryOp {
        op: crate::syntax::BinaryOperator::Minus,
        left: Box::new(imax),
        right: Box::new(imin.clone()),
      };
      let one_plus_diff = Expr::BinaryOp {
        op: crate::syntax::BinaryOperator::Plus,
        left: Box::new(Expr::Integer(1)),
        right: Box::new(diff),
      };
      let half = Expr::BinaryOp {
        op: crate::syntax::BinaryOperator::Divide,
        left: Box::new(one_plus_diff),
        right: Box::new(Expr::Integer(2)),
      };
      let ceiling = Expr::FunctionCall {
        name: "Ceiling".to_string(),
        args: vec![half].into(),
      };
      let max_call = Expr::FunctionCall {
        name: "Max".to_string(),
        args: vec![Expr::Integer(1), ceiling].into(),
      };
      let med = Expr::BinaryOp {
        op: crate::syntax::BinaryOperator::Plus,
        left: Box::new(Expr::Integer(-1)),
        right: Box::new(Expr::BinaryOp {
          op: crate::syntax::BinaryOperator::Plus,
          left: Box::new(imin),
          right: Box::new(max_call),
        }),
      };
      crate::evaluator::evaluate_expr_to_expr(&med).ok()
    }
    "InverseGammaDistribution" if dargs.len() == 2 => {
      let a = dargs[0].clone();
      let b = dargs[1].clone();
      // Median = b / InverseGammaRegularized[a, 1/2]
      let half = Expr::FunctionCall {
        name: "Rational".to_string(),
        args: vec![Expr::Integer(1), Expr::Integer(2)].into(),
      };
      let denom = Expr::FunctionCall {
        name: "InverseGammaRegularized".to_string(),
        args: vec![a, half].into(),
      };
      let med = Expr::BinaryOp {
        op: crate::syntax::BinaryOperator::Divide,
        left: Box::new(b),
        right: Box::new(denom),
      };
      crate::evaluator::evaluate_expr_to_expr(&med).ok()
    }
    "GompertzMakehamDistribution" if dargs.len() == 2 => {
      let lambda = dargs[0].clone();
      let xi = dargs[1].clone();
      // Median = Log[1 + Log[2]/xi] / lambda
      let log2 = Expr::FunctionCall {
        name: "Log".to_string(),
        args: vec![Expr::Integer(2)].into(),
      };
      let log2_over_xi = Expr::BinaryOp {
        op: crate::syntax::BinaryOperator::Divide,
        left: Box::new(log2),
        right: Box::new(xi),
      };
      let one_plus = Expr::BinaryOp {
        op: crate::syntax::BinaryOperator::Plus,
        left: Box::new(Expr::Integer(1)),
        right: Box::new(log2_over_xi),
      };
      let log_arg = Expr::FunctionCall {
        name: "Log".to_string(),
        args: vec![one_plus].into(),
      };
      let med = Expr::BinaryOp {
        op: crate::syntax::BinaryOperator::Divide,
        left: Box::new(log_arg),
        right: Box::new(lambda),
      };
      crate::evaluator::evaluate_expr_to_expr(&med).ok()
    }
    // Median[dist] = Quantile[dist, 1/2]; delegate to the closed-form quantile
    // for the gamma family. (BetaDistribution is handled by an earlier arm.)
    "GammaDistribution" | "ChiSquareDistribution" => {
      let half = crate::functions::math_ast::make_rational(1, 2);
      crate::functions::math_ast::quantile_distribution_closed_form(
        name, dargs, &half,
      )
    }
    _ => None,
  }
}

/// `Median[HypoexponentialDistribution[{lambda_1, ..., lambda_n}]]`.
///
/// The CDF F(x) of a hypoexponential with distinct positive rates is
///   F(x) = 1 - Σ_i c_i e^(-lambda_i x),  c_i = Π_{j ≠ i} lambda_j/(lambda_j - lambda_i).
/// The median is the solution of F(x) = 1/2. For a single rate the
/// distribution collapses to ExponentialDistribution[lambda] (median
/// Log[2]/lambda). For integer-rate cases we additionally check whether
/// x = Log[2] (i.e. u = 1/2) is a root — when so, F has the closed-form
/// median Log[2].
fn hypoexponential_median(arg: &Expr) -> Option<Expr> {
  let rates = if let Expr::List(items) = arg {
    items
  } else {
    return None;
  };
  if rates.is_empty() {
    return None;
  }
  // Single-rate case: HypoexponentialDistribution[{lambda}] = Exp(lambda).
  if rates.len() == 1 {
    let lambda = rates[0].clone();
    let log2 = Expr::FunctionCall {
      name: "Log".to_string(),
      args: vec![Expr::Integer(2)].into(),
    };
    let med = Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Divide,
      left: Box::new(log2),
      right: Box::new(lambda),
    };
    return crate::evaluator::evaluate_expr_to_expr(&med).ok();
  }

  // Multi-rate case: only handle distinct positive-integer rates and test
  // whether u = 1/2 is a root of the CDF polynomial Σ_i c_i u^lambda_i = 1/2.
  let mut int_rates: Vec<i128> = Vec::with_capacity(rates.len());
  for r in rates.iter() {
    let Expr::Integer(n) = r else { return None };
    if *n <= 0 {
      return None;
    }
    int_rates.push(*n);
  }
  // Distinct rates required for the simple partial-fraction formula.
  let mut sorted = int_rates.clone();
  sorted.sort();
  for w in sorted.windows(2) {
    if w[0] == w[1] {
      return None;
    }
  }

  // Compute Σ_i c_i / 2^lambda_i as an exact rational p/q (with q > 0).
  // We accumulate as (num, den) in lowest terms.
  let n = int_rates.len();
  let mut sum_num: i128 = 0;
  let mut sum_den: i128 = 1;
  for i in 0..n {
    let li = int_rates[i];
    // c_i = Π_{j ≠ i} lambda_j / (lambda_j - lambda_i).
    let mut ci_num: i128 = 1;
    let mut ci_den: i128 = 1;
    for j in 0..n {
      if j == i {
        continue;
      }
      let lj = int_rates[j];
      ci_num = ci_num.checked_mul(lj)?;
      let diff = lj - li;
      ci_den = ci_den.checked_mul(diff)?;
    }
    // term = c_i / 2^lambda_i = ci_num / (ci_den * 2^li)
    let pow_two = 1i128.checked_shl(li.try_into().ok()?)?;
    let term_den = ci_den.checked_mul(pow_two)?;
    let term_num = ci_num;
    // Add term_num/term_den to sum_num/sum_den.
    let new_num =
      sum_num.checked_mul(term_den)? + term_num.checked_mul(sum_den)?;
    let new_den = sum_den.checked_mul(term_den)?;
    let g = gcd_i128(new_num.abs(), new_den.abs()).max(1);
    sum_num = new_num / g;
    sum_den = new_den / g;
    if sum_den < 0 {
      sum_num = -sum_num;
      sum_den = -sum_den;
    }
  }

  // sum equals 1/2 iff 2 * sum_num == sum_den.
  if sum_num.checked_mul(2)? == sum_den {
    return Some(Expr::FunctionCall {
      name: "Log".to_string(),
      args: vec![Expr::Integer(2)].into(),
    });
  }
  None
}

fn gcd_i128(a: i128, b: i128) -> i128 {
  let (mut a, mut b) = (a.abs(), b.abs());
  while b != 0 {
    let t = a % b;
    a = b;
    b = t;
  }
  a
}

/// Numeric ordering key for a Median element. Handles plain numbers and
/// rationals directly, and falls back to numericizing symbolic-but-real values
/// (Pi, E, Sin[1], ...) via `N[expr]` so an exact list like {Pi, E, 1} can be
/// sorted. Returns `None` for non-numeric expressions.
fn median_sort_key(e: &Expr) -> Option<f64> {
  if let Some(n) = crate::functions::math_ast::expr_to_num(e) {
    return Some(n);
  }
  let n_call = Expr::FunctionCall {
    name: "N".to_string(),
    args: vec![e.clone()].into(),
  };
  match crate::evaluator::evaluate_expr_to_expr(&n_call).ok()? {
    Expr::Real(f) => Some(f),
    Expr::Integer(n) => Some(n as f64),
    other => crate::functions::math_ast::expr_to_num(&other),
  }
}

/// AST-based Median: calculate median of a list.
pub fn median_ast(list: &Expr) -> Result<Expr, InterpreterError> {
  // Distribution-form input: Median[dist] → quantile at p = 1/2.
  if let Expr::FunctionCall { name, args } = list
    && let Some(med) = distribution_median(name, args)
  {
    return Ok(med);
  }
  let items = match list {
    Expr::List(items) => items,
    _ => {
      crate::functions::math_ast::emit_rectt_if_numeric(
        "Median",
        std::slice::from_ref(list),
      );
      return Ok(Expr::FunctionCall {
        name: "Median".to_string(),
        args: vec![list.clone()].into(),
      });
    }
  };

  // Median requires a rectangular array of real numbers; a ragged/mixed array
  // or one with symbolic or complex entries emits Median::rectn. (Empty lists
  // pass through to the unevaluated handling below.)
  if !items.is_empty()
    && let Some(uneval) =
      crate::functions::math_ast::rectn_if_not_real_rectangular(
        "Median",
        std::slice::from_ref(list),
      )
  {
    return Ok(uneval);
  }

  if items.is_empty() {
    return Ok(Expr::FunctionCall {
      name: "Median".to_string(),
      args: vec![Expr::List(vec![].into())].into(),
    });
  }

  // Check for list-of-lists (matrix) input → columnwise median
  if items.iter().all(|item| matches!(item, Expr::List(_))) {
    let rows: Vec<&crate::ExprList> = items
      .iter()
      .filter_map(|item| {
        if let Expr::List(row) = item {
          Some(row)
        } else {
          None
        }
      })
      .collect();
    if !rows.is_empty() {
      let ncols = rows[0].len();
      if rows.iter().all(|r| r.len() == ncols) {
        let mut result = Vec::new();
        for col in 0..ncols {
          let column: Vec<Expr> = rows.iter().map(|r| r[col].clone()).collect();
          let col_median = median_ast(&Expr::List(column.into()))?;
          result.push(col_median);
        }
        return Ok(Expr::List(result.into()));
      }
    }
  }

  // Quantity lists sharing one unit: sort by magnitude, then take the middle
  // element (odd count) or the mean of the two middle ones (even count).
  // Matches wolframscript (e.g. Median[{Quantity[2,"m"], Quantity[4,"m"]}]).
  let is_quantity = |e: &Expr| {
    matches!(e, Expr::FunctionCall { name, args }
      if name == "Quantity" && args.len() == 2)
  };
  if items.iter().all(is_quantity) {
    let unit_key = |e: &Expr| -> Option<String> {
      if let Expr::FunctionCall { args, .. } = e {
        Some(crate::syntax::expr_to_string(&args[1]))
      } else {
        None
      }
    };
    let mag = |e: &Expr| -> Option<f64> {
      if let Expr::FunctionCall { args, .. } = e {
        expr_to_f64(&args[0])
      } else {
        None
      }
    };
    let u0 = unit_key(&items[0]);
    if u0.is_some()
      && items.iter().all(|i| unit_key(i) == u0)
      && items.iter().all(|i| mag(i).is_some())
    {
      let mut sorted: Vec<Expr> = items.iter().cloned().collect();
      sorted.sort_by(|a, b| {
        mag(a)
          .unwrap()
          .partial_cmp(&mag(b).unwrap())
          .unwrap_or(std::cmp::Ordering::Equal)
      });
      let len = sorted.len();
      if len % 2 == 1 {
        return Ok(sorted[len / 2].clone());
      }
      let avg = Expr::BinaryOp {
        op: crate::syntax::BinaryOperator::Divide,
        left: Box::new(Expr::FunctionCall {
          name: "Plus".to_string(),
          args: vec![sorted[len / 2 - 1].clone(), sorted[len / 2].clone()]
            .into(),
        }),
        right: Box::new(Expr::Integer(2)),
      };
      return crate::evaluator::evaluate_expr_to_expr(&avg);
    }
  }

  // Check if all items are integers
  let all_integers = items.iter().all(|i| matches!(i, Expr::Integer(_)));

  if all_integers {
    // Sort integer values
    let mut int_values: Vec<i128> = items
      .iter()
      .filter_map(|i| {
        if let Expr::Integer(n) = i {
          Some(*n)
        } else {
          None
        }
      })
      .collect();
    int_values.sort();

    let len = int_values.len();
    if len % 2 == 1 {
      Ok(Expr::Integer(int_values[len / 2]))
    } else {
      // Average of two middle values
      let a = int_values[len / 2 - 1];
      let b = int_values[len / 2];
      let sum = a + b;
      if sum % 2 == 0 {
        Ok(Expr::Integer(sum / 2))
      } else {
        // Return as Rational
        fn gcd(a: i128, b: i128) -> i128 {
          if b == 0 { a } else { gcd(b, a % b) }
        }
        let g = gcd(sum.abs(), 2);
        Ok(Expr::FunctionCall {
          name: "Rational".to_string(),
          args: vec![Expr::Integer(sum / g), Expr::Integer(2 / g)].into(),
        })
      }
    }
  } else {
    // Numeric sort key for every element. `median_sort_key` handles rationals
    // and symbolic-but-real values (Pi, E, Sin[1], ...) so we can order an
    // exact list like {Pi, E, 1} or {1/2, 1/3, 1/4}. If any element has no
    // numeric value the rectn check above should already have bailed, but stay
    // safe.
    let mut keyed: Vec<(f64, &Expr)> = Vec::with_capacity(items.len());
    for item in items {
      if let Some(n) = median_sort_key(item) {
        keyed.push((n, item));
      } else {
        return Ok(Expr::FunctionCall {
          name: "Median".to_string(),
          args: vec![list.clone()].into(),
        });
      }
    }
    keyed.sort_by(|a, b| {
      a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal)
    });

    let len = keyed.len();

    // Median returns the middle element verbatim (odd count) or the mean of the
    // two middle elements (even count), preserving each element's exact/inexact
    // nature: the selected element keeps its type, and the even-case mean is
    // re-evaluated so it simplifies (e.g. {1/2,1/3,1/4,1/5} -> 7/24,
    // {Pi,E,1,2} -> (2 + E)/2, {1.,3} -> 2.).
    if len % 2 == 1 {
      Ok(keyed[len / 2].1.clone())
    } else {
      let avg = Expr::BinaryOp {
        op: crate::syntax::BinaryOperator::Divide,
        left: Box::new(Expr::FunctionCall {
          name: "Plus".to_string(),
          args: vec![keyed[len / 2 - 1].1.clone(), keyed[len / 2].1.clone()]
            .into(),
        }),
        right: Box::new(Expr::Integer(2)),
      };
      crate::evaluator::evaluate_expr_to_expr(&avg)
    }
  }
}

/// Helper to convert an expression to f64, handling Rational.
fn take_expr_to_f64(expr: &Expr) -> Option<f64> {
  match expr {
    Expr::Integer(n) => Some(*n as f64),
    Expr::Real(f) => Some(*f),
    Expr::FunctionCall { name, args }
      if name == "Rational" && args.len() == 2 =>
    {
      if let (Expr::Integer(a), Expr::Integer(b)) = (&args[0], &args[1]) {
        Some(*a as f64 / *b as f64)
      } else {
        None
      }
    }
    _ => None,
  }
}

/// TakeLargest/TakeSmallest over an association: rank entries by their
/// (numeric) value and return an association of the `n` extreme key -> value
/// pairs, sorted by value (descending for largest, ascending for smallest;
/// stable, so ties keep their original order).
fn take_extreme_assoc(
  pairs: &[(Expr, Expr)],
  n: i128,
  largest: bool,
) -> Option<Result<Expr, InterpreterError>> {
  // Require every value to be numeric; otherwise let the caller leave the
  // call unevaluated.
  let mut keyed: Vec<((Expr, Expr), f64)> = Vec::with_capacity(pairs.len());
  for (k, v) in pairs {
    let val = take_expr_to_f64(v)?;
    keyed.push(((k.clone(), v.clone()), val));
  }
  if n < 0 || n as usize > keyed.len() {
    return None;
  }
  keyed.sort_by(|a, b| {
    let o = a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal);
    if largest { o.reverse() } else { o }
  });
  let result: Vec<(Expr, Expr)> = keyed
    .into_iter()
    .take(n as usize)
    .map(|(kv, _)| kv)
    .collect();
  Some(Ok(Expr::Association(result)))
}

/// AST-based TakeLargest: take n largest elements.
///
/// Non-numeric elements (like `Missing[...]`) are silently dropped,
/// matching Wolfram's default ExcludedForms -> {_Missing} behavior.
pub fn take_largest_ast(
  list: &Expr,
  n: i128,
) -> Result<Expr, InterpreterError> {
  if let Expr::Association(pairs) = list
    && let Some(result) = take_extreme_assoc(pairs, n, true)
  {
    return result;
  }
  let items = match list {
    Expr::List(items) => items,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "TakeLargest".to_string(),
        args: vec![list.clone(), Expr::Integer(n)].into(),
      });
    }
  };

  // Extract numeric values, skipping non-numeric entries.
  let mut keyed: Vec<(f64, Expr)> = Vec::new();
  for item in items {
    if let Some(v) = take_expr_to_f64(item) {
      keyed.push((v, item.clone()));
    }
  }

  // Only fail if we'd need more numeric values than the list contains.
  if n as usize > keyed.len() {
    return Ok(Expr::FunctionCall {
      name: "TakeLargest".to_string(),
      args: vec![list.clone(), Expr::Integer(n)].into(),
    });
  }

  keyed
    .sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

  let result: Vec<Expr> =
    keyed.into_iter().take(n as usize).map(|(_, e)| e).collect();

  Ok(Expr::List(result.into()))
}

/// AST-based TakeLargest with explicit ExcludedForms option.
///
/// Items matching any pattern in `excluded_forms` are filtered out
/// before sorting. Remaining items are sorted in canonical descending
/// order (the reverse of `Sort`), then the first `n` are returned.
pub fn take_largest_excluded_ast(
  list: &Expr,
  n: i128,
  excluded_forms: &[Expr],
) -> Result<Expr, InterpreterError> {
  let items = match list {
    Expr::List(items) => items,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "TakeLargest".to_string(),
        args: vec![list.clone(), Expr::Integer(n)].into(),
      });
    }
  };

  let filtered: Vec<Expr> = items
    .iter()
    .filter(|it| {
      !excluded_forms.iter().any(|p| {
        crate::evaluator::pattern_matching::match_pattern(it, p).is_some()
      })
    })
    .cloned()
    .collect();

  if n as usize > filtered.len() {
    return Ok(Expr::FunctionCall {
      name: "TakeLargest".to_string(),
      args: vec![list.clone(), Expr::Integer(n)].into(),
    });
  }

  let mut sorted = filtered;
  sorted.sort_by(|a, b| super::sorting::canonical_cmp(b, a));
  sorted.truncate(n as usize);
  Ok(Expr::List(sorted.into()))
}

/// AST-based TakeSmallest with explicit ExcludedForms option.
pub fn take_smallest_excluded_ast(
  list: &Expr,
  n: i128,
  excluded_forms: &[Expr],
) -> Result<Expr, InterpreterError> {
  let items = match list {
    Expr::List(items) => items,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "TakeSmallest".to_string(),
        args: vec![list.clone(), Expr::Integer(n)].into(),
      });
    }
  };

  let filtered: Vec<Expr> = items
    .iter()
    .filter(|it| {
      !excluded_forms.iter().any(|p| {
        crate::evaluator::pattern_matching::match_pattern(it, p).is_some()
      })
    })
    .cloned()
    .collect();

  if n as usize > filtered.len() {
    return Ok(Expr::FunctionCall {
      name: "TakeSmallest".to_string(),
      args: vec![list.clone(), Expr::Integer(n)].into(),
    });
  }

  let mut sorted = filtered;
  sorted.sort_by(super::sorting::canonical_cmp);
  sorted.truncate(n as usize);
  Ok(Expr::List(sorted.into()))
}

/// AST-based TakeSmallest: take n smallest elements.
///
/// Non-numeric elements (like `Missing[...]`) are silently dropped,
/// matching Wolfram's default ExcludedForms -> {_Missing} behavior.
pub fn take_smallest_ast(
  list: &Expr,
  n: i128,
) -> Result<Expr, InterpreterError> {
  if let Expr::Association(pairs) = list
    && let Some(result) = take_extreme_assoc(pairs, n, false)
  {
    return result;
  }
  let items = match list {
    Expr::List(items) => items,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "TakeSmallest".to_string(),
        args: vec![list.clone(), Expr::Integer(n)].into(),
      });
    }
  };

  // Extract numeric values, skipping non-numeric entries.
  let mut keyed: Vec<(f64, Expr)> = Vec::new();
  for item in items {
    if let Some(v) = take_expr_to_f64(item) {
      keyed.push((v, item.clone()));
    }
  }

  if n as usize > keyed.len() {
    return Ok(Expr::FunctionCall {
      name: "TakeSmallest".to_string(),
      args: vec![list.clone(), Expr::Integer(n)].into(),
    });
  }

  keyed
    .sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

  let result: Vec<Expr> =
    keyed.into_iter().take(n as usize).map(|(_, e)| e).collect();

  Ok(Expr::List(result.into()))
}

/// AST-based MinMax: return {min, max} of a list.
///
/// Supports the optional second argument that expands the returned
/// interval:
///   MinMax[list, d]             → {min - d,    max + d}
///   MinMax[list, {dMin, dMax}]  → {min - dMin, max + dMax}
///   MinMax[list, Scaled[d]]     → {min - d*r,  max + d*r}  with r = max - min
pub fn min_max_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() || args.len() > 2 {
    return Ok(Expr::FunctionCall {
      name: "MinMax".to_string(),
      args: args.to_vec().into(),
    });
  }
  // MinMax of an Interval is {Min[iv], Max[iv]}; reuse the list path so the
  // optional expansion second argument still applies.
  if let Expr::FunctionCall { name, .. } = &args[0]
    && name == "Interval"
  {
    let bound = |head: &str| {
      crate::evaluator::evaluate_function_call_ast(head, &[args[0].clone()])
    };
    // Only reduce when the bounds resolve to concrete numbers; symbolic
    // intervals stay unevaluated.
    if let (Ok(mn), Ok(mx)) = (bound("Min"), bound("Max"))
      && take_expr_to_f64(&mn).is_some()
      && take_expr_to_f64(&mx).is_some()
    {
      let mut new_args = args.to_vec();
      new_args[0] = Expr::List(vec![mn, mx].into());
      return min_max_ast(&new_args);
    }
    return Ok(Expr::FunctionCall {
      name: "MinMax".to_string(),
      args: args.to_vec().into(),
    });
  }
  let list = &args[0];
  let items = match list {
    Expr::List(items) => items,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "MinMax".to_string(),
        args: args.to_vec().into(),
      });
    }
  };

  if items.is_empty() {
    return Ok(Expr::List(
      vec![
        Expr::Identifier("Infinity".to_string()),
        Expr::UnaryOp {
          op: crate::syntax::UnaryOperator::Minus,
          operand: Box::new(Expr::Identifier("Infinity".to_string())),
        },
      ]
      .into(),
    ));
  }

  // Evaluate Min and Max through the normal evaluator so the result keeps
  // exact integer / rational arithmetic.
  let min_expr =
    crate::evaluator::evaluate_expr_to_expr(&Expr::FunctionCall {
      name: "Min".to_string(),
      args: items.clone(),
    })?;
  let max_expr =
    crate::evaluator::evaluate_expr_to_expr(&Expr::FunctionCall {
      name: "Max".to_string(),
      args: items.clone(),
    })?;

  // If no expansion argument was provided, return {min, max} directly.
  let Some(expansion) = args.get(1) else {
    return Ok(Expr::List(vec![min_expr, max_expr].into()));
  };

  // Extract (dMin, dMax) from the expansion argument.
  let (d_min, d_max): (Expr, Expr) = match expansion {
    Expr::List(pair) if pair.len() == 2 => (pair[0].clone(), pair[1].clone()),
    Expr::FunctionCall { name, args: a }
      if name == "Scaled" && a.len() == 1 =>
    {
      // Scale the expansion by the data range r = max - min.
      let range = Expr::FunctionCall {
        name: "Plus".to_string(),
        args: vec![
          max_expr.clone(),
          Expr::FunctionCall {
            name: "Times".to_string(),
            args: vec![Expr::Integer(-1), min_expr.clone()].into(),
          },
        ]
        .into(),
      };
      let scaled = Expr::FunctionCall {
        name: "Times".to_string(),
        args: vec![a[0].clone(), range].into(),
      };
      let scaled = crate::evaluator::evaluate_expr_to_expr(&scaled)?;
      (scaled.clone(), scaled)
    }
    d => (d.clone(), d.clone()),
  };

  let new_min = crate::evaluator::evaluate_expr_to_expr(&Expr::FunctionCall {
    name: "Plus".to_string(),
    args: vec![
      min_expr,
      Expr::FunctionCall {
        name: "Times".to_string(),
        args: vec![Expr::Integer(-1), d_min].into(),
      },
    ]
    .into(),
  })?;
  let new_max = crate::evaluator::evaluate_expr_to_expr(&Expr::FunctionCall {
    name: "Plus".to_string(),
    args: vec![max_expr, d_max].into(),
  })?;

  Ok(Expr::List(vec![new_min, new_max].into()))
}

/// Gather[list] - gathers elements into sublists of identical elements
/// Emit `<F>::list: List expected at position 1 in <call>.` and build
/// the unevaluated call (Gather/GatherBy/Tally require a List subject).
fn list_expected_message(fname: &str, args: Vec<Expr>) -> Expr {
  let call = Expr::FunctionCall {
    name: fname.to_string(),
    args: args.into(),
  };
  crate::emit_message(&format!(
    "{}::list: List expected at position 1 in {}.",
    fname,
    crate::syntax::format_expr(&call, crate::syntax::ExprForm::Output)
  ));
  call
}

pub fn gather_ast(list: &Expr) -> Result<Expr, InterpreterError> {
  let items = match list {
    Expr::List(items) => items,
    _ => {
      return Ok(list_expected_message("Gather", vec![list.clone()]));
    }
  };
  let mut groups: Vec<Vec<Expr>> = Vec::new();
  for item in items {
    let found = groups.iter_mut().find(|g| {
      crate::syntax::expr_to_string(&g[0])
        == crate::syntax::expr_to_string(item)
    });
    if let Some(group) = found {
      group.push(item.clone());
    } else {
      groups.push(vec![item.clone()]);
    }
  }
  Ok(Expr::List(
    groups.into_iter().map(|v| Expr::List(v.into())).collect(),
  ))
}

/// Gather[list, test] - gather into sublists using a custom equivalence test.
/// Each element joins the first existing group whose first (representative)
/// element satisfies `test[rep, elem]`, otherwise it starts a new group.
pub fn gather_with_test_ast(
  list: &Expr,
  test: &Expr,
) -> Result<Expr, InterpreterError> {
  let items = match list {
    Expr::List(items) => items,
    _ => {
      return Ok(list_expected_message(
        "Gather",
        vec![list.clone(), test.clone()],
      ));
    }
  };
  let mut groups: Vec<Vec<Expr>> = Vec::new();
  'outer: for item in items {
    for group in groups.iter_mut() {
      let result =
        super::utilities::apply_func_to_two_args(test, &group[0], item)?;
      if matches!(result, Expr::Identifier(ref s) if s == "True") {
        group.push(item.clone());
        continue 'outer;
      }
    }
    groups.push(vec![item.clone()]);
  }
  Ok(Expr::List(
    groups.into_iter().map(|v| Expr::List(v.into())).collect(),
  ))
}

/// GatherBy[list, f] - gathers elements into sublists by applying f.
/// GatherBy[list, {f1, f2, ...}] - nested gather: first by `f1`, then each
/// resulting group is recursively gathered by `f2`, and so on.
pub fn gather_by_ast(
  func: &Expr,
  list: &Expr,
) -> Result<Expr, InterpreterError> {
  // Handle the nested list-of-functions form.
  if let Expr::List(funcs) = func {
    let items = match list {
      Expr::List(items) => items.clone(),
      _ => {
        return Ok(list_expected_message(
          "GatherBy",
          vec![list.clone(), func.clone()],
        ));
      }
    };
    return gather_by_nested(&items, funcs);
  }
  let items = match list {
    Expr::List(items) => items,
    _ => {
      return Ok(list_expected_message(
        "GatherBy",
        vec![list.clone(), func.clone()],
      ));
    }
  };
  let mut groups: Vec<(String, Vec<Expr>)> = Vec::new();
  for item in items {
    let key = apply_func_ast(func, item)?;
    let key_str = crate::syntax::expr_to_string(&key);
    let found = groups.iter_mut().find(|(k, _)| *k == key_str);
    if let Some((_, group)) = found {
      group.push(item.clone());
    } else {
      groups.push((key_str, vec![item.clone()]));
    }
  }
  Ok(Expr::List(
    groups
      .into_iter()
      .map(|(_, v)| Expr::List(v.into()))
      .collect(),
  ))
}

fn gather_by_nested(
  items: &[Expr],
  funcs: &[Expr],
) -> Result<Expr, InterpreterError> {
  if funcs.is_empty() {
    // No further grouping: return the items as a flat list unchanged.
    return Ok(Expr::List(items.to_vec().into()));
  }
  let func = &funcs[0];
  let rest = &funcs[1..];
  let mut groups: Vec<(String, Vec<Expr>)> = Vec::new();
  for item in items {
    let key = apply_func_ast(func, item)?;
    let key_str = crate::syntax::expr_to_string(&key);
    let found = groups.iter_mut().find(|(k, _)| *k == key_str);
    if let Some((_, group)) = found {
      group.push(item.clone());
    } else {
      groups.push((key_str, vec![item.clone()]));
    }
  }
  let mut result: Vec<Expr> = Vec::with_capacity(groups.len());
  for (_, group) in groups {
    if rest.is_empty() {
      result.push(Expr::List(group.into()));
    } else {
      result.push(gather_by_nested(&group, rest)?);
    }
  }
  Ok(Expr::List(result.into()))
}

/// Split works on any nonatomic expression, keeping its head on the
/// result and every group; atoms emit `Split::normal` (with the given
/// extra display arguments) and return None.
fn split_subject<'a>(
  list: &'a Expr,
  display_args: &[Expr],
) -> Option<(&'a [Expr], Option<&'a str>)> {
  match list {
    Expr::List(items) => Some((items.as_slice(), None)),
    Expr::FunctionCall { name, args } => {
      Some((args.as_slice(), Some(name.as_str())))
    }
    _ => {
      let mut call_args = vec![list.clone()];
      call_args.extend(display_args.iter().cloned());
      crate::emit_message(&format!(
        "Split::normal: Nonatomic expression expected at position 1 in {}.",
        crate::syntax::format_expr(
          &Expr::FunctionCall {
            name: "Split".to_string(),
            args: call_args.into(),
          },
          crate::syntax::ExprForm::Output
        )
      ));
      None
    }
  }
}

fn wrap_groups(groups: Vec<Vec<Expr>>, head: Option<&str>) -> Expr {
  let wrap = |v: Vec<Expr>| -> Expr {
    match head {
      Some(h) => Expr::FunctionCall {
        name: h.to_string(),
        args: v.into(),
      },
      None => Expr::List(v.into()),
    }
  };
  wrap(groups.into_iter().map(wrap).collect())
}

/// Split[list] - splits into sublists of identical consecutive elements
pub fn split_ast(list: &Expr) -> Result<Expr, InterpreterError> {
  let Some((items, head)) = split_subject(list, &[]) else {
    return Ok(Expr::FunctionCall {
      name: "Split".to_string(),
      args: vec![list.clone()].into(),
    });
  };
  if items.is_empty() {
    return Ok(wrap_groups(Vec::new(), head));
  }
  let mut groups: Vec<Vec<Expr>> = vec![vec![items[0].clone()]];
  for item in items.iter().skip(1) {
    let last_group = groups.last().unwrap();
    if crate::syntax::expr_to_string(&last_group[0])
      == crate::syntax::expr_to_string(item)
    {
      groups.last_mut().unwrap().push(item.clone());
    } else {
      groups.push(vec![item.clone()]);
    }
  }
  Ok(wrap_groups(groups, head))
}

/// Split[list, test] - splits list where consecutive elements satisfy test function
pub fn split_with_test_ast(
  list: &Expr,
  test: &Expr,
) -> Result<Expr, InterpreterError> {
  let Some((items, head)) = split_subject(list, &[test.clone()]) else {
    return Ok(Expr::FunctionCall {
      name: "Split".to_string(),
      args: vec![list.clone(), test.clone()].into(),
    });
  };
  if items.is_empty() {
    return Ok(wrap_groups(Vec::new(), head));
  }
  let mut groups: Vec<Vec<Expr>> = vec![vec![items[0].clone()]];
  for item in items.iter().skip(1) {
    let last_item = groups.last().unwrap().last().unwrap();
    let test_result = apply_func_to_two_args(test, last_item, item)?;
    let passes = matches!(
      &test_result,
      Expr::Identifier(name) if name == "True"
    );
    if passes {
      groups.last_mut().unwrap().push(item.clone());
    } else {
      groups.push(vec![item.clone()]);
    }
  }
  Ok(wrap_groups(groups, head))
}

/// SplitBy[list, f] - splits into sublists of consecutive elements with same f value
/// SplitBy[list, {f1, f2, ...}] - nested split: first by f1, then by f2 within
/// each resulting group, and so on. Equivalent to
/// `Map[SplitBy[#, f2] &, SplitBy[list, f1]]` with a final `Fold`.
pub fn split_by_ast(
  func: &Expr,
  list: &Expr,
) -> Result<Expr, InterpreterError> {
  // If `func` is a list of functions, apply them hierarchically.
  if let Expr::List(funcs) = func {
    // The base case `SplitBy[list, {}]` is a no-op wrap — Mathematica
    // returns the list unchanged at that level.
    if funcs.is_empty() {
      return Ok(list.clone());
    }
    // Apply the first function to get a top-level partition, then recurse
    // on each group with the rest of the functions.
    let first = &funcs[0];
    let rest_funcs = Expr::List(funcs[1..].to_vec().into());
    let grouped = split_by_ast(first, list)?;
    let groups: Vec<Expr> = match &grouped {
      Expr::List(items) => items.to_vec(),
      _ => return Ok(grouped),
    };
    let mut result = Vec::with_capacity(groups.len());
    for g in groups {
      if funcs.len() == 1 {
        result.push(g);
      } else {
        result.push(split_by_ast(&rest_funcs, &g)?);
      }
    }
    return Ok(Expr::List(result.into()));
  }

  let items = match list {
    Expr::List(items) => items,
    _ => {
      // SplitBy delegates to Split, so wolframscript's message shows
      // the desugared test: Split[x, g[#1] === g[#2] & ].
      let show = |e: &Expr| {
        crate::syntax::format_expr(e, crate::syntax::ExprForm::Output)
      };
      crate::emit_message(&format!(
        "Split::normal: Nonatomic expression expected at position 1 in Split[{}, {}[#1] === {}[#2] & ].",
        show(list),
        show(func),
        show(func)
      ));
      return Ok(Expr::FunctionCall {
        name: "SplitBy".to_string(),
        args: vec![list.clone(), func.clone()].into(),
      });
    }
  };
  if items.is_empty() {
    return Ok(Expr::List(vec![].into()));
  }
  let mut prev_key = apply_func_ast(func, &items[0])?;
  let mut groups: Vec<Vec<Expr>> = vec![vec![items[0].clone()]];
  for item in items.iter().skip(1) {
    let key = apply_func_ast(func, item)?;
    if crate::syntax::expr_to_string(&key)
      == crate::syntax::expr_to_string(&prev_key)
    {
      groups.last_mut().unwrap().push(item.clone());
    } else {
      groups.push(vec![item.clone()]);
      prev_key = key;
    }
  }
  Ok(Expr::List(
    groups.into_iter().map(|v| Expr::List(v.into())).collect(),
  ))
}

/// Convert a bin-edge expression to `f64`, accepting `Infinity` / `-Infinity`
/// in whichever representation the evaluator preserves them.
fn edge_to_f64(e: &Expr) -> Option<f64> {
  if let Some(v) = numeric_expr_to_f64(e) {
    return Some(v);
  }
  // Bare Infinity symbol → +∞.
  if matches!(e, Expr::Identifier(s) if s == "Infinity") {
    return Some(f64::INFINITY);
  }
  // DirectedInfinity[1] → +∞, DirectedInfinity[-1] → -∞.
  if let Expr::FunctionCall { name, args } = e
    && name == "DirectedInfinity"
    && args.len() == 1
  {
    return match &args[0] {
      Expr::Integer(1) => Some(f64::INFINITY),
      Expr::Integer(-1) => Some(f64::NEG_INFINITY),
      _ => None,
    };
  }
  // Times[-1, +∞] → -∞.
  if let Expr::FunctionCall { name, args } = e
    && name == "Times"
    && args.len() == 2
    && matches!(&args[0], Expr::Integer(-1))
  {
    let inner = edge_to_f64(&args[1])?;
    return Some(-inner);
  }
  // -Infinity is stored as UnaryOp[Minus, Infinity] inside an unevaluated
  // List (the evaluator leaves it as-is to preserve the surface syntax).
  if let Expr::UnaryOp {
    op: crate::syntax::UnaryOperator::Minus,
    operand,
  } = e
  {
    let inner = edge_to_f64(operand)?;
    return Some(-inner);
  }
  None
}

/// BinCounts[data, {min, max, dx}] - count data points in equal-width bins
/// Bins are [min, min+dx), [min+dx, min+2dx), ..., [max-dx, max)
pub fn bin_counts_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let data = match &args[0] {
    Expr::List(items) => items,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "BinCounts".to_string(),
        args: args.to_vec().into(),
      });
    }
  };

  // Extract numeric values from data, skip non-numeric
  let values: Vec<f64> = data.iter().filter_map(numeric_expr_to_f64).collect();

  let (min_val, max_val, dx) = if args.len() == 1 {
    // BinCounts[data] - default dx=1, aligned to integer boundaries
    if values.is_empty() {
      return Ok(Expr::List(vec![].into()));
    }
    let data_min = values.iter().cloned().fold(f64::INFINITY, f64::min);
    let data_max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let dx = 1.0;
    let mut lo = (data_min / dx).floor() * dx;
    if (data_min - lo).abs() < 1e-12 {
      lo -= dx;
    }
    let mut hi = (data_max / dx).ceil() * dx;
    if (data_max - hi).abs() < 1e-12 {
      hi += dx;
    }
    (lo, hi, dx)
  } else if args.len() == 2 {
    match &args[1] {
      // BinCounts[data, dx]
      Expr::Integer(dx_int) => {
        if values.is_empty() {
          return Ok(Expr::List(vec![].into()));
        }
        let data_min = values.iter().cloned().fold(f64::INFINITY, f64::min);
        let data_max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let dx = *dx_int as f64;
        let mut lo = (data_min / dx).floor() * dx;
        if (data_min - lo).abs() < 1e-12 {
          lo -= dx;
        }
        let mut hi = (data_max / dx).ceil() * dx;
        if (data_max - hi).abs() < 1e-12 {
          hi += dx;
        }
        (lo, hi, dx)
      }
      Expr::Real(dx_f) => {
        if values.is_empty() {
          return Ok(Expr::List(vec![].into()));
        }
        let data_min = values.iter().cloned().fold(f64::INFINITY, f64::min);
        let data_max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let dx = *dx_f;
        let mut lo = (data_min / dx).floor() * dx;
        if (data_min - lo).abs() < 1e-12 {
          lo -= dx;
        }
        let mut hi = (data_max / dx).ceil() * dx;
        if (data_max - hi).abs() < 1e-12 {
          hi += dx;
        }
        (lo, hi, dx)
      }
      // BinCounts[data, {{e1, e2, ..., en}}] — explicit bin-edge form.
      // Bins are [e_i, e_{i+1}); the result has length n-1.
      Expr::List(outer)
        if outer.len() == 1
          && matches!(&outer[0], Expr::List(edges) if edges.len() >= 2) =>
      {
        let Expr::List(edges) = &outer[0] else {
          unreachable!()
        };
        let mut numeric_edges: Vec<f64> = Vec::with_capacity(edges.len());
        for e in edges.iter() {
          let v = if let Some(v) = edge_to_f64(e) {
            v
          } else {
            return Ok(Expr::FunctionCall {
              name: "BinCounts".to_string(),
              args: args.to_vec().into(),
            });
          };
          numeric_edges.push(v);
        }
        let mut counts = vec![0i128; numeric_edges.len() - 1];
        for &v in &values {
          for i in 0..(numeric_edges.len() - 1) {
            if numeric_edges[i] <= v && v < numeric_edges[i + 1] {
              counts[i] += 1;
              break;
            }
          }
        }
        let counts_exprs: Vec<Expr> =
          counts.into_iter().map(Expr::Integer).collect();
        return Ok(Expr::List(counts_exprs.into()));
      }
      // BinCounts[data, {min, max, dx}]
      Expr::List(spec) if spec.len() == 3 => {
        let min_v = match numeric_expr_to_f64(&spec[0]) {
          Some(v) => v,
          None => {
            return Ok(Expr::FunctionCall {
              name: "BinCounts".to_string(),
              args: args.to_vec().into(),
            });
          }
        };
        let max_v = match numeric_expr_to_f64(&spec[1]) {
          Some(v) => v,
          None => {
            return Ok(Expr::FunctionCall {
              name: "BinCounts".to_string(),
              args: args.to_vec().into(),
            });
          }
        };
        let dx = match numeric_expr_to_f64(&spec[2]) {
          Some(v) => v,
          None => {
            return Ok(Expr::FunctionCall {
              name: "BinCounts".to_string(),
              args: args.to_vec().into(),
            });
          }
        };
        (min_v, max_v, dx)
      }
      _ => {
        return Ok(Expr::FunctionCall {
          name: "BinCounts".to_string(),
          args: args.to_vec().into(),
        });
      }
    }
  } else {
    return Ok(Expr::FunctionCall {
      name: "BinCounts".to_string(),
      args: args.to_vec().into(),
    });
  };

  if dx <= 0.0 {
    return Err(InterpreterError::EvaluationError(
      "BinCounts: bin width must be positive".into(),
    ));
  }

  // Only whole bins [min + i*dx, min + (i+1)*dx) that fit within [min, max]
  // are produced; a trailing partial range (e.g. {0, 10, 3} leaves [9, 10])
  // is dropped, and out-of-range values are not capped into the last bin.
  let num_bins = ((max_val - min_val) / dx + 1e-9).floor().max(0.0) as usize;
  let mut counts = vec![0i128; num_bins];

  for &v in &values {
    if v < min_val {
      continue;
    }
    let idx = ((v - min_val) / dx).floor();
    if idx >= 0.0 && (idx as usize) < num_bins {
      counts[idx as usize] += 1;
    }
  }

  Ok(Expr::List(counts.into_iter().map(Expr::Integer).collect()))
}

/// BinLists[data, {min, max, dx}] - group data points into equal-width bins
/// Returns lists of elements per bin instead of counts
pub fn bin_lists_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let data = match &args[0] {
    Expr::List(items) => items,
    _ => {
      crate::emit_message(
        "BinLists::vectmat: The first argument is expected to be a unit-compatible vector or a matrix with unit-compatible columns.",
      );
      return Ok(Expr::FunctionCall {
        name: "BinLists".to_string(),
        args: args.to_vec().into(),
      });
    }
  };

  // Extract numeric values paired with original expressions
  let values: Vec<(f64, &Expr)> = data
    .iter()
    .filter_map(|e| numeric_expr_to_f64(e).map(|v| (v, e)))
    .collect();

  let (min_val, max_val, dx) = if args.len() == 1 {
    if values.is_empty() {
      return Ok(Expr::List(vec![].into()));
    }
    let data_min = values.iter().map(|(v, _)| *v).fold(f64::INFINITY, f64::min);
    let data_max = values
      .iter()
      .map(|(v, _)| *v)
      .fold(f64::NEG_INFINITY, f64::max);
    let dx = 1.0;
    let mut lo = (data_min / dx).floor() * dx;
    if (data_min - lo).abs() < 1e-12 {
      lo -= dx;
    }
    let mut hi = (data_max / dx).ceil() * dx;
    if (data_max - hi).abs() < 1e-12 {
      hi += dx;
    }
    (lo, hi, dx)
  } else if args.len() == 2 {
    match &args[1] {
      Expr::Integer(dx_int) => {
        if values.is_empty() {
          return Ok(Expr::List(vec![].into()));
        }
        let data_min =
          values.iter().map(|(v, _)| *v).fold(f64::INFINITY, f64::min);
        let data_max = values
          .iter()
          .map(|(v, _)| *v)
          .fold(f64::NEG_INFINITY, f64::max);
        let dx = *dx_int as f64;
        let mut lo = (data_min / dx).floor() * dx;
        if (data_min - lo).abs() < 1e-12 {
          lo -= dx;
        }
        let mut hi = (data_max / dx).ceil() * dx;
        if (data_max - hi).abs() < 1e-12 {
          hi += dx;
        }
        (lo, hi, dx)
      }
      Expr::Real(dx_f) => {
        if values.is_empty() {
          return Ok(Expr::List(vec![].into()));
        }
        let data_min =
          values.iter().map(|(v, _)| *v).fold(f64::INFINITY, f64::min);
        let data_max = values
          .iter()
          .map(|(v, _)| *v)
          .fold(f64::NEG_INFINITY, f64::max);
        let dx = *dx_f;
        let mut lo = (data_min / dx).floor() * dx;
        if (data_min - lo).abs() < 1e-12 {
          lo -= dx;
        }
        let mut hi = (data_max / dx).ceil() * dx;
        if (data_max - hi).abs() < 1e-12 {
          hi += dx;
        }
        (lo, hi, dx)
      }
      // BinLists[data, {{e1, e2, ..., en}}] — explicit bin-edge form.
      Expr::List(outer)
        if outer.len() == 1
          && matches!(&outer[0], Expr::List(edges) if edges.len() >= 2) =>
      {
        let Expr::List(edges) = &outer[0] else {
          unreachable!()
        };
        let mut numeric_edges: Vec<f64> = Vec::with_capacity(edges.len());
        for e in edges.iter() {
          let v = if let Some(v) = edge_to_f64(e) {
            v
          } else {
            return Ok(Expr::FunctionCall {
              name: "BinLists".to_string(),
              args: args.to_vec().into(),
            });
          };
          numeric_edges.push(v);
        }
        let mut bins: Vec<Vec<Expr>> =
          (0..numeric_edges.len() - 1).map(|_| Vec::new()).collect();
        for (v, expr) in &values {
          for i in 0..(numeric_edges.len() - 1) {
            if numeric_edges[i] <= *v && *v < numeric_edges[i + 1] {
              bins[i].push((*expr).clone());
              break;
            }
          }
        }
        let result_exprs: Vec<Expr> =
          bins.into_iter().map(|b| Expr::List(b.into())).collect();
        return Ok(Expr::List(result_exprs.into()));
      }
      Expr::List(spec) if spec.len() == 3 => {
        let min_v = match numeric_expr_to_f64(&spec[0]) {
          Some(v) => v,
          None => {
            return Ok(Expr::FunctionCall {
              name: "BinLists".to_string(),
              args: args.to_vec().into(),
            });
          }
        };
        let max_v = match numeric_expr_to_f64(&spec[1]) {
          Some(v) => v,
          None => {
            return Ok(Expr::FunctionCall {
              name: "BinLists".to_string(),
              args: args.to_vec().into(),
            });
          }
        };
        let dx = match numeric_expr_to_f64(&spec[2]) {
          Some(v) => v,
          None => {
            return Ok(Expr::FunctionCall {
              name: "BinLists".to_string(),
              args: args.to_vec().into(),
            });
          }
        };
        (min_v, max_v, dx)
      }
      _ => {
        return Ok(Expr::FunctionCall {
          name: "BinLists".to_string(),
          args: args.to_vec().into(),
        });
      }
    }
  } else {
    return Ok(Expr::FunctionCall {
      name: "BinLists".to_string(),
      args: args.to_vec().into(),
    });
  };

  if dx <= 0.0 {
    return Err(InterpreterError::EvaluationError(
      "BinLists: bin width must be positive".into(),
    ));
  }

  // Only whole bins [min + i*dx, min + (i+1)*dx) that fit within [min, max]
  // are produced — a trailing partial range (e.g. {0, 10, 3} leaves [9, 10])
  // is dropped. Values outside any bin are not capped into the last one.
  let num_bins = ((max_val - min_val) / dx + 1e-9).floor().max(0.0) as usize;
  let mut bins: Vec<Vec<Expr>> = vec![Vec::new(); num_bins];

  for &(v, expr) in &values {
    if v < min_val {
      continue;
    }
    let idx = ((v - min_val) / dx).floor();
    if idx >= 0.0 && (idx as usize) < num_bins {
      bins[idx as usize].push(expr.clone());
    }
  }

  Ok(Expr::List(
    bins.into_iter().map(|v| Expr::List(v.into())).collect(),
  ))
}

/// Round a positive value to the nearest "nice" number (1, 2, 5, 10, 20, 50, …)
/// using log-scale distance to decide which is closest.
fn nice_number(x: f64) -> f64 {
  if x <= 0.0 {
    return 1.0;
  }
  let exp = x.log10().floor() as i32;
  let base = 10f64.powi(exp);
  let candidates = [1.0 * base, 2.0 * base, 5.0 * base, 10.0 * base];
  let log_x = x.ln();
  candidates
    .iter()
    .copied()
    .min_by(|a, b| {
      (a.ln() - log_x)
        .abs()
        .partial_cmp(&(b.ln() - log_x).abs())
        .unwrap()
    })
    .unwrap()
}

/// Compute the interquartile range (Q3 - Q1) of a sorted slice.
fn interquartile_range(sorted: &[f64]) -> f64 {
  let n = sorted.len();
  if n < 2 {
    return 0.0;
  }
  let q1 = wolfram_quantile(sorted, 0.25);
  let q3 = wolfram_quantile(sorted, 0.75);
  q3 - q1
}

/// Wolfram's default Quantile: h = n*p, j = floor(h), g = h - j.
/// If g > 0: x_{j+1}, else x_j (1-based indexing).
fn wolfram_quantile(sorted: &[f64], p: f64) -> f64 {
  let n = sorted.len();
  if n == 0 {
    return 0.0;
  }
  let h = n as f64 * p;
  let j = h.floor() as usize;
  let g = h - j as f64;
  if g > 1e-12 {
    // x_{j+1}, 0-based: sorted[j]
    sorted[j.min(n - 1)]
  } else {
    // x_j, 0-based: sorted[j-1], but j could be 0
    if j == 0 {
      sorted[0]
    } else {
      sorted[(j - 1).min(n - 1)]
    }
  }
}

/// True when every value is an integer multiple of `dx` (within tolerance).
/// In that case wolframscript centers the histogram bins on the values (edges
/// offset by dx/2) so no value lands on a bin boundary.
fn all_multiples_of(values: &[f64], dx: f64) -> bool {
  dx > 0.0
    && values.iter().all(|&v| {
      let r = v / dx;
      (r - r.round()).abs() < 1e-9 * (1.0 + r.abs())
    })
}

/// Round the Freedman-Diaconis bin-width estimate to a "nice" number
/// (1, 2, 5, 10, …) the way wolframscript does: of the two nice candidates
/// bracketing the estimate, one the data is commensurate with (every value
/// an integer multiple — those bins get centered on the values) wins even
/// when the other is log-nearer; otherwise the log-nearest is chosen.
/// Verified against wolframscript: integer data with estimate 1.82 gets
/// dx = 1 (commensurate) over the nearer 2, while even-valued data with
/// estimate 9.28 gets the plain nearest 10 (2 is not a bracketing
/// candidate, and neither 5 nor 10 is commensurate).
fn wl_nice_bin_width(est: f64, values: &[f64]) -> f64 {
  if est <= 0.0 {
    return 1.0;
  }
  let base = 10f64.powi(est.log10().floor() as i32);
  let candidates = [base, 2.0 * base, 5.0 * base, 10.0 * base];
  let hi_idx = candidates.iter().position(|&c| c >= est).unwrap_or(3);
  let (lo, hi) = (candidates[hi_idx.saturating_sub(1)], candidates[hi_idx]);
  let lo_comm = all_multiples_of(values, lo);
  let hi_comm = all_multiples_of(values, hi);
  if lo_comm != hi_comm {
    return if lo_comm { lo } else { hi };
  }
  if (est / lo).ln().abs() <= (hi / est).ln().abs() {
    lo
  } else {
    hi
  }
}

/// Wolfram-compatible 1-D histogram bin placement, shared by HistogramList
/// and BubbleHistogram. Returns `(min_edge, max_edge, dx, centered)` for
/// auto-binning (`width == None`, Freedman-Diaconis estimate rounded via
/// `wl_nice_bin_width`) or an explicit bin width. `centered` is set when the
/// data is commensurate with `dx` and the bins are centered on the values;
/// wolframscript then reports the edges as reals even when integer-valued.
/// `values` must be non-empty.
pub(crate) fn wl_bin_spec(
  values: &[f64],
  width: Option<f64>,
) -> (f64, f64, f64, bool) {
  let (data_min, data_max, dx) = match width {
    Some(dx) => {
      let mn = values.iter().cloned().fold(f64::INFINITY, f64::min);
      let mx = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
      (mn, mx, dx)
    }
    None => {
      let mut sorted = values.to_vec();
      sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
      let data_min = sorted[0];
      let data_max = sorted[sorted.len() - 1];
      let n = sorted.len() as f64;
      let iqr = interquartile_range(&sorted);
      let dx = if iqr > 0.0 {
        wl_nice_bin_width(2.0 * iqr / n.cbrt(), &sorted)
      } else if data_max > data_min {
        // Fallback: use range / Sturges' rule
        let sturges_bins = (n.log2() + 1.0).ceil().max(1.0);
        wl_nice_bin_width((data_max - data_min) / sturges_bins, &sorted)
      } else {
        // All values identical
        let v = data_min.abs();
        if v == 0.0 { 1.0 } else { nice_number(v) }
      };
      (data_min, data_max, dx)
    }
  };
  if data_max > data_min && all_multiples_of(values, dx) {
    // Center bins on the values: edges at value ± dx/2.
    (data_min - dx / 2.0, data_max + dx / 2.0, dx, true)
  } else {
    let lo = (data_min / dx).floor() * dx;
    let mut hi = (data_max / dx).ceil() * dx;
    // Ensure data_max is strictly inside the last bin
    if (data_max - hi).abs() < 1e-12 * dx.max(1.0) {
      hi += dx;
    }
    (lo, hi, dx, false)
  }
}

/// Bin edges for the `HistogramList[data, n]` bin-count spec, both as exact
/// (integer/rational) or Real `Expr`s for display and as `f64`s for counting.
pub(crate) struct NBinEdges {
  pub(crate) exprs: Vec<Expr>,
  pub(crate) f64s: Vec<f64>,
}

/// wolframscript's `Equal` on machine reals: equal when they agree in all
/// but their last seven binary digits (relative tolerance 2^-46).
fn ws_machine_eq(a: f64, b: f64) -> bool {
  a == b || (a - b).abs() <= a.abs().max(b.abs()) * 2f64.powi(-46)
}

/// The f64 value of `mant * 10^e` (correctly rounded while `10^|e|` fits in
/// an i128; the astronomically large/small tail falls back to `powi`).
fn nice_width_f64(mant: i128, e: i32) -> f64 {
  if e >= 0 {
    match 10i128
      .checked_pow(e as u32)
      .and_then(|p| p.checked_mul(mant))
    {
      Some(v) => v as f64,
      None => mant as f64 * 10f64.powi(e),
    }
  } else {
    match 10i128.checked_pow(e.unsigned_abs()) {
      Some(d) => mant as f64 / d as f64,
      None => mant as f64 * 10f64.powi(e),
    }
  }
}

/// wolframscript's `defaultBinwidthSmoother`: snap a width to the linearly
/// nearest of {1, 2, 5, 10}*10^Floor[Log[10, w]] (ties to the smaller),
/// returned as an exact `(mantissa, exponent)` pair with mantissa 1/2/5.
/// `None` when the width is ~0 (all data identical).
fn ws_smooth_width(w: f64) -> Option<(i128, i32)> {
  if w.abs() <= 2.0 * f64::EPSILON {
    return None;
  }
  let e = (w.ln() / 10f64.ln()).floor() as i32;
  let mut best = (1i128, e);
  let mut best_diff = f64::INFINITY;
  for mant in [1i128, 2, 5, 10] {
    let diff = (nice_width_f64(mant, e) - w).abs();
    if diff < best_diff {
      best_diff = diff;
      best = (mant, e);
    }
  }
  Some(if best.0 == 10 { (1, best.1 + 1) } else { best })
}

/// The mantissa test from wolframscript's `BinOffset`: the granularity's
/// `MantissaExponent` mantissa equals 0.1/0.2/0.5/1. up to machine-`Equal`
/// tolerance (so float noise like 0.09999999999999998 still passes as 1.).
fn ws_gran_mantissa_nice(g: f64) -> bool {
  if !(g > 0.0) || !g.is_finite() {
    return false;
  }
  let e = (g.ln() / 10f64.ln()).floor() as i32;
  let mut m = g / 10f64.powi(e + 1);
  while m >= 1.0 {
    m /= 10.0;
  }
  while m < 0.1 {
    m *= 10.0;
  }
  [0.1, 0.2, 0.5, 1.0].iter().any(|&t| ws_machine_eq(m, t))
}

/// `n * 10^e` as an exact Integer (e >= 0) or reduced Rational (e < 0).
fn pow10_multiple_expr(n: i128, e: i32) -> Option<Expr> {
  if e >= 0 {
    Some(Expr::Integer(n.checked_mul(10i128.checked_pow(e as u32)?)?))
  } else {
    let mut num = n;
    let mut den = 10i128.checked_pow(e.unsigned_abs())?;
    let g = gcd_i128(num, den).max(1);
    num /= g;
    den /= g;
    Some(if den == 1 {
      Expr::Integer(num)
    } else {
      Expr::FunctionCall {
        name: "Rational".to_string(),
        args: vec![Expr::Integer(num), Expr::Integer(den)].into(),
      }
    })
  }
}

/// All-identical data: exactly `n` machine-real bins over mean ± 1/2
/// (wolframscript's `min === max` branch with `steps = 1/n`).
fn zero_range_bins(values: &[f64], n: usize) -> NBinEdges {
  let mean = values.iter().sum::<f64>() / values.len() as f64;
  let base = mean - 0.5;
  let inv = 1.0 / n as f64;
  let f64s: Vec<f64> = (0..=n).map(|i| base + i as f64 * inv).collect();
  NBinEdges {
    exprs: f64s.iter().map(|&x| Expr::Real(x)).collect(),
    f64s,
  }
}

/// Bin edges for `HistogramList[data, n]`, replicating wolframscript's
/// internal `userBinningN` exactly (decoded from the kernel's DownValues):
///
///   delta = (max-min)/(n-1)            (plain max-min for n == 1)
///   delta = Max[delta, granularity]    (granularity = smallest nonzero gap
///                                       between sorted values; exactly 1
///                                       for single-point data)
///   delta = smoother(delta)            (nearest of {1,2,5,10}*10^k, exact)
///   edges = Range[Floor[(min+eps)(1+2 eps), delta],
///                 Ceiling[(max+eps)(1+2 eps), delta], delta] - offset
///
/// The epsilon fudge pushes a maximum lying exactly on an edge into an extra
/// bin (and, for negative on-edge minima, adds an extra bin below — a
/// wolframscript quirk this reproduces faithfully). For n == 1 only the two
/// end edges are kept. The `offset` is granularity/2 when granularity equals
/// delta (machine-`Equal` tolerance), its mantissa is 1/2/5, and the FIRST
/// data value is a multiple of it — that centers the bins on the values and
/// turns the edges into machine Reals; otherwise the edges stay exact
/// integers/rationals. `values` must be in the original data order.
pub(crate) fn wl_user_binning_n(values: &[f64], n: i128) -> Option<NBinEdges> {
  if values.is_empty() || !(1..=1_000_000).contains(&n) {
    return None;
  }
  let nn = n as usize;
  let min_d = values.iter().cloned().fold(f64::INFINITY, f64::min);
  let max_d = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
  if !min_d.is_finite() || !max_d.is_finite() {
    return None;
  }
  let raw = if n == 1 {
    max_d - min_d
  } else {
    (max_d - min_d) / (n - 1) as f64
  };
  let single = values.len() == 1;
  let gran = if single {
    1.0
  } else {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let g = sorted
      .windows(2)
      .map(|w| w[1] - w[0])
      .filter(|&d| d != 0.0)
      .fold(f64::INFINITY, f64::min);
    if g.is_finite() { g } else { 0.0 }
  };

  let (mant, e) = match ws_smooth_width(gran.max(raw)) {
    None => return Some(zero_range_bins(values, nn)),
    Some(pair) => pair,
  };
  let delta_f = nice_width_f64(mant, e);
  let x_lo = (min_d + f64::EPSILON) * (1.0 + 2.0 * f64::EPSILON);
  let x_hi = (max_d + f64::EPSILON) * (1.0 + 2.0 * f64::EPSILON);
  let lo = (x_lo / delta_f).floor();
  let hi = (x_hi / delta_f).ceil();
  if !(lo.abs() < 4.0e18 && hi.abs() < 4.0e18) {
    return None;
  }
  let (m_lo, m_hi) = (lo as i128, hi as i128);
  if m_lo >= m_hi {
    return Some(zero_range_bins(values, nn));
  }
  let multiples: Vec<i128> = if n == 1 {
    vec![m_lo, m_hi]
  } else {
    if m_hi - m_lo > 1_000_000 {
      return None;
    }
    (m_lo..=m_hi).collect()
  };

  // BinOffset[delta, granularity, Automatic, first]: half-granularity
  // centering (using the first data value only — wolframscript's result
  // really does depend on the data order here).
  let centered = ws_gran_mantissa_nice(gran)
    && ws_machine_eq(gran, delta_f)
    && ws_machine_eq((values[0] / gran).round_ties_even() * gran, values[0]);

  let mut exprs = Vec::with_capacity(multiples.len());
  let mut f64s = Vec::with_capacity(multiples.len());
  for &m in &multiples {
    let num = m.checked_mul(mant)?;
    let ef = nice_width_f64(num, e);
    if centered && single {
      // Single point: granularity is the exact integer 1 (and so is delta),
      // so the half-bin offset stays exact: m - 1/2 = (2m - 1)/2.
      exprs.push(Expr::FunctionCall {
        name: "Rational".to_string(),
        args: vec![
          Expr::Integer(num.checked_mul(2)?.checked_sub(1)?),
          Expr::Integer(2),
        ]
        .into(),
      });
      f64s.push(ef - 0.5);
    } else if centered {
      let o = gran / 2.0;
      exprs.push(Expr::Real(ef - o));
      f64s.push(ef - o);
    } else {
      exprs.push(pow10_multiple_expr(num, e)?);
      f64s.push(ef);
    }
  }
  Some(NBinEdges { exprs, f64s })
}

/// Count `values` into the strict half-open bins `[e_i, e_{i+1})` defined by
/// `edges` — wolframscript's bin-count path really does drop a value lying
/// exactly on the last edge (or outside the edges entirely).
fn count_into_edges(values: &[f64], edges: &[f64]) -> Vec<i128> {
  let nb = edges.len().saturating_sub(1);
  let mut counts = vec![0i128; nb];
  for &v in values {
    let k = edges.partition_point(|&edge| edge <= v);
    if k > 0 && k <= nb {
      counts[k - 1] += 1;
    }
  }
  counts
}

/// HistogramList[data] - returns {bin_edges, counts}
/// HistogramList[data, n] - target bin count
/// HistogramList[data, {dx}] - explicit bin width
/// HistogramList[data, {min, max, dx}] - explicit bin specification
pub fn histogram_list_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let data = match &args[0] {
    Expr::List(items) => items,
    _ => {
      crate::emit_message(&format!(
        "HistogramList::ldata: {} is not a valid dataset or list of datasets.",
        crate::syntax::expr_to_string(&args[0])
      ));
      return Ok(Expr::FunctionCall {
        name: "HistogramList".to_string(),
        args: args.to_vec().into(),
      });
    }
  };

  let values: Vec<f64> = data.iter().filter_map(numeric_expr_to_f64).collect();

  if values.is_empty() {
    return Ok(Expr::List(
      vec![Expr::List(vec![].into()), Expr::List(vec![].into())].into(),
    ));
  }

  // `force_real_edges` is set when the bins are centered on commensurate data:
  // wolframscript then reports the edges as reals (e.g. `{0.5, 1.5, 2.5}` and
  // `{1., 3., 5.}`) even when they are integer-valued.
  let (min_val, max_val, dx, force_real_edges) = if args.len() == 1 {
    // Auto-binning: Freedman-Diaconis rule with nice number rounding
    wl_bin_spec(&values, None)
  } else if args.len() == 2 {
    match &args[1] {
      // HistogramList[data, n] — target bin count
      Expr::Integer(n) if *n >= 1 => {
        let bins = match wl_user_binning_n(&values, *n) {
          Some(b) => b,
          None => {
            return Ok(Expr::FunctionCall {
              name: "HistogramList".to_string(),
              args: args.to_vec().into(),
            });
          }
        };
        let counts = count_into_edges(&values, &bins.f64s);
        return Ok(Expr::List(
          vec![
            Expr::List(bins.exprs.into()),
            Expr::List(counts.into_iter().map(Expr::Integer).collect()),
          ]
          .into(),
        ));
      }
      // Invalid bare bin counts emit ::hbins; a positive Real then falls
      // back to automatic binning, everything else stays unevaluated.
      Expr::Integer(_) | Expr::Real(_) => {
        crate::emit_message(&format!(
          "HistogramList::hbins: The bin specification {} cannot be used to determine either how many or which bins to use.",
          crate::syntax::expr_to_string(&args[1])
        ));
        if matches!(&args[1], Expr::Real(v) if *v > 0.0) {
          wl_bin_spec(&values, None)
        } else {
          return Ok(Expr::FunctionCall {
            name: "HistogramList".to_string(),
            args: args.to_vec().into(),
          });
        }
      }
      // HistogramList[data, {dx}]
      Expr::List(spec) if spec.len() == 1 => {
        let dx = match numeric_expr_to_f64(&spec[0]) {
          Some(v) if v > 0.0 => v,
          _ => {
            return Ok(Expr::FunctionCall {
              name: "HistogramList".to_string(),
              args: args.to_vec().into(),
            });
          }
        };
        wl_bin_spec(&values, Some(dx))
      }
      // HistogramList[data, {min, max, dx}]
      Expr::List(spec) if spec.len() == 3 => {
        let min_v = match numeric_expr_to_f64(&spec[0]) {
          Some(v) => v,
          None => {
            return Ok(Expr::FunctionCall {
              name: "HistogramList".to_string(),
              args: args.to_vec().into(),
            });
          }
        };
        let max_v = match numeric_expr_to_f64(&spec[1]) {
          Some(v) => v,
          None => {
            return Ok(Expr::FunctionCall {
              name: "HistogramList".to_string(),
              args: args.to_vec().into(),
            });
          }
        };
        let dx = match numeric_expr_to_f64(&spec[2]) {
          Some(v) if v > 0.0 => v,
          _ => {
            return Ok(Expr::FunctionCall {
              name: "HistogramList".to_string(),
              args: args.to_vec().into(),
            });
          }
        };
        (min_v, max_v, dx, false)
      }
      _ => {
        return Ok(Expr::FunctionCall {
          name: "HistogramList".to_string(),
          args: args.to_vec().into(),
        });
      }
    }
  } else {
    return Ok(Expr::FunctionCall {
      name: "HistogramList".to_string(),
      args: args.to_vec().into(),
    });
  };

  if dx <= 0.0 {
    return Err(InterpreterError::EvaluationError(
      "HistogramList: bin width must be positive".into(),
    ));
  }

  // Only whole bins [min + i*dx, min + (i+1)*dx) that fit within [min, max]
  // are produced; a trailing partial range is dropped and out-of-range values
  // are not capped into the last bin.
  let num_bins = ((max_val - min_val) / dx + 1e-9).floor().max(0.0) as usize;
  if num_bins == 0 {
    return Ok(Expr::List(
      vec![Expr::List(vec![].into()), Expr::List(vec![].into())].into(),
    ));
  }

  let mut counts = vec![0i128; num_bins];
  for &v in &values {
    if v < min_val {
      continue;
    }
    let idx = ((v - min_val) / dx).floor();
    if idx >= 0.0 && (idx as usize) < num_bins {
      counts[idx as usize] += 1;
    }
  }

  // Build bin edges list
  let mut edges = Vec::with_capacity(num_bins + 1);
  for i in 0..=num_bins {
    let e = min_val + i as f64 * dx;
    edges.push(if force_real_edges {
      Expr::Real(e)
    } else {
      f64_to_expr(e)
    });
  }

  Ok(Expr::List(
    vec![
      Expr::List(edges.into()),
      Expr::List(counts.into_iter().map(Expr::Integer).collect()),
    ]
    .into(),
  ))
}

/// TakeLargestBy[list, f, n] - take the n largest elements sorted by f
/// Resolve a `TakeLargestBy`/`TakeSmallestBy` count: a non-negative integer,
/// or `UpTo[n]` clamped to `len`. Returns `None` for anything else.
fn take_count_or_upto(spec: &Expr, len: usize) -> Option<usize> {
  match spec {
    Expr::Integer(n) if *n >= 0 => Some(*n as usize),
    Expr::FunctionCall { name, args } if name == "UpTo" && args.len() == 1 => {
      match &args[0] {
        Expr::Integer(n) if *n >= 0 => Some((*n as usize).min(len)),
        _ => None,
      }
    }
    _ => None,
  }
}

/// Shared helper for TakeLargestBy/TakeSmallestBy on an association: rank the
/// key→value pairs by `f` applied to each value and keep the top/bottom `n`,
/// returning an association ordered by that ranking.
fn take_by_assoc(
  pairs: &[(Expr, Expr)],
  f: &Expr,
  n: usize,
  largest: bool,
) -> Result<Expr, InterpreterError> {
  let mut with_keys: Vec<(Expr, (Expr, Expr))> = Vec::new();
  for (k, v) in pairs {
    let key = crate::evaluator::evaluate_expr_to_expr(&Expr::FunctionCall {
      name: "Apply".to_string(),
      args: vec![f.clone(), Expr::List(vec![v.clone()].into())].into(),
    })?;
    with_keys.push((key, (k.clone(), v.clone())));
  }
  with_keys.sort_by(|a, b| {
    let ord = compare_exprs(&a.0, &b.0);
    // ord > 0 means a.key < b.key.
    let base = if ord > 0 {
      std::cmp::Ordering::Greater
    } else if ord < 0 {
      std::cmp::Ordering::Less
    } else {
      std::cmp::Ordering::Equal
    };
    if largest { base } else { base.reverse() }
  });
  let result: Vec<(Expr, Expr)> =
    with_keys.into_iter().take(n).map(|(_, kv)| kv).collect();
  Ok(Expr::Association(result))
}

pub fn take_largest_by_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 3 {
    return Err(InterpreterError::EvaluationError(
      "TakeLargestBy expects exactly 3 arguments".into(),
    ));
  }
  if let Expr::Association(pairs) = &args[0] {
    match take_count_or_upto(&args[2], pairs.len()) {
      Some(n) => return take_by_assoc(pairs, &args[1], n, true),
      None => {
        return Ok(Expr::FunctionCall {
          name: "TakeLargestBy".to_string(),
          args: args.to_vec().into(),
        });
      }
    }
  }
  let list = match &args[0] {
    Expr::List(items) => items,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "TakeLargestBy".to_string(),
        args: args.to_vec().into(),
      });
    }
  };
  let f = &args[1];
  let n = match take_count_or_upto(&args[2], list.len()) {
    Some(n) => n,
    None => {
      return Ok(Expr::FunctionCall {
        name: "TakeLargestBy".to_string(),
        args: args.to_vec().into(),
      });
    }
  };

  // Compute f[item] for each item
  let mut with_keys: Vec<(Expr, Expr)> = Vec::new();
  for item in list {
    let key = crate::evaluator::evaluate_expr_to_expr(&Expr::FunctionCall {
      name: "Apply".to_string(),
      args: vec![f.clone(), Expr::List(vec![item.clone()].into())].into(),
    })?;
    with_keys.push((key, item.clone()));
  }
  // Sort descending by key (largest first)
  // compare_exprs returns 1 if first < second in canonical order
  with_keys.sort_by(|a, b| {
    let ord = compare_exprs(&a.0, &b.0);
    // ord > 0 means a.key < b.key, so b should come first (descending)
    if ord > 0 {
      std::cmp::Ordering::Greater
    } else if ord < 0 {
      std::cmp::Ordering::Less
    } else {
      std::cmp::Ordering::Equal
    }
  });
  let result: Vec<Expr> =
    with_keys.into_iter().take(n).map(|(_, v)| v).collect();
  Ok(Expr::List(result.into()))
}

/// TakeSmallestBy[list, f, n] - take the n smallest elements sorted by f
pub fn take_smallest_by_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 3 {
    return Err(InterpreterError::EvaluationError(
      "TakeSmallestBy expects exactly 3 arguments".into(),
    ));
  }
  if let Expr::Association(pairs) = &args[0] {
    match take_count_or_upto(&args[2], pairs.len()) {
      Some(n) => return take_by_assoc(pairs, &args[1], n, false),
      None => {
        return Ok(Expr::FunctionCall {
          name: "TakeSmallestBy".to_string(),
          args: args.to_vec().into(),
        });
      }
    }
  }
  let list = match &args[0] {
    Expr::List(items) => items,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "TakeSmallestBy".to_string(),
        args: args.to_vec().into(),
      });
    }
  };
  let f = &args[1];
  let n = match take_count_or_upto(&args[2], list.len()) {
    Some(n) => n,
    None => {
      return Ok(Expr::FunctionCall {
        name: "TakeSmallestBy".to_string(),
        args: args.to_vec().into(),
      });
    }
  };

  // Compute f[item] for each item
  let mut with_keys: Vec<(Expr, Expr)> = Vec::new();
  for item in list {
    let key = crate::evaluator::evaluate_expr_to_expr(&Expr::FunctionCall {
      name: "Apply".to_string(),
      args: vec![f.clone(), Expr::List(vec![item.clone()].into())].into(),
    })?;
    with_keys.push((key, item.clone()));
  }
  // Sort ascending by key (smallest first)
  // compare_exprs returns 1 if first < second in canonical order
  with_keys.sort_by(|a, b| {
    let ord = compare_exprs(&a.0, &b.0);
    // ord > 0 means a.key < b.key, so a should come first (ascending)
    if ord > 0 {
      std::cmp::Ordering::Less
    } else if ord < 0 {
      std::cmp::Ordering::Greater
    } else {
      std::cmp::Ordering::Equal
    }
  });
  let result: Vec<Expr> =
    with_keys.into_iter().take(n).map(|(_, v)| v).collect();
  Ok(Expr::List(result.into()))
}

/// AllSameBy[list, f] - True if f[x] gives the same value for all elements
pub fn all_same_by_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "AllSameBy expects 2 arguments".into(),
    ));
  }
  let items = match &args[0] {
    Expr::List(items) => items,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "AllSameBy".to_string(),
        args: args.to_vec().into(),
      });
    }
  };
  if items.is_empty() {
    return Ok(bool_to_expr(true));
  }
  let first_val = apply_func_ast(&args[1], &items[0])?;
  let first_str = crate::syntax::expr_to_string(&first_val);
  for item in &items[1..] {
    let val = apply_func_ast(&args[1], item)?;
    if crate::syntax::expr_to_string(&val) != first_str {
      return Ok(bool_to_expr(false));
    }
  }
  Ok(bool_to_expr(true))
}

/// `ClusteringComponents[list]` — assign each element of a 1-D numeric list
/// to a cluster index. Uses a single-largest-gap split: sort the values,
/// find the largest consecutive gap, and partition into two groups at that
/// gap. Returns `{1, 1, …, 2, 2, …}` mapping each input position to a
/// cluster index — matching wolframscript's
/// `ClusteringComponents[{1, 2, 3, 1, 2, 10, 100}]` → `{1, 1, 1, 1, 1, 1, 2}`.
///
/// Falls back to the unevaluated form for inputs that aren't a flat list of
/// numbers — multi-dimensional or symbolic clustering would need a more
/// general algorithm.
pub fn clustering_components_ast(
  list: &Expr,
) -> Result<Expr, InterpreterError> {
  let items = match list {
    Expr::List(items) => items.clone(),
    _ => {
      crate::emit_message(
        "ClusteringComponents::nosup: This type of data is not supported.",
      );
      return Ok(Expr::FunctionCall {
        name: "ClusteringComponents".to_string(),
        args: vec![list.clone()].into(),
      });
    }
  };
  if items.is_empty() {
    return Ok(Expr::List(vec![].into()));
  }
  let mut values: Vec<f64> = Vec::with_capacity(items.len());
  for item in &items {
    if let Some(v) = expr_to_f64(item) {
      values.push(v);
    } else {
      return Ok(Expr::FunctionCall {
        name: "ClusteringComponents".to_string(),
        args: vec![list.clone()].into(),
      });
    }
  }
  // All identical: a single cluster.
  let (min, max) = values
    .iter()
    .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), v| {
      (lo.min(*v), hi.max(*v))
    });
  if min == max {
    return Ok(Expr::List(vec![Expr::Integer(1); values.len()].into()));
  }
  // Find the largest gap between consecutive sorted values.
  let mut sorted = values.clone();
  sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
  let mut max_gap = 0.0_f64;
  let mut max_gap_lower = sorted[0];
  for w in sorted.windows(2) {
    let gap = w[1] - w[0];
    if gap > max_gap {
      max_gap = gap;
      max_gap_lower = w[0];
    }
  }
  // Threshold: half-way through the largest gap. Values ≤ threshold go to
  // cluster 1 (the lower band), the rest to cluster 2.
  let threshold = max_gap_lower + max_gap / 2.0;
  let labels: Vec<Expr> = values
    .iter()
    .map(|v| Expr::Integer(if *v <= threshold { 1 } else { 2 }))
    .collect();
  Ok(Expr::List(labels.into()))
}

/// `ClusteringComponents[list, n]` — split a 1-D numeric list into
/// `n` clusters by cutting at the `n - 1` largest consecutive gaps
/// in the sorted values, then label each input position by the
/// cluster index of its value (lower bands = lower labels).
///
/// This matches wolframscript's *partition* for the audit cases
/// (e.g. `{1, 2, 3, 7, 8}` with `n = 2` → `{1, 1, 1, 2, 2}`); the
/// label *numbering* differs from wolframscript when wolframscript
/// chooses a non-ascending order — both are valid clusterings of the
/// same data.
pub fn clustering_components_n_ast(
  list: &Expr,
  n_expr: &Expr,
) -> Result<Expr, InterpreterError> {
  let n = match n_expr {
    Expr::Integer(k) if *k >= 1 => *k as usize,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "ClusteringComponents".to_string(),
        args: vec![list.clone(), n_expr.clone()].into(),
      });
    }
  };
  let items = match list {
    Expr::List(items) => items.clone(),
    _ => {
      crate::emit_message(
        "ClusteringComponents::nosup: This type of data is not supported.",
      );
      return Ok(Expr::FunctionCall {
        name: "ClusteringComponents".to_string(),
        args: vec![list.clone(), n_expr.clone()].into(),
      });
    }
  };
  if items.is_empty() {
    return Ok(Expr::List(vec![].into()));
  }
  let mut values: Vec<f64> = Vec::with_capacity(items.len());
  for item in &items {
    if let Some(v) = expr_to_f64(item) {
      values.push(v);
    } else {
      return Ok(Expr::FunctionCall {
        name: "ClusteringComponents".to_string(),
        args: vec![list.clone(), n_expr.clone()].into(),
      });
    }
  }

  // n = 1: everything in cluster 1.
  if n == 1 {
    return Ok(Expr::List(vec![Expr::Integer(1); values.len()].into()));
  }
  // n ≥ number of unique values: each unique value forms its own
  // cluster; pad to at most `values.len()` distinct clusters.
  let max_clusters = values.len().min(n);

  // Sort with original indices so we can find the largest gaps.
  let mut indexed: Vec<(usize, f64)> =
    values.iter().enumerate().map(|(i, &v)| (i, v)).collect();
  indexed
    .sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

  // Gaps between consecutive sorted values.
  let mut gaps: Vec<(usize, f64)> =
    Vec::with_capacity(indexed.len().saturating_sub(1));
  for i in 0..indexed.len().saturating_sub(1) {
    gaps.push((i, indexed[i + 1].1 - indexed[i].1));
  }
  // Pick the top `max_clusters - 1` gap *positions*. Sort by gap size
  // descending, take the first `max_clusters - 1`, then re-sort by
  // position so the cluster boundaries are in left-to-right order.
  let cuts_needed = max_clusters.saturating_sub(1);
  gaps
    .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
  let mut cut_positions: Vec<usize> =
    gaps.iter().take(cuts_needed).map(|(p, _)| *p).collect();
  cut_positions.sort_unstable();

  // Assign cluster labels (1-indexed, ascending in value).
  let mut sorted_labels: Vec<usize> = vec![0; indexed.len()];
  let mut cluster = 1usize;
  let mut cut_idx = 0usize;
  for (i, label) in sorted_labels.iter_mut().enumerate() {
    if cut_idx < cut_positions.len() && i > cut_positions[cut_idx] {
      cluster += 1;
      cut_idx += 1;
    }
    *label = cluster;
  }

  // Map back to original positions.
  let mut out: Vec<Expr> = vec![Expr::Integer(0); values.len()];
  for (sorted_pos, label) in sorted_labels.iter().enumerate() {
    let orig_idx = indexed[sorted_pos].0;
    out[orig_idx] = Expr::Integer(*label as i128);
  }
  Ok(Expr::List(out.into()))
}

/// Try to read a `FindClusters` input as `keys -> vals` (single rule
/// whose left/right sides are equal-length lists) or as a list of rules
/// `{key -> val, …}`. Returns `(keys, values)` on success.
fn extract_rule_style_input(list: &Expr) -> Option<(Vec<Expr>, Vec<Expr>)> {
  // Form: `{k1, k2, …} -> {v1, v2, …}` (single rule containing two equal-length lists).
  if let Expr::Rule {
    pattern,
    replacement,
  } = list
    && let (Expr::List(ks), Expr::List(vs)) =
      (pattern.as_ref(), replacement.as_ref())
    && ks.len() == vs.len()
    && !ks.is_empty()
  {
    return Some((ks.to_vec(), vs.to_vec()));
  }
  // Form: `{k1 -> v1, k2 -> v2, …}` — every list element is a Rule.
  if let Expr::List(items) = list
    && !items.is_empty()
    && items.iter().all(|e| matches!(e, Expr::Rule { .. }))
  {
    let mut ks = Vec::with_capacity(items.len());
    let mut vs = Vec::with_capacity(items.len());
    for item in items {
      if let Expr::Rule {
        pattern,
        replacement,
      } = item
      {
        ks.push(pattern.as_ref().clone());
        vs.push(replacement.as_ref().clone());
      }
    }
    return Some((ks, vs));
  }
  None
}

/// Cluster `keys` (1-D numeric) via single-largest-gap split, then
/// group the corresponding `vals` and return `{{vals_high}, {vals_low}}`
/// — wolframscript's `FindClusters[{1->a, 2->b, 10->c}]` is
/// `{{c}, {a, b}}`, with the high-key cluster listed first.
fn cluster_keys_emit_values(
  keys: &[Expr],
  vals: &[Expr],
  raw_input: &Expr,
) -> Result<Expr, InterpreterError> {
  // Numeric keys only.
  let mut numeric_keys: Vec<f64> = Vec::with_capacity(keys.len());
  for k in keys {
    match expr_to_f64(k) {
      Some(v) => numeric_keys.push(v),
      None => {
        return Ok(Expr::FunctionCall {
          name: "FindClusters".to_string(),
          args: vec![raw_input.clone()].into(),
        });
      }
    }
  }
  if numeric_keys.is_empty() {
    return Ok(Expr::List(vec![].into()));
  }
  if numeric_keys.len() == 1 {
    return Ok(Expr::List(
      vec![Expr::List(vec![vals[0].clone()].into())].into(),
    ));
  }
  // Locate the largest gap on the *sorted* keys.
  let n = numeric_keys.len();
  let mut sort_idx: Vec<usize> = (0..n).collect();
  sort_idx.sort_by(|&a, &b| {
    numeric_keys[a]
      .partial_cmp(&numeric_keys[b])
      .unwrap_or(std::cmp::Ordering::Equal)
  });
  let mut max_gap = f64::NEG_INFINITY;
  let mut max_gap_pos = 0usize;
  for i in 0..n - 1 {
    let g = numeric_keys[sort_idx[i + 1]] - numeric_keys[sort_idx[i]];
    if g > max_gap {
      max_gap = g;
      max_gap_pos = i;
    }
  }
  // All keys identical — single cluster.
  if max_gap == 0.0 || !max_gap.is_finite() {
    return Ok(Expr::List(vec![Expr::List(vals.to_vec().into())].into()));
  }
  // Threshold halfway through the largest gap.
  let threshold = numeric_keys[sort_idx[max_gap_pos]] + max_gap / 2.0;
  // Bucket vals into low/high groups, preserving input order within each.
  let mut low: Vec<Expr> = Vec::new();
  let mut high: Vec<Expr> = Vec::new();
  for i in 0..n {
    if numeric_keys[i] <= threshold {
      low.push(vals[i].clone());
    } else {
      high.push(vals[i].clone());
    }
  }
  // wolframscript prints the high-key cluster first.
  Ok(Expr::List(
    vec![Expr::List(high.into()), Expr::List(low.into())].into(),
  ))
}

/// `FindClusters[list]` / `FindClusters[list, k]` — dispatch entry that
/// hands off to the 1-arg path when no cluster count is given, or runs
/// the explicit `k-1` largest-gap split when `k` is provided. Also
/// accepts a `DistanceFunction -> fn` option, which switches to a
/// distance-based equivalence-class clustering.
pub fn find_clusters_ast_n(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() || args.len() > 3 {
    return Err(InterpreterError::EvaluationError(
      "FindClusters expects 1 to 3 arguments".into(),
    ));
  }
  // Pull out a DistanceFunction option from any trailing arg. Other
  // recognised options (Method, CriterionFunction, …) are accepted
  // and silently ignored — the default algorithm is used regardless.
  let mut distance_fn: Option<Expr> = None;
  let mut k_arg: Option<&Expr> = None;
  for arg in &args[1..] {
    if let Some(fnexpr) = extract_distance_function_option(arg) {
      distance_fn = Some(fnexpr);
    } else if is_ignorable_findclusters_option(arg) {
      // skip — unsupported but recognised; fall back to default
    } else {
      k_arg = Some(arg);
    }
  }
  if let Some(fnexpr) = distance_fn {
    return find_clusters_distance_fn(&args[0], &fnexpr, args);
  }
  if let Some(kexp) = k_arg {
    let k = match kexp {
      Expr::Integer(n) if *n >= 1 => *n as usize,
      _ => {
        return Ok(Expr::FunctionCall {
          name: "FindClusters".to_string(),
          args: args.to_vec().into(),
        });
      }
    };
    return find_clusters_with_k(&args[0], k, args);
  }
  find_clusters_ast(&args[0])
}

/// Recognise `Method`, `CriterionFunction`, `PerformanceGoal`,
/// `WorkingPrecision`, etc. — `FindClusters` options that Woxi
/// accepts but doesn't use to switch algorithm.
fn is_ignorable_findclusters_option(opt: &Expr) -> bool {
  let key = match opt {
    Expr::Rule { pattern, .. } | Expr::RuleDelayed { pattern, .. } => {
      if let Expr::Identifier(s) = pattern.as_ref() {
        s.as_str()
      } else {
        return false;
      }
    }
    Expr::FunctionCall { name, args }
      if (name == "Rule" || name == "RuleDelayed") && args.len() == 2 =>
    {
      if let Expr::Identifier(s) = &args[0] {
        s.as_str()
      } else {
        return false;
      }
    }
    _ => return false,
  };
  matches!(
    key,
    "Method"
      | "CriterionFunction"
      | "PerformanceGoal"
      | "WorkingPrecision"
      | "Weights"
      | "FeatureExtractor"
      | "FeatureNames"
      | "RandomSeeding"
      | "Tolerance"
  )
}

/// Extract `DistanceFunction -> fn` from a single option argument.
fn extract_distance_function_option(opt: &Expr) -> Option<Expr> {
  if let Expr::Rule {
    pattern,
    replacement,
  } = opt
    && let Expr::Identifier(s) = pattern.as_ref()
    && s == "DistanceFunction"
  {
    return Some((**replacement).clone());
  }
  if let Expr::FunctionCall { name, args } = opt
    && (name == "Rule" || name == "RuleDelayed")
    && args.len() == 2
    && let Expr::Identifier(s) = &args[0]
    && s == "DistanceFunction"
  {
    return Some(args[1].clone());
  }
  None
}

/// `FindClusters[list, DistanceFunction -> fn]` — group items into
/// equivalence classes where every internal pairwise distance is zero.
/// Clusters are emitted in descending order of their first item's input
/// index, matching wolframscript on simple cases like the Length-based
/// distance test in the docs.
fn find_clusters_distance_fn(
  list: &Expr,
  distance_fn: &Expr,
  raw_args: &[Expr],
) -> Result<Expr, InterpreterError> {
  let items = match list {
    Expr::List(items) => items.clone(),
    _ => {
      return Ok(Expr::FunctionCall {
        name: "FindClusters".to_string(),
        args: raw_args.to_vec().into(),
      });
    }
  };
  if items.is_empty() {
    return Ok(Expr::List(vec![].into()));
  }
  let n = items.len();
  // Union-Find over indices, joining pairs with zero distance.
  let mut parent: Vec<usize> = (0..n).collect();
  fn find(parent: &mut [usize], x: usize) -> usize {
    let mut root = x;
    while parent[root] != root {
      root = parent[root];
    }
    let mut cur = x;
    while parent[cur] != root {
      let nxt = parent[cur];
      parent[cur] = root;
      cur = nxt;
    }
    root
  }
  for i in 0..n {
    for j in (i + 1)..n {
      let d =
        crate::functions::list_helpers_ast::utilities::apply_func_to_two_args(
          distance_fn,
          &items[i],
          &items[j],
        )?;
      let is_zero = matches!(&d, Expr::Integer(0))
        || matches!(&d, Expr::Real(v) if *v == 0.0);
      if is_zero {
        let ri = find(&mut parent, i);
        let rj = find(&mut parent, j);
        if ri != rj {
          parent[ri] = rj;
        }
      }
    }
  }
  // Group indices by root, preserving input order within each group.
  let mut groups: Vec<(usize, Vec<Expr>)> = Vec::new();
  for i in 0..n {
    let r = find(&mut parent, i);
    if let Some(g) = groups.iter_mut().find(|(rt, _)| *rt == r) {
      g.1.push(items[i].clone());
    } else {
      groups.push((r, vec![items[i].clone()]));
    }
  }
  // Wolframscript orders the clusters with the highest first-input-index
  // first (later-encountered cluster first).
  groups.sort_by(|a, b| b.0.cmp(&a.0));
  let result: Vec<Expr> = groups
    .into_iter()
    .map(|(_, g)| Expr::List(g.into()))
    .collect();
  Ok(Expr::List(result.into()))
}

/// `FindClusters[list, k]` — split a 1-D numeric list into exactly `k`
/// clusters by cutting at the `k - 1` largest gaps between sorted
/// values. Preserves the input ordering inside each cluster, so e.g.
/// `FindClusters[{1, 2, 10, 11, 20, 21}, 2]` → `{{1, 2, 10, 11}, {20, 21}}`.
/// Single-linkage agglomerative clustering of `String` items into `k`
/// groups using `EditDistance` (Levenshtein) as the pairwise metric.
/// Returns clusters in the order of their first input element so the
/// output is deterministic and preserves input order within each
/// cluster.
fn agglomerative_cluster_strings(items: &[Expr], k: usize) -> Expr {
  let n = items.len();
  if n == 0 {
    return Expr::List(vec![].into());
  }
  if k <= 1 {
    return Expr::List(vec![Expr::List(items.to_vec().into())].into());
  }
  if k >= n {
    return Expr::List(
      items
        .iter()
        .map(|e| Expr::List(vec![e.clone()].into()))
        .collect(),
    );
  }
  let strs: Vec<&str> = items
    .iter()
    .map(|e| match e {
      Expr::String(s) => s.as_str(),
      _ => "",
    })
    .collect();
  // Pairwise EditDistance.
  let dist =
    |i: usize, j: usize| -> u32 { edit_distance_str(strs[i], strs[j]) };
  // Each item starts in its own cluster (cluster id == initial item idx).
  let mut cluster_of: Vec<usize> = (0..n).collect();
  let mut active: std::collections::BTreeSet<usize> = (0..n).collect();
  // Repeatedly merge the two closest clusters until `k` remain.
  while active.len() > k {
    let mut best: Option<(u32, usize, usize)> = None;
    let act: Vec<usize> = active.iter().copied().collect();
    for i in 0..act.len() {
      for j in (i + 1)..act.len() {
        let ci = act[i];
        let cj = act[j];
        // Single-linkage: minimum pairwise distance between any items
        // currently labelled `ci` vs any items labelled `cj`.
        let mut min_d = u32::MAX;
        for p in 0..n {
          if cluster_of[p] != ci {
            continue;
          }
          for q in 0..n {
            if cluster_of[q] != cj {
              continue;
            }
            let d = dist(p, q);
            if d < min_d {
              min_d = d;
            }
          }
        }
        if best.is_none_or(|(d, _, _)| min_d < d) {
          best = Some((min_d, ci, cj));
        }
      }
    }
    let (_, ci, cj) = best.unwrap();
    // Merge cj into ci.
    for label in &mut cluster_of {
      if *label == cj {
        *label = ci;
      }
    }
    active.remove(&cj);
  }
  // Emit clusters in the order their first member appears in the input.
  let mut seen: std::collections::BTreeSet<usize> =
    std::collections::BTreeSet::new();
  let mut order: Vec<usize> = Vec::new();
  for i in 0..n {
    if seen.insert(cluster_of[i]) {
      order.push(cluster_of[i]);
    }
  }
  let groups: Vec<Expr> = order
    .into_iter()
    .map(|cid| {
      let members: Vec<Expr> = (0..n)
        .filter(|&i| cluster_of[i] == cid)
        .map(|i| items[i].clone())
        .collect();
      Expr::List(members.into())
    })
    .collect();
  Expr::List(groups.into())
}

/// Levenshtein distance between two byte-strings (ASCII fast path).
fn edit_distance_str(a: &str, b: &str) -> u32 {
  let a: Vec<char> = a.chars().collect();
  let b: Vec<char> = b.chars().collect();
  let (m, n) = (a.len(), b.len());
  if m == 0 {
    return n as u32;
  }
  if n == 0 {
    return m as u32;
  }
  let mut prev: Vec<u32> = (0..=n as u32).collect();
  let mut curr: Vec<u32> = vec![0; n + 1];
  for i in 1..=m {
    curr[0] = i as u32;
    for j in 1..=n {
      let cost = if a[i - 1] == b[j - 1] { 0 } else { 1 };
      curr[j] = (prev[j] + 1).min(curr[j - 1] + 1).min(prev[j - 1] + cost);
    }
    std::mem::swap(&mut prev, &mut curr);
  }
  prev[n]
}

fn find_clusters_with_k(
  list: &Expr,
  k: usize,
  raw_args: &[Expr],
) -> Result<Expr, InterpreterError> {
  let items = match list {
    Expr::List(items) => items.clone(),
    _ => {
      return Ok(Expr::FunctionCall {
        name: "FindClusters".to_string(),
        args: raw_args.to_vec().into(),
      });
    }
  };
  if items.is_empty() {
    return Ok(Expr::List(vec![].into()));
  }
  if k == 1 {
    return Ok(Expr::List(vec![Expr::List(items)].into()));
  }
  if k >= items.len() {
    // One element per cluster.
    return Ok(Expr::List(
      items
        .into_iter()
        .map(|e| Expr::List(vec![e].into()))
        .collect(),
    ));
  }
  // Convert items to f64 for gap analysis. If any item isn't numeric,
  // try the string-input path via agglomerative clustering on
  // `EditDistance`.
  let mut values: Vec<f64> = Vec::with_capacity(items.len());
  let mut all_numeric = true;
  for item in &items {
    match expr_to_f64(item) {
      Some(v) => values.push(v),
      None => {
        all_numeric = false;
        break;
      }
    }
  }
  if !all_numeric {
    if items.iter().all(|e| matches!(e, Expr::String(_))) {
      return Ok(agglomerative_cluster_strings(&items, k));
    }
    return Ok(Expr::FunctionCall {
      name: "FindClusters".to_string(),
      args: raw_args.to_vec().into(),
    });
  }
  // Sort indices by value to find gaps in ascending value order.
  let n = values.len();
  let mut sort_idx: Vec<usize> = (0..n).collect();
  sort_idx.sort_by(|&a, &b| {
    values[a]
      .partial_cmp(&values[b])
      .unwrap_or(std::cmp::Ordering::Equal)
  });
  // Compute (gap, position-in-sorted) for each adjacent pair.
  let mut gaps: Vec<(f64, usize)> = (0..n - 1)
    .map(|i| (values[sort_idx[i + 1]] - values[sort_idx[i]], i))
    .collect();
  // Pick the (k-1) largest-gap positions.
  gaps
    .sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
  let mut cut_positions: Vec<usize> =
    gaps.iter().take(k - 1).map(|(_, p)| *p).collect();
  cut_positions.sort();
  // Assign each input index a cluster label (1..=k) based on its
  // position in the sorted order and the cut positions. Lower-value
  // elements end up in cluster 1.
  let mut labels: Vec<usize> = vec![0; n];
  let mut cluster: usize = 1;
  let mut next_cut = 0usize;
  for (sorted_pos, &orig_idx) in sort_idx.iter().enumerate() {
    if next_cut < cut_positions.len() && sorted_pos > cut_positions[next_cut] {
      cluster += 1;
      next_cut += 1;
    }
    labels[orig_idx] = cluster;
  }
  // Group elements by cluster label, preserving input order.
  let mut groups: Vec<Vec<Expr>> = vec![Vec::new(); k];
  for (i, item) in items.iter().enumerate() {
    let c = labels[i];
    if c >= 1 && c <= k {
      groups[c - 1].push(item.clone());
    }
  }
  let result: Vec<Expr> = groups
    .into_iter()
    .filter(|g| !g.is_empty())
    .map(|v| Expr::List(v.into()))
    .collect();
  Ok(Expr::List(result.into()))
}

/// `FindClusters[list]` — partition `list` into clusters, returning each
/// cluster as a sublist. Reuses `ClusteringComponents` to obtain a label
/// per element, then groups by label in cluster-id order.
///
/// Falls back to the unevaluated form for inputs that
/// `ClusteringComponents` can't label (e.g. non-numeric data or shapes
/// the underlying algorithm doesn't yet handle).
pub fn find_clusters_ast(list: &Expr) -> Result<Expr, InterpreterError> {
  // Rule-style inputs: `{key -> val, ...}` and `keys -> vals` — cluster by
  // the *keys* and emit the matching values, ordered with the high-key
  // cluster first (matching wolframscript's
  // `FindClusters[{1->a, 2->b, 10->c}]` → `{{c}, {a, b}}`).
  if let Some((keys, vals)) = extract_rule_style_input(list) {
    return cluster_keys_emit_values(&keys, &vals, list);
  }
  let items = match list {
    Expr::List(items) => items.clone(),
    _ => {
      return Ok(Expr::FunctionCall {
        name: "FindClusters".to_string(),
        args: vec![list.clone()].into(),
      });
    }
  };
  if items.is_empty() {
    return Ok(Expr::List(vec![].into()));
  }
  // Delegate the labeling step to ClusteringComponents.
  let components = clustering_components_ast(list)?;
  let labels = match &components {
    Expr::List(ls) if ls.len() == items.len() => ls.clone(),
    _ => {
      return Ok(Expr::FunctionCall {
        name: "FindClusters".to_string(),
        args: vec![list.clone()].into(),
      });
    }
  };
  // Determine the highest cluster id; group elements by id, preserving
  // the input order within each group.
  let mut max_id: i128 = 0;
  for l in &labels {
    if let Expr::Integer(k) = l {
      if *k > max_id {
        max_id = *k;
      }
    } else {
      return Ok(Expr::FunctionCall {
        name: "FindClusters".to_string(),
        args: vec![list.clone()].into(),
      });
    }
  }
  let n = max_id as usize;
  let mut groups: Vec<Vec<Expr>> = vec![Vec::new(); n];
  for (item, label) in items.iter().zip(labels.iter()) {
    if let Expr::Integer(k) = label
      && *k >= 1
      && (*k as usize) <= n
    {
      groups[(*k as usize) - 1].push(item.clone());
    }
  }
  // Drop empty clusters (defensive — shouldn't occur with valid labels).
  let result: Vec<Expr> = groups
    .into_iter()
    .filter(|g| !g.is_empty())
    .map(|v| Expr::List(v.into()))
    .collect();
  Ok(Expr::List(result.into()))
}
