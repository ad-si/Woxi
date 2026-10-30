//! Folding arithmetic over univariate interval `Piecewise` expressions into
//! one `Piecewise`: `u^2 Piecewise[{{1, 0 <= u < 1}}, 0] + 4 Piecewise[{{1,
//! 1 <= u < 2}}, 0]` is `Piecewise[{{u^2, 0 <= u < 1}, {4, 1 <= u < 2}}, 0]`.
//!
//! Every condition of every `Piecewise` must compare one variable `x` with
//! numbers. The real line is cut at all of their bounds into points and open
//! intervals; on each the expression is evaluated with every `Piecewise`
//! replaced by the value it selects there, and adjacent stretches with the
//! same value merge into one case.

use crate::InterpreterError;
use crate::syntax::{ComparisonOp, Expr};

fn call(name: &str, args: Vec<Expr>) -> Expr {
  Expr::FunctionCall {
    name: name.to_string(),
    args: args.into(),
  }
}

fn eval(e: &Expr) -> Result<Expr, InterpreterError> {
  crate::evaluator::evaluate_expr_to_expr(e)
}

fn is_piecewise(e: &Expr) -> bool {
  matches!(e, Expr::FunctionCall { name, args } if name == "Piecewise" && !args.is_empty())
}

/// Every Piecewise inside the arithmetic of `e` (not inside other heads).
fn collect_piecewise<'a>(e: &'a Expr, out: &mut Vec<&'a Expr>) -> bool {
  match e {
    _ if is_piecewise(e) => {
      out.push(e);
      true
    }
    Expr::FunctionCall { name, args }
      if matches!(name.as_str(), "Plus" | "Times" | "Power") =>
    {
      let mut any = false;
      for a in args {
        any |= collect_piecewise(a, out);
      }
      any
    }
    Expr::BinaryOp { left, right, .. } => {
      let l = collect_piecewise(left, out);
      let r = collect_piecewise(right, out);
      l || r
    }
    Expr::UnaryOp { operand, .. } => collect_piecewise(operand, out),
    _ => false,
  }
}

/// The variable and numeric bounds a condition compares, or None when it is
/// not a comparison of one variable with numbers.
fn condition_bounds(
  cond: &Expr,
  var: &mut Option<String>,
  bounds: &mut Vec<Expr>,
) -> bool {
  let mut operand = |o: &Expr| -> bool {
    if let Expr::Identifier(n) = o
      && !matches!(n.as_str(), "True" | "False")
      && crate::functions::math_ast::try_eval_to_f64(o).is_none()
    {
      match var {
        Some(v) if v != n => false,
        _ => {
          *var = Some(n.clone());
          true
        }
      }
    } else if crate::functions::math_ast::try_eval_to_f64(o).is_some() {
      bounds.push(o.clone());
      true
    } else {
      false
    }
  };
  match cond {
    Expr::Identifier(t) if t == "True" || t == "False" => true,
    Expr::Comparison {
      operands,
      operators,
    } => {
      operators.iter().all(|op| {
        matches!(
          op,
          ComparisonOp::Less
            | ComparisonOp::LessEqual
            | ComparisonOp::Greater
            | ComparisonOp::GreaterEqual
            | ComparisonOp::Equal
            | ComparisonOp::NotEqual
        )
      }) && operands.iter().all(&mut operand)
    }
    Expr::FunctionCall { name, args } if name == "Inequality" => args
      .iter()
      .enumerate()
      .all(|(i, a)| i % 2 == 1 || operand(a)),
    Expr::FunctionCall { name, args }
      if matches!(name.as_str(), "And" | "Or" | "Not") =>
    {
      args.iter().all(|a| condition_bounds(a, var, bounds))
    }
    _ => false,
  }
}

/// A Piecewise's cases and default.
fn piecewise_parts(pw: &Expr) -> Option<(Vec<(Expr, Expr)>, Expr)> {
  let Expr::FunctionCall { args, .. } = pw else {
    return None;
  };
  let Expr::List(cases) = &args[0] else {
    return None;
  };
  let mut out = Vec::with_capacity(cases.len());
  for c in cases {
    let Expr::List(pair) = c else { return None };
    if pair.len() != 2 {
      return None;
    }
    out.push((pair[0].clone(), pair[1].clone()));
  }
  let default = args.get(1).cloned().unwrap_or(Expr::Integer(0));
  Some((out, default))
}

/// Replace each Piecewise in `e` (structurally equal to one of `pws`) by
/// the value it takes at the sample point.
fn select(e: &Expr, choices: &[(Expr, Expr)]) -> Expr {
  if is_piecewise(e) {
    for (pw, value) in choices {
      if crate::evaluator::pattern_matching::expr_equal(pw, e) {
        return value.clone();
      }
    }
  }
  match e {
    Expr::FunctionCall { name, args }
      if matches!(name.as_str(), "Plus" | "Times" | "Power") =>
    {
      call(name, args.iter().map(|a| select(a, choices)).collect())
    }
    Expr::BinaryOp { op, left, right } => Expr::BinaryOp {
      op: *op,
      left: Box::new(select(left, choices)),
      right: Box::new(select(right, choices)),
    },
    Expr::UnaryOp { op, operand } => Expr::UnaryOp {
      op: *op,
      operand: Box::new(select(operand, choices)),
    },
    _ => e.clone(),
  }
}

/// One stretch of the real line: a single point, or the open interval
/// between two bounds (`None` for an infinite end).
#[derive(Clone)]
enum Region {
  Point(Expr),
  Open(Option<Expr>, Option<Expr>),
}

/// `expr` with its interval Piecewise parts folded into one Piecewise, or
/// None when it has none or a condition is not a univariate comparison.
pub fn fold_piecewise(expr: &Expr) -> Result<Option<Expr>, InterpreterError> {
  if is_piecewise(expr) {
    return Ok(None);
  }
  let mut pws = Vec::new();
  if !collect_piecewise(expr, &mut pws) || pws.is_empty() {
    return Ok(None);
  }
  let mut var: Option<String> = None;
  let mut bounds: Vec<Expr> = Vec::new();
  let mut parts = Vec::with_capacity(pws.len());
  for pw in &pws {
    let Some((cases, default)) = piecewise_parts(pw) else {
      return Ok(None);
    };
    for (_, cond) in &cases {
      if !condition_bounds(cond, &mut var, &mut bounds) {
        return Ok(None);
      }
    }
    parts.push(((*pw).clone(), cases, default));
  }
  let Some(var) = var else {
    return Ok(None);
  };
  let x = Expr::Identifier(var);
  // Sorted distinct bounds.
  let mut keyed: Vec<(f64, Expr)> = Vec::new();
  for b in bounds {
    let Some(v) = crate::functions::math_ast::try_eval_to_f64(&b) else {
      return Ok(None);
    };
    if !keyed.iter().any(|(k, _)| *k == v) {
      keyed.push((v, b));
    }
  }
  keyed.sort_by(|a, b| a.0.total_cmp(&b.0));
  if keyed.is_empty() || keyed.len() > 64 {
    return Ok(None);
  }
  let mut regions: Vec<Region> = Vec::new();
  regions.push(Region::Open(None, Some(keyed[0].1.clone())));
  for i in 0..keyed.len() {
    regions.push(Region::Point(keyed[i].1.clone()));
    let next = keyed.get(i + 1).map(|(_, b)| b.clone());
    regions.push(Region::Open(Some(keyed[i].1.clone()), next));
  }
  let sample = |r: &Region| -> Result<Expr, InterpreterError> {
    let one = Expr::Integer(1);
    eval(&match r {
      Region::Point(p) => p.clone(),
      Region::Open(None, Some(b)) => call(
        "Plus",
        vec![b.clone(), call("Times", vec![Expr::Integer(-1), one])],
      ),
      Region::Open(Some(a), None) => call("Plus", vec![a.clone(), one]),
      Region::Open(Some(a), Some(b)) => call(
        "Times",
        vec![
          call("Rational", vec![Expr::Integer(1), Expr::Integer(2)]),
          call("Plus", vec![a.clone(), b.clone()]),
        ],
      ),
      Region::Open(None, None) => one,
    })
  };
  let substitute = |e: &Expr, at: &Expr| {
    crate::functions::polynomial_ast::solve::substitute_expr(e, &x, at)
  };
  let mut values: Vec<Expr> = Vec::with_capacity(regions.len());
  for r in &regions {
    let at = sample(r)?;
    let mut choices = Vec::with_capacity(parts.len());
    for (pw, cases, default) in &parts {
      let mut chosen = default.clone();
      for (value, cond) in cases {
        let truth = eval(&substitute(cond, &at))?;
        match &truth {
          Expr::Identifier(t) if t == "True" => {
            chosen = value.clone();
            break;
          }
          Expr::Identifier(t) if t == "False" => {}
          _ => return Ok(None),
        }
      }
      choices.push((pw.clone(), chosen));
    }
    values.push(eval(&select(expr, &choices))?);
  }
  // Merge runs of equal values.
  let same =
    |a: &Expr, b: &Expr| crate::evaluator::pattern_matching::expr_equal(a, b);
  let mut runs: Vec<(usize, usize)> = Vec::new();
  for i in 0..values.len() {
    match runs.last_mut() {
      Some((_, end)) if same(&values[*end], &values[i]) => *end = i,
      _ => runs.push((i, i)),
    }
  }
  // The default is the expression with every Piecewise at its default.
  let defaults: Vec<(Expr, Expr)> = parts
    .iter()
    .map(|(pw, _, default)| (pw.clone(), default.clone()))
    .collect();
  let default = eval(&select(expr, &defaults))?;
  let le = |a: &Expr, op: ComparisonOp, b: &Expr| Expr::Comparison {
    operands: vec![a.clone(), b.clone()],
    operators: vec![op],
  };
  let mut cases = Vec::new();
  for (start, end) in runs {
    if same(&values[start], &default) {
      continue;
    }
    let lower = match &regions[start] {
      Region::Point(p) => Some((p.clone(), true)),
      Region::Open(Some(a), _) => Some((a.clone(), false)),
      Region::Open(None, _) => None,
    };
    let upper = match &regions[end] {
      Region::Point(p) => Some((p.clone(), true)),
      Region::Open(_, Some(b)) => Some((b.clone(), false)),
      Region::Open(_, None) => None,
    };
    let lop = |closed: bool| if closed { "LessEqual" } else { "Less" };
    let cond = match (lower, upper) {
      (Some((a, true)), Some(_)) if start == end => {
        le(&x, ComparisonOp::Equal, &a)
      }
      (Some((a, ac)), Some((b, bc))) => call(
        "Inequality",
        vec![
          a,
          Expr::Identifier(lop(ac).into()),
          x.clone(),
          Expr::Identifier(lop(bc).into()),
          b,
        ],
      ),
      (Some((a, ac)), None) => le(
        &x,
        if ac {
          ComparisonOp::GreaterEqual
        } else {
          ComparisonOp::Greater
        },
        &a,
      ),
      (None, Some((b, bc))) => le(
        &x,
        if bc {
          ComparisonOp::LessEqual
        } else {
          ComparisonOp::Less
        },
        &b,
      ),
      (None, None) => Expr::Identifier("True".into()),
    };
    cases.push(Expr::List(vec![values[start].clone(), cond].into()));
  }
  if cases.is_empty() {
    return Ok(Some(default));
  }
  // wolframscript lists the cases in canonical order of their values.
  cases.sort_by(|a, b| {
    let (Expr::List(a), Expr::List(b)) = (a, b) else {
      return std::cmp::Ordering::Equal;
    };
    crate::functions::list_helpers_ast::sorting::canonical_cmp(&a[0], &b[0])
  });
  Ok(Some(eval(&call(
    "Piecewise",
    vec![Expr::List(cases.into()), default],
  ))?))
}
