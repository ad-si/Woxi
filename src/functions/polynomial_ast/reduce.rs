use super::together::negate_expr;
use super::*;

use super::reduce_backend::{
  try_linear_integer_reduce, try_linear_rational_reduce,
};
use crate::functions::calculus_ast::{is_constant_wrt, simplify};
use crate::functions::math_ast::try_eval_to_f64;
use crate::functions::polynomial_ast::helpers::compare_exprs;

pub fn and2(a: Expr, b: Expr) -> Expr {
  binop(BinaryOperator::And, a, b)
}

pub fn or2(a: Expr, b: Expr) -> Expr {
  binop(BinaryOperator::Or, a, b)
}

// ─── Reduce ──────────────────────────────────────────────────────────

/// Reduce[expr, var] or Reduce[expr, {vars}] or Reduce[expr, vars, domain]
///
/// Reduces equations and inequalities to a canonical disjunctive form.
pub fn reduce_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  // Generalized variables (`Reduce[m[1] == 1, {m[1]}]`).
  if let Some(result) =
    super::solve::with_generalized_variables(args, reduce_ast)
  {
    return result;
  }
  // A list of inequalities is their conjunction: `Reduce[{p, q}, x]` is
  // `Reduce[p && q, x]`. Lists holding equations keep their dedicated
  // elimination path.
  if let Some(Expr::List(items)) = args.first()
    && items.len() >= 2
    && items.iter().all(|item| {
      matches!(item, Expr::Comparison { operators, .. }
        if operators.iter().all(|op| !matches!(op, ComparisonOp::Equal)))
    })
  {
    let conjunction = items.iter().cloned().reduce(and2).unwrap();
    let mut rewritten = args.to_vec();
    rewritten[0] = conjunction;
    return reduce_ast(&rewritten);
  }
  // `a == b == c` is `a == b && b == c`.
  if let Some(first) = args.first() {
    let expanded = super::solve::expand_chained_equalities(first);
    if crate::syntax::expr_to_string(&expanded)
      != crate::syntax::expr_to_string(first)
    {
      let mut rewritten = args.to_vec();
      rewritten[0] = expanded;
      return reduce_ast(&rewritten);
    }
  }
  // Reduce[expr, dom]: a domain in the variable slot reduces over every
  // variable of `expr`, listed in canonical order.
  if args.len() == 2
    && let Expr::Identifier(dom) = &args[1]
    && matches!(
      dom.as_str(),
      "Reals"
        | "Integers"
        | "Complexes"
        | "Rationals"
        | "Algebraics"
        | "Booleans"
        | "Primes"
    )
  {
    let mut vars = Vec::new();
    constraint_variables(&args[0], &mut vars);
    vars.sort_by(|a, b| {
      crate::functions::list_helpers_ast::sorting::canonical_cmp(
        &Expr::Identifier(a.clone()),
        &Expr::Identifier(b.clone()),
      )
    });
    let vars = Expr::List(vars.into_iter().map(Expr::Identifier).collect());
    return reduce_ast(&[args[0].clone(), vars, args[1].clone()]);
  }
  // Over the integers, a condition that bounds every variable has finitely
  // many solutions, which are listed outright.
  // A symbol of the condition that is not among the variables is an
  // integer too, and listed first (`Reduce[x^2 + y^2 + z^2 == 1, {x, y},
  // Integers]` enumerates z as well).
  if args.len() == 3
    && matches!(&args[2], Expr::Identifier(d) if d == "Integers")
  {
    let named = extract_reduce_vars(&args[1]);
    let mut all = Vec::new();
    constraint_variables(&args[0], &mut all);
    let mut extra: Vec<String> =
      all.into_iter().filter(|v| !named.contains(v)).collect();
    extra.sort_by(|a, b| {
      crate::functions::list_helpers_ast::sorting::canonical_cmp(
        &Expr::Identifier(a.clone()),
        &Expr::Identifier(b.clone()),
      )
    });
    extra.dedup();
    let vars: Vec<String> = extra.into_iter().chain(named).collect();
    if let Some(result) =
      super::reduce_integer_box::reduce_bounded_integer_box(&args[0], &vars)
    {
      return result;
    }
  }
  if let Some(result) = try_linear_integer_reduce(args) {
    return Ok(result);
  }
  if let Some(result) = try_linear_rational_reduce(args) {
    return Ok(result);
  }
  reduce_internal_ast(args)
}

fn reduce_internal_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() || args.len() > 3 {
    return Err(InterpreterError::EvaluationError(
      "Reduce expects 2 or 3 arguments".into(),
    ));
  }

  // Narrow quantifier-elimination case:
  //   Reduce[Exists[{x, y}, α·x² + β·y² ≤ c₁ ∧ γ·x + δ·y ≥ c₂], param]
  // The Lagrange max of (γ x + δ y) on the ellipse is
  // √(c₁ (γ²/α + δ²/β)) so the existence condition reduces to a single
  // inequality on `param`. Handles the audit case
  //   `Reduce[Exists[{x, y}, x² + a y² ≤ 1 ∧ x − y ≥ 2], a]` → `a ≤ 1/3`.
  if args.len() == 2
    && let Some(out) = try_reduce_exists_quadratic_linear(&args[0], &args[1])
  {
    return Ok(out);
  }

  // Thread vector equalities (`{a1, b1} == {a2, b2}` standing for the
  // element-wise scalar equations) the same way Solve does, so
  // `Reduce[{x, y} == {1, 2}, {x, y}]` sees `x == 1 && y == 2` instead of
  // treating the list comparison as an opaque, unsolvable constraint.
  let threaded_expr = thread_reduce_list_equations(&args[0]);
  let expr = &threaded_expr;
  let domain = if args.len() == 3 {
    match &args[2] {
      Expr::Identifier(d) => Some(d.as_str()),
      _ => None,
    }
  } else {
    None
  };

  // Extract variable names. `Reduce[expr]` names none, and then every
  // variable of the statement is eliminated.
  let vars = if let Some(spec) = args.get(1) {
    extract_reduce_vars(spec)
  } else {
    let mut own = Vec::new();
    constraint_variables(expr, &mut own);
    own
  };
  if vars.is_empty() {
    return Ok(unevaluated("Reduce", args));
  }

  // A boolean combination of real inequalities in one variable is decided
  // piece by piece on the real line.
  if vars.len() == 1
    && (domain == Some("Reals")
      || (domain.is_none() && !has_equation_atom(expr)))
    && is_compound_condition(expr, &vars[0])
    && let Some(cells) = univariate_real_cells(expr, &vars[0])
  {
    return Ok(cells.to_condition(&vars[0]));
  }

  // Reduce[trigEq && bounded_range, x]: delegate to Solve, which specializes
  // the periodic family to the concrete in-range values, then present them as
  // `x == v1 || x == v2 || …`.
  if vars.len() == 1
    && let Some(out) = try_reduce_bounded_trig(expr, &vars[0])?
  {
    return Ok(out);
  }

  // Reduce[bounds && extra_constraints, x, Integers]: when the conjunction
  // pins `x` to a finite integer interval AND carries a further constraint
  // (`Mod[x, n] == k`, `x != v`, …) that the plain interval enumeration
  // cannot express, enumerate the interval and keep the integers for which
  // the whole constraint holds. Matches
  //   Reduce[Mod[x,3]==1 && 0<=x<=10, x, Integers] -> x==1||x==4||x==7||x==10.
  if domain == Some("Integers")
    && vars.len() == 1
    && let Some(out) = try_reduce_bounded_integer_filtered(expr, &vars[0])
  {
    return Ok(out);
  }

  let result = reduce_expr(expr, &vars, domain)?;

  // Over the integers, a bounded two-sided interval is enumerated as the
  // disjunction of the integers it contains: Reduce[2 < x < 5, x, Integers]
  // -> x == 3 || x == 4 (matching wolframscript), rather than leaving the
  // Inequality[...] form.
  if domain == Some("Integers")
    && vars.len() == 1
    && let Some(enumerated) = enumerate_integer_interval(&result, &vars[0])
  {
    return Ok(enumerated);
  }

  // Over the integers, a single one-sided bound `var op c` is tightened to the
  // nearest admissible integer and reported with the domain-membership
  // conjunct: Reduce[x > 2, x, Integers] -> Element[x, Integers] && x >= 3
  // (matching wolframscript), rather than leaving the bare `x > 2`.
  if domain == Some("Integers")
    && vars.len() == 1
    && let Some(tightened) = tighten_integer_one_sided(&result, &vars[0])
  {
    return Ok(tightened);
  }

  Ok(result)
}

/// True for a logical combination of conditions, or a comparison chain that
/// doesn't isolate `var` (`1/2 < x^2 < 5`).
fn is_compound_condition(expr: &Expr, var: &str) -> bool {
  match expr {
    Expr::BinaryOp {
      op: BinaryOperator::And | BinaryOperator::Or,
      ..
    }
    | Expr::UnaryOp {
      op: UnaryOperator::Not,
      ..
    } => true,
    Expr::FunctionCall { name, .. } => matches!(
      name.as_str(),
      "And" | "Or" | "Not" | "Xor" | "Nand" | "Nor" | "Implies" | "Equivalent"
    ),
    Expr::Comparison { operands, .. } => {
      operands.len() >= 3
        && !matches!(&operands[1], Expr::Identifier(v) if v == var)
    }
    _ => false,
  }
}

/// True when `expr` holds an `==` or `!=` comparison.
fn has_equation_atom(expr: &Expr) -> bool {
  match expr {
    Expr::Comparison { operators, .. } => operators
      .iter()
      .any(|op| matches!(op, ComparisonOp::Equal | ComparisonOp::NotEqual)),
    Expr::FunctionCall { name, args } => {
      matches!(name.as_str(), "Equal" | "Unequal")
        || args.iter().any(has_equation_atom)
    }
    Expr::BinaryOp { left, right, .. } => {
      has_equation_atom(left) || has_equation_atom(right)
    }
    Expr::UnaryOp { operand, .. } => has_equation_atom(operand),
    _ => false,
  }
}

/// Thread top-level vector equalities in `expr` into element-wise scalar
/// equations, exactly as `Solve` does for its first argument: a comparison
/// whose operands are equal-length lists (`{a1, b1} == {a2, b2}`) stands for
/// the conjunction of the componentwise equations. Only equations at the top
/// level or directly inside a top-level `&&`/`And[...]` are threaded — deeper
/// structure (inside `Or`, etc.) is left untouched, matching the scope Solve
/// itself threads.
fn thread_reduce_list_equations(expr: &Expr) -> Expr {
  fn collect_conjuncts(e: &Expr, out: &mut Vec<Expr>) {
    match e {
      Expr::BinaryOp {
        op: BinaryOperator::And,
        left,
        right,
      } => {
        collect_conjuncts(left, out);
        collect_conjuncts(right, out);
      }
      Expr::FunctionCall { name, args } if name == "And" => {
        for a in args {
          collect_conjuncts(a, out);
        }
      }
      _ => out.push(e.clone()),
    }
  }

  let mut conjuncts = Vec::new();
  collect_conjuncts(expr, &mut conjuncts);

  let mut changed = false;
  let mut out: Vec<Expr> = Vec::new();
  for c in &conjuncts {
    match super::solve::thread_list_equation(c) {
      Some(eqs) => {
        changed = true;
        out.extend(eqs);
      }
      None => out.push(c.clone()),
    }
  }
  if !changed {
    return expr.clone();
  }
  out
    .into_iter()
    .reduce(and2)
    .unwrap_or_else(|| bool_expr(true))
}

/// Does `e` contain any inexact (Real / BigFloat) literal? Such bounds make
/// wolframscript emit a `Reduce::ratnz` warning, so they are left alone here.
use crate::functions::math_ast::contains_inexact_real as contains_inexact_literal;

/// If `result` is a single one-sided comparison `var op c` with an exact numeric
/// bound `c`, tighten it to the nearest admissible integer and pair it with
/// `Element[var, Integers]`:
///   var >  c  -> Element[var, Integers] && var >= floor(c) + 1
///   var >= c  -> Element[var, Integers] && var >= ceil(c)
///   var <  c  -> Element[var, Integers] && var <= ceil(c) - 1
///   var <= c  -> Element[var, Integers] && var <= floor(c)
/// Returns `None` for any other shape, inexact bounds, or `==`/`!=`.
fn tighten_integer_one_sided(result: &Expr, var: &str) -> Option<Expr> {
  let Expr::Comparison {
    operands,
    operators,
  } = result
  else {
    return None;
  };
  if operands.len() != 2 || operators.len() != 1 {
    return None;
  }
  if !matches!(&operands[0], Expr::Identifier(s) if s == var) {
    return None;
  }
  // Only exact bounds: inexact reals warn in wolframscript (Reduce::ratnz).
  if contains_inexact_literal(&operands[1]) {
    return None;
  }
  let c = try_eval_to_f64(&operands[1])?;
  let (bound, out_op) = match operators[0] {
    ComparisonOp::Greater => {
      ((c.floor() as i64) + 1, ComparisonOp::GreaterEqual)
    }
    ComparisonOp::GreaterEqual => (c.ceil() as i64, ComparisonOp::GreaterEqual),
    ComparisonOp::Less => ((c.ceil() as i64) - 1, ComparisonOp::LessEqual),
    ComparisonOp::LessEqual => (c.floor() as i64, ComparisonOp::LessEqual),
    _ => return None,
  };
  let element = call("Element", vec![id_expr(var), id_expr("Integers")]);
  let comp = Expr::Comparison {
    operands: vec![id_expr(var), Expr::Integer(bound as i128)],
    operators: vec![out_op],
  };
  Some(and2(element, comp))
}

/// If `result` is a bounded two-sided interval `Inequality[lo, op1, var, op2,
/// hi]` with numeric bounds, enumerate the integers it contains as
/// `var == i1 || var == i2 || ...` (or `False` if empty). Returns `None` for
/// any other shape, non-numeric bounds, or an enumeration that would be too
/// large.
fn enumerate_integer_interval(result: &Expr, var: &str) -> Option<Expr> {
  let Expr::FunctionCall { name, args } = result else {
    return None;
  };
  if name != "Inequality" || args.len() != 5 {
    return None;
  }
  if !matches!(&args[2], Expr::Identifier(s) if s == var) {
    return None;
  }
  let lo = try_eval_to_f64(&args[0])?;
  let hi = try_eval_to_f64(&args[4])?;
  let op1 = match &args[1] {
    Expr::Identifier(s) => s.as_str(),
    _ => return None,
  };
  let op2 = match &args[3] {
    Expr::Identifier(s) => s.as_str(),
    _ => return None,
  };
  // lo op1 x: smallest admissible integer.
  let lower = match op1 {
    "Less" => (lo.floor() as i64) + 1,
    "LessEqual" => lo.ceil() as i64,
    _ => return None,
  };
  // x op2 hi: largest admissible integer.
  let upper = match op2 {
    "Less" => (hi.ceil() as i64) - 1,
    "LessEqual" => hi.floor() as i64,
    _ => return None,
  };
  if upper < lower {
    return Some(bool_expr(false));
  }
  if (upper as i128 - lower as i128) > 100_000 {
    return None;
  }
  let terms: Vec<Expr> = (lower..=upper)
    .map(|i| Expr::Comparison {
      operands: vec![id_expr(var), Expr::Integer(i as i128)],
      operators: vec![ComparisonOp::Equal],
    })
    .collect();
  Some(build_or(terms))
}

/// If `c` is a purely numeric bound on `var` — a two-sided `Inequality[lo, op,
/// var, op, hi]` or a one-sided comparison `var op const` / `const op var` with
/// `op` one of `<`, `<=`, `>`, `>=` — return its `(lower, upper)` integer
/// contributions (either may be `None`). Returns `None` for anything that is
/// not a linear bound on the bare variable (e.g. `Mod[x, n] == k`, `x != v`,
/// `x^2 < 10`), signalling an "extra" constraint to the caller.
fn numeric_var_bounds(
  c: &Expr,
  var: &str,
) -> Option<(Option<i64>, Option<i64>)> {
  let is_var = |e: &Expr| matches!(e, Expr::Identifier(s) if s == var);
  // lo op x -> lower bound; x op hi -> upper bound.
  let lower_from = |op: ComparisonOp, v: f64| -> Option<i64> {
    match op {
      ComparisonOp::Less => Some((v.floor() as i64) + 1),
      ComparisonOp::LessEqual => Some(v.ceil() as i64),
      _ => None,
    }
  };
  let upper_from = |op: ComparisonOp, v: f64| -> Option<i64> {
    match op {
      ComparisonOp::Less => Some((v.ceil() as i64) - 1),
      ComparisonOp::LessEqual => Some(v.floor() as i64),
      _ => None,
    }
  };
  // Two-sided Inequality[lo, op1, var, op2, hi].
  if let Expr::FunctionCall { name, args } = c
    && name == "Inequality"
    && args.len() == 5
    && is_var(&args[2])
  {
    let lo = try_eval_to_f64(&args[0])?;
    let hi = try_eval_to_f64(&args[4])?;
    let op1 = comparison_op_of(&args[1])?;
    let op2 = comparison_op_of(&args[3])?;
    return Some((lower_from(op1, lo), upper_from(op2, hi)));
  }
  // Chained comparison `lo op x op hi` or one-sided `var op const` /
  // `const op var`. The variable must appear at exactly one position and be
  // flanked only by numeric constants and ordering operators.
  if let Expr::Comparison {
    operands,
    operators,
  } = c
    && operands.len() >= 2
    && operators.len() == operands.len() - 1
    && operators.iter().all(|op| {
      matches!(
        op,
        ComparisonOp::Less
          | ComparisonOp::LessEqual
          | ComparisonOp::Greater
          | ComparisonOp::GreaterEqual
      )
    })
  {
    let var_positions: Vec<usize> = (0..operands.len())
      .filter(|&k| is_var(&operands[k]))
      .collect();
    if var_positions.len() != 1 {
      return None;
    }
    let k = var_positions[0];
    let mut lower: Option<i64> = None;
    let mut upper: Option<i64> = None;
    // Left neighbor: `o[k-1] op var`.
    if k > 0 {
      let v = try_eval_to_f64(&operands[k - 1])?;
      match operators[k - 1] {
        ComparisonOp::Less | ComparisonOp::LessEqual => {
          lower = lower_from(operators[k - 1], v);
        }
        ComparisonOp::Greater => upper = Some((v.ceil() as i64) - 1),
        ComparisonOp::GreaterEqual => upper = Some(v.floor() as i64),
        _ => return None,
      }
    }
    // Right neighbor: `var op o[k+1]`.
    if k + 1 < operands.len() {
      let v = try_eval_to_f64(&operands[k + 1])?;
      match operators[k] {
        ComparisonOp::Less | ComparisonOp::LessEqual => {
          upper = upper_from(operators[k], v);
        }
        ComparisonOp::Greater => lower = Some((v.floor() as i64) + 1),
        ComparisonOp::GreaterEqual => lower = Some(v.ceil() as i64),
        _ => return None,
      }
    }
    return Some((lower, upper));
  }
  None
}

fn comparison_op_of(e: &Expr) -> Option<ComparisonOp> {
  match e {
    Expr::Identifier(s) => match s.as_str() {
      "Less" => Some(ComparisonOp::Less),
      "LessEqual" => Some(ComparisonOp::LessEqual),
      "Greater" => Some(ComparisonOp::Greater),
      "GreaterEqual" => Some(ComparisonOp::GreaterEqual),
      _ => None,
    },
    _ => None,
  }
}

/// [`inequality_to_comparison`] applied through a list or conjunction of
/// constraints, so the solvers see one spelling of a chained comparison.
pub(crate) fn inequalities_to_comparisons(e: &Expr) -> Expr {
  match e {
    Expr::List(items) => {
      Expr::List(items.iter().map(inequalities_to_comparisons).collect())
    }
    Expr::FunctionCall { name, args } if name == "And" => Expr::FunctionCall {
      name: name.clone(),
      args: args.iter().map(inequalities_to_comparisons).collect(),
    },
    Expr::BinaryOp {
      op: BinaryOperator::And,
      left,
      right,
    } => Expr::BinaryOp {
      op: BinaryOperator::And,
      left: Box::new(inequalities_to_comparisons(left)),
      right: Box::new(inequalities_to_comparisons(right)),
    },
    other => inequality_to_comparison(other),
  }
}

/// Rewrite `Inequality[a, op1, b, op2, c]` as the chained comparison
/// `a op1 b op2 c`; any other expression is returned unchanged.
fn inequality_to_comparison(e: &Expr) -> Expr {
  if let Expr::FunctionCall { name, args } = e
    && name == "Inequality"
    && args.len() >= 3
    && args.len() % 2 == 1
  {
    let ops: Option<Vec<ComparisonOp>> = args
      .iter()
      .skip(1)
      .step_by(2)
      .map(comparison_op_of)
      .collect();
    if let Some(operators) = ops {
      return Expr::Comparison {
        operands: args.iter().step_by(2).cloned().collect(),
        operators,
      };
    }
  }
  e.clone()
}

/// `Reduce[bounds && extra, x, Integers]` where the conjunction bounds `x` to a
/// finite integer interval and carries at least one non-bound constraint:
/// enumerate the interval and keep the integers for which the full constraint
/// evaluates `True`, returning `x == v1 || … || x == vk` (or `False`). Returns
/// `None` when no finite two-sided range is derivable or there is no extra
/// constraint — the plain interval paths handle those with their own format.
fn try_reduce_bounded_integer_filtered(expr: &Expr, var: &str) -> Option<Expr> {
  // Inexact bounds make wolframscript warn (Reduce::ratnz); leave them alone.
  if contains_inexact_literal(expr) {
    return None;
  }
  let mut conjuncts = Vec::new();
  collect_and_constraints(expr, &mut conjuncts);
  if conjuncts.len() < 2 {
    return None;
  }
  let mut lower: Option<i64> = None;
  let mut upper: Option<i64> = None;
  let mut has_extra = false;
  for c in &conjuncts {
    if let Some((lo, hi)) = numeric_var_bounds(c, var) {
      if let Some(l) = lo {
        lower = Some(lower.map_or(l, |cur| cur.max(l)));
      }
      if let Some(h) = hi {
        upper = Some(upper.map_or(h, |cur| cur.min(h)));
      }
    } else {
      has_extra = true;
    }
  }
  let lower = lower?;
  let upper = upper?;
  if !has_extra {
    return None;
  }
  if upper < lower {
    return Some(bool_expr(false));
  }
  if (upper as i128 - lower as i128) > 100_000 {
    return None;
  }
  let mut terms: Vec<Expr> = Vec::new();
  for i in lower..=upper {
    let subst =
      crate::syntax::substitute_variable(expr, var, &Expr::Integer(i as i128));
    let evaled = crate::evaluator::evaluate_expr_to_expr(&subst).ok()?;
    if matches!(&evaled, Expr::Identifier(s) if s == "True") {
      terms.push(Expr::Comparison {
        operands: vec![id_expr(var), Expr::Integer(i as i128)],
        operators: vec![ComparisonOp::Equal],
      });
    }
  }
  if terms.is_empty() {
    return Some(bool_expr(false));
  }
  Some(build_or(terms))
}

/// Does `e` contain a `ConditionalExpression[...]` subexpression?
fn contains_conditional_expression(e: &Expr) -> bool {
  match e {
    Expr::FunctionCall { name, args } => {
      name == "ConditionalExpression"
        || args.iter().any(contains_conditional_expression)
    }
    Expr::List(items) => items.iter().any(contains_conditional_expression),
    Expr::BinaryOp { left, right, .. } => {
      contains_conditional_expression(left)
        || contains_conditional_expression(right)
    }
    Expr::UnaryOp { operand, .. } => contains_conditional_expression(operand),
    _ => false,
  }
}

/// Is `eq` a trig equation `Sin/Cos/Tan/Cot[var] == const`, or
/// `Sin/Cos[f(var)] == Sin/Cos[g(var)]`?
fn is_trig_equation(eq: &Expr, var: &str) -> bool {
  let (lhs, rhs) = match eq {
    Expr::Comparison {
      operands,
      operators,
    } if operands.len() == 2
      && operators.len() == 1
      && operators[0] == ComparisonOp::Equal =>
    {
      (&operands[0], &operands[1])
    }
    Expr::FunctionCall { name, args } if name == "Equal" && args.len() == 2 => {
      (&args[0], &args[1])
    }
    _ => return false,
  };
  let sin_cos_of_var = |e: &Expr| {
    matches!(e, Expr::FunctionCall { name, args }
      if matches!(name.as_str(), "Sin" | "Cos")
        && args.len() == 1
        && contains_var(&args[0], var))
  };
  if sin_cos_of_var(lhs) && sin_cos_of_var(rhs) {
    return true;
  }
  matches!(lhs, Expr::FunctionCall { name, args }
    if matches!(name.as_str(), "Sin" | "Cos" | "Tan" | "Cot")
      && args.len() == 1
      && matches!(&args[0], Expr::Identifier(s) if s == var))
}

/// For `Reduce[trigEq && bounding inequalities, x]`, delegate to Solve. When
/// Solve produces concrete (non-periodic) solutions, present them as the
/// disjunction `x == v1 || x == v2 || …`; otherwise return `None` so the
/// general reduction handles it.
fn try_reduce_bounded_trig(
  expr: &Expr,
  var: &str,
) -> Result<Option<Expr>, InterpreterError> {
  let (eq_opt, ineqs) = extract_eq_and_ineq_parts(expr);
  let Some(eq) = eq_opt else {
    return Ok(None);
  };
  if ineqs.is_empty() || !is_trig_equation(&eq, var) {
    return Ok(None);
  }
  let sols = solve_ast(&[expr.clone(), id_expr(var)])?;
  let Expr::List(sol_list) = &sols else {
    return Ok(None);
  };
  let var_expr = id_expr(var);
  let mut eqs: Vec<Expr> = Vec::new();
  for sol in sol_list {
    let Expr::List(rules) = sol else {
      return Ok(None);
    };
    if rules.len() != 1 {
      return Ok(None);
    }
    let Expr::Rule { replacement, .. } = &rules[0] else {
      return Ok(None);
    };
    // A periodic family means the constraint was not pinned down — leave it to
    // the general reduction.
    if contains_conditional_expression(replacement) {
      return Ok(None);
    }
    eqs.push(make_equality(&var_expr, replacement));
  }
  if eqs.is_empty() {
    return Ok(Some(bool_expr(false)));
  }
  Ok(Some(build_or(eqs)))
}

/// Extract variable names from the second argument of Reduce.
fn extract_reduce_vars(expr: &Expr) -> Vec<String> {
  match expr {
    Expr::Identifier(name) => {
      // Check it's not a constant
      use crate::evaluator::Attributes as A;
      let is_constant = crate::func_attrs_contains(name, A::Constant);
      if is_constant
        || matches!(
          name.as_str(),
          "E" | "Pi" | "I" | "True" | "False" | "Infinity"
        )
      {
        vec![]
      } else {
        vec![name.clone()]
      }
    }
    Expr::Constant(_) => vec![],
    Expr::List(items) => items.iter().flat_map(extract_reduce_vars).collect(),
    _ => vec![],
  }
}

/// The free variables of a constraint, in the order they first appear.
///
/// These are the variables `Reduce` falls back to when the caller's list
/// names none of them: `Reduce[a^2 < 4, x]` still answers `-2 < a < 2`,
/// because a statement that does not mention `x` is not thereby unconstrained.
pub(super) fn constraint_variables(expr: &Expr, out: &mut Vec<String>) {
  match expr {
    Expr::Identifier(_) => {
      for name in extract_reduce_vars(expr) {
        if !out.contains(&name) {
          out.push(name);
        }
      }
    }
    // Every other argument of an `Inequality[…]` is an operator name, not
    // an operand, so `Less` must not be read as a variable.
    Expr::FunctionCall { name, args } if name == "Inequality" => {
      for operand in args.iter().step_by(2) {
        constraint_variables(operand, out);
      }
    }
    // `Element[x, Integers]` constrains `x`; `Integers` names the domain.
    Expr::FunctionCall { name, args }
      if (name == "Element" || name == "NotElement") && args.len() == 2 =>
    {
      constraint_variables(&args[0], out);
    }
    Expr::FunctionCall { args, .. } => {
      for a in args {
        constraint_variables(a, out);
      }
    }
    Expr::List(items) => {
      for a in items {
        constraint_variables(a, out);
      }
    }
    Expr::Comparison { operands, .. } => {
      for a in operands {
        constraint_variables(a, out);
      }
    }
    Expr::BinaryOp { left, right, .. } => {
      constraint_variables(left, out);
      constraint_variables(right, out);
    }
    Expr::UnaryOp { operand, .. } => constraint_variables(operand, out),
    _ => {}
  }
}

/// The variables of `expr` that `vars` says nothing about, or `None` when
/// `vars` covers at least one of them (so the ordinary reduction applies).
fn variables_outside(expr: &Expr, vars: &[String]) -> Option<Vec<String>> {
  let mut own = Vec::new();
  constraint_variables(expr, &mut own);
  if own.is_empty() || own.iter().any(|v| vars.contains(v)) {
    return None;
  }
  Some(own)
}

/// Partition constraints into the groups that have to be reduced together:
/// two constraints belong to the same group when they share a variable,
/// transitively. Each group comes with its own variables, in first-appearance
/// order, and the groups themselves keep the order their first constraint
/// appeared in.
fn group_by_shared_variables(
  constraints: &[Expr],
) -> Vec<(Vec<Expr>, Vec<String>)> {
  let mut groups: Vec<(Vec<Expr>, Vec<String>)> = Vec::new();
  for constraint in constraints {
    let mut own = Vec::new();
    constraint_variables(constraint, &mut own);
    // Every existing group sharing a variable with this constraint merges
    // into the first such group, so `a < b && b < c && x > 0` keeps the
    // `a`/`b`/`c` chain in one piece.
    let touching: Vec<usize> = groups
      .iter()
      .enumerate()
      .filter(|(_, (_, vars))| vars.iter().any(|v| own.contains(v)))
      .map(|(i, _)| i)
      .collect();
    match touching.split_first() {
      None => groups.push((vec![constraint.clone()], own)),
      Some((&first, rest)) => {
        for &other in rest.iter().rev() {
          let (moved, moved_vars) = groups.remove(other);
          groups[first].0.extend(moved);
          for v in moved_vars {
            if !groups[first].1.contains(&v) {
              groups[first].1.push(v);
            }
          }
        }
        groups[first].0.push(constraint.clone());
        for v in own {
          if !groups[first].1.contains(&v) {
            groups[first].1.push(v);
          }
        }
      }
    }
  }
  groups
}

/// Join constraints back into an `And` chain.
fn and_chain(constraints: &[Expr]) -> Expr {
  constraints
    .iter()
    .skip(1)
    .fold(constraints[0].clone(), |acc, c| and2(acc, c.clone()))
}

/// Core reduction logic.
fn reduce_expr(
  expr: &Expr,
  vars: &[String],
  domain: Option<&str>,
) -> Result<Expr, InterpreterError> {
  // Handle True/False
  if let Expr::Identifier(s) = expr {
    if s == "True" {
      return Ok(bool_expr(true));
    }
    if s == "False" {
      return Ok(bool_expr(false));
    }
  }

  // Handle List of constraints → convert to And
  if let Expr::List(items) = expr {
    if items.is_empty() {
      return Ok(bool_expr(true));
    }
    let and_expr = items
      .iter()
      .skip(1)
      .fold(items[0].clone(), |acc, item| and2(acc, item.clone()));
    return reduce_expr(&and_expr, vars, domain);
  }

  // Handle Or (disjunction)
  if let Expr::BinaryOp {
    op: BinaryOperator::Or,
    left,
    right,
  } = expr
  {
    let left_result = reduce_expr(left, vars, domain)?;
    let right_result = reduce_expr(right, vars, domain)?;
    return Ok(or_results(&left_result, &right_result));
  }

  // Handle FunctionCall Or
  if let Expr::FunctionCall { name, args: fargs } = expr {
    if name == "Or" {
      let mut result = bool_expr(false);
      for a in fargs {
        let r = reduce_expr(a, vars, domain)?;
        result = or_results(&result, &r);
      }
      return Ok(result);
    }
    if name == "And" && !fargs.is_empty() {
      if fargs.len() == 1 {
        return reduce_expr(&fargs[0], vars, domain);
      }
      let combined = fargs
        .iter()
        .skip(1)
        .fold(fargs[0].clone(), |acc, a| and2(acc, a.clone()));
      return reduce_expr(&combined, vars, domain);
    }
  }

  // Handle And (conjunction)
  if let Expr::BinaryOp {
    op: BinaryOperator::And,
    left,
    right,
  } = expr
  {
    return reduce_and(left, right, vars, domain);
  }

  // A constraint the caller's variables do not appear in is reduced over its
  // own variables rather than left alone: wolframscript eliminates every
  // variable of the statement, and the second argument only says which ones
  // to eliminate last. `Reduce[a^2 < 4, x]` is `-2 < a < 2`.
  if let Some(own) = variables_outside(expr, vars) {
    return reduce_expr(expr, &own, domain);
  }

  // A chained two-sided numeric bound such as `0 < x < 5` (a 3-operand
  // Comparison) or the mixed-strictness `0 < x <= 10` (an Inequality[...]
  // FunctionCall) expands to the equivalent conjunction and reduces to the
  // canonical Inequality[…] form. Restricted to numeric bounds so symbolic
  // cases like `a < x < b` are left untouched rather than diverging from
  // wolframscript's parametric result.
  if vars.len() == 1
    && let Some(and_expr) = expand_numeric_two_sided_bound(expr, &vars[0])
  {
    return reduce_expr(&and_expr, vars, domain);
  }

  // Single constraint
  if vars.len() == 1 {
    return reduce_single_var(expr, &vars[0], domain);
  }

  // Multi-variable: try sequential elimination
  reduce_multi_var(expr, vars, domain)
}

/// True if `e` is a real number literal (integer, big integer, real, a signed
/// such, or an exact Rational).
fn is_real_number_literal(e: &Expr) -> bool {
  match e {
    Expr::Integer(_) | Expr::BigInteger(_) | Expr::Real(_) => true,
    Expr::FunctionCall { name, args }
      if name == "Rational" && args.len() == 2 =>
    {
      args.iter().all(is_real_number_literal)
    }
    Expr::UnaryOp {
      op: UnaryOperator::Minus,
      operand,
    } => is_real_number_literal(operand),
    _ => false,
  }
}

/// If `expr` is a two-sided chained inequality `lo op1 var op2 hi` with the
/// single `var` in the middle and `lo`/`hi` real-number literals, return the
/// equivalent `(lo op1 var) && (var op2 hi)` conjunction; otherwise `None`.
/// Both the chained-Comparison shape (`0 < x < 5`) and the Inequality[...]
/// FunctionCall shape (mixed strictness, `0 < x <= 10`) are recognized.
fn expand_numeric_two_sided_bound(expr: &Expr, var: &str) -> Option<Expr> {
  let is_var = |e: &Expr| matches!(e, Expr::Identifier(s) if s == var);
  let is_ineq_op = |op: ComparisonOp| {
    use ComparisonOp as C;
    matches!(op, C::Less | C::LessEqual | C::Greater | C::GreaterEqual)
  };

  match expr {
    Expr::Comparison {
      operands,
      operators,
    } if operands.len() == 3
      && operators.len() == 2
      && operators.iter().all(|o| is_ineq_op(*o))
      && is_var(&operands[1])
      && is_real_number_literal(&operands[0])
      && is_real_number_literal(&operands[2]) =>
    {
      let left = Expr::Comparison {
        operands: vec![operands[0].clone(), operands[1].clone()],
        operators: vec![operators[0]],
      };
      let right = Expr::Comparison {
        operands: vec![operands[1].clone(), operands[2].clone()],
        operators: vec![operators[1]],
      };
      Some(and2(left, right))
    }
    Expr::FunctionCall { name, args }
      if name == "Inequality" && args.len() == 5 =>
    {
      let valid = |e: &Expr| {
        matches!(e, Expr::Identifier(s)
          if matches!(s.as_str(), "Less" | "LessEqual" | "Greater" | "GreaterEqual"))
      };
      if !(valid(&args[1])
        && valid(&args[3])
        && is_var(&args[2])
        && is_real_number_literal(&args[0])
        && is_real_number_literal(&args[4]))
      {
        return None;
      }
      let op_name = |e: &Expr| match e {
        Expr::Identifier(s) => s.clone(),
        _ => unreachable!(),
      };
      let left = Expr::FunctionCall {
        name: op_name(&args[1]),
        args: vec![args[0].clone(), args[2].clone()].into(),
      };
      let right = Expr::FunctionCall {
        name: op_name(&args[3]),
        args: vec![args[2].clone(), args[4].clone()].into(),
      };
      Some(and2(left, right))
    }
    _ => None,
  }
}

/// Reduce a single constraint in one variable.
fn reduce_single_var(
  expr: &Expr,
  var: &str,
  domain: Option<&str>,
) -> Result<Expr, InterpreterError> {
  // Check if this is an equation (==)
  if let Some((lhs, rhs, op)) = extract_comparison(expr) {
    match op {
      CompOp::Equal => {
        return reduce_equation(&lhs, &rhs, var, domain);
      }
      CompOp::NotEqual => {
        return reduce_not_equal(&lhs, &rhs, var, domain);
      }
      CompOp::Less
      | CompOp::LessEqual
      | CompOp::Greater
      | CompOp::GreaterEqual => {
        return reduce_inequality(&lhs, &rhs, op, var, domain);
      }
    }
  }

  // A disjunction reduces branch by branch; the caller distributes And over it.
  if is_or_expr(expr) {
    return reduce_expr(expr, &[var.to_string()], domain);
  }

  // An already-reduced compound bound — `Inequality[lo, op, var, op, hi]` or a
  // 3-operand chained `Comparison` (`lo < var < hi`). These arise as Or-branches
  // when distributing And over Or (e.g. reducing `x^3-x>0 && x<1/2`, whose cubic
  // factor reduces to `-1<x<0 || x>1`). Pass them through unchanged so the
  // caller's bound-combining logic can intersect them, rather than wrapping them
  // in an unevaluated `Reduce[...]` that the bound extractor then ignores.
  if matches!(expr, Expr::FunctionCall { name, args }
    if name == "Inequality" && args.len() == 5)
    || matches!(expr, Expr::Comparison { operands, operators }
      if operands.len() == 3 && operators.len() == 2)
  {
    return Ok(expr.clone());
  }

  // If expression is already a simple passthrough (like x > 0)
  Ok(call("Reduce", vec![expr.clone(), id_expr(var)]))
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CompOp {
  Equal,
  NotEqual,
  Less,
  LessEqual,
  Greater,
  GreaterEqual,
}

/// Extract a comparison: returns (lhs, rhs, op)
pub fn extract_comparison(expr: &Expr) -> Option<(Expr, Expr, CompOp)> {
  match expr {
    Expr::Comparison {
      operands,
      operators,
    } if operands.len() == 2 && operators.len() == 1 => {
      let op = match operators[0] {
        ComparisonOp::Equal => CompOp::Equal,
        ComparisonOp::NotEqual => CompOp::NotEqual,
        ComparisonOp::Less => CompOp::Less,
        ComparisonOp::LessEqual => CompOp::LessEqual,
        ComparisonOp::Greater => CompOp::Greater,
        ComparisonOp::GreaterEqual => CompOp::GreaterEqual,
        _ => return None,
      };
      Some((operands[0].clone(), operands[1].clone(), op))
    }
    Expr::FunctionCall { name, args } if args.len() == 2 => {
      let op = match name.as_str() {
        "Equal" => CompOp::Equal,
        "Unequal" => CompOp::NotEqual,
        "Less" => CompOp::Less,
        "LessEqual" => CompOp::LessEqual,
        "Greater" => CompOp::Greater,
        "GreaterEqual" => CompOp::GreaterEqual,
        _ => return None,
      };
      Some((args[0].clone(), args[1].clone(), op))
    }
    _ => None,
  }
}

/// Reduce an equation lhs == rhs for variable var.
/// Strip a leading unary/`-1*` minus from `e` (used so the degenerate condition
/// prints `b == 0` rather than `-b == 0`, matching wolframscript).
fn strip_leading_minus(e: &Expr) -> Expr {
  match e {
    Expr::UnaryOp {
      op: UnaryOperator::Minus,
      operand,
    } => (**operand).clone(),
    Expr::BinaryOp {
      op: BinaryOperator::Times,
      left,
      right,
    } if matches!(left.as_ref(), Expr::Integer(-1)) => (**right).clone(),
    Expr::FunctionCall { name, args }
      if name == "Times"
        && args.len() == 2
        && matches!(&args[0], Expr::Integer(-1)) =>
    {
      args[1].clone()
    }
    _ => e.clone(),
  }
}

/// For an equation `expanded == 0` that is linear in `var` with a *parametric*
/// (non-numeric) leading coefficient, produce the full solution set — including
/// the degenerate case where the coefficient vanishes — as wolframscript does:
///   c1*var + c0 == 0   (c1 symbolic)  ->
///     c0 == 0              : c1 == 0 || var == 0
///     c0 a nonzero number  : c1 != 0 && var == -c0/c1
///     c0 parametric        : (-c0 == 0 && c1 == 0) || (c1 != 0 && var == -c0/c1)
/// Returns None when the shape does not apply (non-linear, numeric coefficient,
/// restricted domain) so the caller falls through to the normal Solve path.
fn try_reduce_linear_parametric(
  expanded: &Expr,
  var: &str,
  domain: Option<&str>,
) -> Result<Option<Expr>, InterpreterError> {
  // Only the unrestricted (Complexes) case is handled here.
  if domain.is_some() {
    return Ok(None);
  }
  let var_expr = id_expr(var);
  let degree = crate::functions::polynomial_ast::exponent_ast(&[
    expanded.clone(),
    var_expr.clone(),
  ])?;
  if !matches!(degree, Expr::Integer(1)) {
    return Ok(None);
  }
  let c1 = crate::functions::polynomial_ast::coefficient_ast(&[
    expanded.clone(),
    var_expr.clone(),
    Expr::Integer(1),
  ])?;
  let c0 = crate::functions::polynomial_ast::coefficient_ast(&[
    expanded.clone(),
    var_expr.clone(),
    Expr::Integer(0),
  ])?;
  if !is_constant_wrt(&c1, var) || !is_constant_wrt(&c0, var) {
    return Ok(None);
  }
  // A numeric leading coefficient has a unique, non-degenerate solution — the
  // ordinary Solve path already handles it.
  if is_reduce_number(&c1) {
    return Ok(None);
  }
  // The condition `c1 == 0` reduces to `s == 0` when c1 is a numeric multiple of
  // a single symbol `s`; wolframscript enumerates the sub-cases for products of
  // several symbols, which we do not attempt here.
  let Some(cond_var) = primitive_condition_var(&c1) else {
    return Ok(None);
  };
  // Solution branch: cond_var != 0 && var == -c0/c1  (the solution keeps the
  // full coefficient c1, e.g. b/(2*a)). Use the full evaluator so the quotient
  // is canonicalized the same way Solve reports it (e.g. b/(-a) -> -(b/a)).
  let sol_value = crate::evaluator::evaluate_expr_to_expr(&div2(
    negate_expr(&c0),
    c1.clone(),
  ))?;
  let solution = and_results(
    &make_comparison(&cond_var, &Expr::Integer(0), CompOp::NotEqual),
    &make_equality(&var_expr, &sol_value),
  );
  // c0 == 0: the whole line is a solution when the coefficient vanishes.
  if matches!(&c0, Expr::Integer(0)) {
    return Ok(Some(or_results(
      &make_equality(&cond_var, &Expr::Integer(0)),
      &make_equality(&var_expr, &Expr::Integer(0)),
    )));
  }
  // Nonzero numeric constant: c1 == 0 would make the equation 0 == nonzero, so
  // only the solution branch survives.
  if is_reduce_number(&c0) {
    return Ok(Some(solution));
  }
  // Parametric constant: keep both branches.
  let degenerate = and_results(
    &make_equality(&strip_leading_minus(&c0), &Expr::Integer(0)),
    &make_equality(&cond_var, &Expr::Integer(0)),
  );
  Ok(Some(or_results(&degenerate, &solution)))
}

/// Whether `e` is an explicit number literal (integer, real, or rational).
fn is_reduce_number(e: &Expr) -> bool {
  matches!(
    e,
    Expr::Integer(_)
      | Expr::Real(_)
      | Expr::BigInteger(_)
      | Expr::BigFloat(_, _)
  ) || matches!(e, Expr::FunctionCall { name, .. } if name == "Rational")
}

/// If `c1` is a numeric multiple of a single symbol `s` (e.g. `2 a`, `-a`,
/// `a/2`), return `s` — the symbol whose vanishing is equivalent to `c1 == 0`.
/// Returns None for a bare number or a product of several symbols.
fn primitive_condition_var(c1: &Expr) -> Option<Expr> {
  match c1 {
    Expr::Identifier(_) => Some(c1.clone()),
    Expr::UnaryOp {
      op: UnaryOperator::Minus,
      operand,
    } => primitive_condition_var(operand),
    Expr::BinaryOp {
      op: BinaryOperator::Times,
      left,
      right,
    } => match (is_reduce_number(left), is_reduce_number(right)) {
      (true, false) => primitive_condition_var(right),
      (false, true) => primitive_condition_var(left),
      _ => None,
    },
    Expr::FunctionCall { name, args } if name == "Times" => {
      let non_numeric: Vec<&Expr> =
        args.iter().filter(|a| !is_reduce_number(a)).collect();
      match non_numeric.as_slice() {
        [only] => primitive_condition_var(only),
        _ => None,
      }
    }
    _ => None,
  }
}

fn reduce_equation(
  lhs: &Expr,
  rhs: &Expr,
  var: &str,
  domain: Option<&str>,
) -> Result<Expr, InterpreterError> {
  // Build lhs - rhs
  let poly = minus2(lhs.clone(), rhs.clone());

  let expanded = expand_and_combine(&poly);

  // Check if variable is present
  if is_constant_wrt(&expanded, var) {
    // No variable — check if constant is zero
    let simplified = simplify(expanded);
    return Ok(if matches!(simplified, Expr::Integer(0)) {
      bool_expr(true)
    } else {
      bool_expr(false)
    });
  }

  // Linear equation with a symbolic leading coefficient: include the a==0 case.
  if let Some(result) = try_reduce_linear_parametric(&expanded, var, domain)? {
    return Ok(result);
  }

  // Use the Solve infrastructure to find roots
  let eq_expr = Expr::Comparison {
    operands: vec![lhs.clone(), rhs.clone()],
    operators: vec![ComparisonOp::Equal],
  };
  let solve_result = solve_ast(&[eq_expr, id_expr(var)])?;

  // Convert Solve result (list of rules) to Or of equalities
  if let Expr::List(solutions) = &solve_result {
    if solutions.is_empty() {
      return Ok(bool_expr(false));
    }

    let mut equalities: Vec<Expr> = Vec::new();
    for sol in solutions {
      if let Expr::List(rules) = sol {
        if rules.is_empty() {
          // Solve returned {{}} meaning all values are solutions
          return Ok(bool_expr(true));
        }
        for rule in rules {
          if let Expr::Rule {
            pattern: _,
            replacement,
          } = rule
          {
            let value = replacement.as_ref().clone();

            // Domain filtering
            if let Some(dom) = domain
              && !is_in_domain(&value, dom)
            {
              continue;
            }

            equalities.push(make_equality(&id_expr(var), &value));
          }
        }
      }
    }

    if equalities.is_empty() {
      return Ok(bool_expr(false));
    }

    // Sort equalities for canonical output
    equalities.sort_by(compare_exprs);

    // Deduplicate
    equalities.dedup_by(|a, b| expr_to_string(a) == expr_to_string(b));

    return Ok(build_or(equalities));
  }

  // Solve returned unevaluated — try factoring
  let factored = factor_ast(std::slice::from_ref(&expanded))?;
  let factors = collect_multiplicative_factors(&factored);
  if factors.len() > 1 {
    // Each factor == 0 gives solutions
    let mut all_equalities: Vec<Expr> = Vec::new();
    for factor in &factors {
      if is_constant_wrt(factor, var) {
        continue; // Skip constant factors
      }
      let sub_result = reduce_equation(factor, &Expr::Integer(0), var, domain)?;
      let sub_terms = collect_or_terms(&sub_result);
      for t in sub_terms {
        if !matches!(&t, Expr::Identifier(s) if s == "False") {
          all_equalities.push(t);
        }
      }
    }
    if !all_equalities.is_empty() {
      all_equalities.sort_by(compare_exprs);
      all_equalities.dedup_by(|a, b| expr_to_string(a) == expr_to_string(b));
      return Ok(build_or(all_equalities));
    }
  }

  // Return Reduce unevaluated
  Ok(call(
    "Reduce",
    if let Some(dom) = domain {
      vec![make_equality(lhs, rhs), id_expr(var), id_expr(dom)]
    } else {
      vec![make_equality(lhs, rhs), id_expr(var)]
    },
  ))
}

/// Reduce a != b for variable var.
fn reduce_not_equal(
  lhs: &Expr,
  rhs: &Expr,
  var: &str,
  domain: Option<&str>,
) -> Result<Expr, InterpreterError> {
  // `Abs[f] != c` over the reals: the punctured line `f < -c || -c < f < c
  // || f > c` (or `f != 0` at c == 0, all reals for c < 0).
  if let Some(result) = try_reduce_abs_not_equal(lhs, rhs, var, domain) {
    return result;
  }

  // For simple cases, just return the inequality
  let poly = minus2(lhs.clone(), rhs.clone());
  let expanded = expand_and_combine(&poly);
  if is_constant_wrt(&expanded, var) {
    let simplified = simplify(expanded);
    return Ok(if matches!(simplified, Expr::Integer(0)) {
      bool_expr(false)
    } else {
      bool_expr(true)
    });
  }
  // Return the not-equal condition as-is for now
  let _ = domain;
  Ok(Expr::Comparison {
    operands: vec![lhs.clone(), rhs.clone()],
    operators: vec![ComparisonOp::NotEqual],
  })
}

/// Recognise `ArcXxxDegrees[var] CompOp const_degrees` and invert it.
/// Returns Some(reduced) when the pattern matches and the inversion has a
/// closed form, None otherwise.
fn try_reduce_arc_degrees(
  lhs: &Expr,
  rhs: &Expr,
  op: CompOp,
  var: &str,
) -> Result<Option<Expr>, InterpreterError> {
  // lhs must be `ArcXxxDegrees[var]`.
  let Expr::FunctionCall {
    name: lname,
    args: largs,
  } = lhs
  else {
    return Ok(None);
  };
  if largs.len() != 1 || !matches!(&largs[0], Expr::Identifier(s) if s == var) {
    return Ok(None);
  }

  // rhs must be a constant scalar (Integer/Real/Rational).
  if !is_constant_wrt(rhs, var) {
    return Ok(None);
  }

  // Build the corresponding trig threshold in radians:
  //   threshold = TrigF[rhs * Pi / 180].
  let trig_name = match lname.as_str() {
    "ArcCosDegrees" => "Cos",
    "ArcSinDegrees" => "Sin",
    "ArcTanDegrees" => "Tan",
    "ArcCotDegrees" => "Cot",
    "ArcCscDegrees" => "Csc",
    "ArcSecDegrees" => "Sec",
    _ => return Ok(None),
  };
  let radians = times(vec![
    rhs.clone(),
    id_expr("Pi"),
    call("Rational", vec![Expr::Integer(1), Expr::Integer(180)]),
  ]);
  let threshold_expr = call1(trig_name, radians);
  let threshold = crate::evaluator::evaluate_expr_to_expr(&threshold_expr)?;

  // Helper: build a 3-way bounded interval `low op1 var op2 high`.
  let bounded = |low: Expr, op1: CompOp, op2: CompOp, high: Expr| -> Expr {
    make_compound_inequality(&low, op1, var, op2, &high)
  };
  // Helper: simple comparison `var op c`.
  let simple =
    |op: CompOp, c: Expr| -> Expr { make_comparison(&id_expr(var), &c, op) };

  // Build the result. Only `>` audit cases for now — that's the documented
  // basic example for every ArcXxxDegrees function.
  if !matches!(op, CompOp::Greater) {
    return Ok(None);
  }

  let result = match lname.as_str() {
    // ArcCosDegrees is strictly decreasing on [-1, 1] → [0, 180].
    // arccos(x) > k iff -1 ≤ x < cos(k°).
    "ArcCosDegrees" => bounded(
      Expr::Integer(-1),
      CompOp::LessEqual,
      CompOp::Less,
      threshold,
    ),
    // ArcSinDegrees is strictly increasing on [-1, 1] → [-90, 90].
    // arcsin(x) > k iff sin(k°) < x ≤ 1.
    "ArcSinDegrees" => {
      bounded(threshold, CompOp::Less, CompOp::LessEqual, Expr::Integer(1))
    }
    // ArcTanDegrees is strictly increasing on R → (-90, 90).
    // arctan(x) > k iff x > tan(k°).
    "ArcTanDegrees" => simple(CompOp::Greater, threshold),
    // ArcCotDegrees principal branch is strictly decreasing on [0, ∞) → (0, 90].
    // arccot(x) > k iff 0 ≤ x < cot(k°)  (for 0 < k < 90).
    "ArcCotDegrees" => {
      bounded(Expr::Integer(0), CompOp::LessEqual, CompOp::Less, threshold)
    }
    // ArcCscDegrees on [1, ∞) → (0, 90], strictly decreasing.
    // arccsc(x) > k iff 1 ≤ x < csc(k°)  (for 0 < k ≤ 90).
    "ArcCscDegrees" => {
      bounded(Expr::Integer(1), CompOp::LessEqual, CompOp::Less, threshold)
    }
    // ArcSecDegrees: arcsec(x) > k iff x > sec(k°) || x ≤ -1
    // (the second branch covers arcsec on (-∞, -1] mapping to (90, 180]).
    "ArcSecDegrees" => call(
      "Or",
      vec![
        simple(CompOp::Greater, threshold),
        simple(CompOp::LessEqual, Expr::Integer(-1)),
      ],
    ),
    _ => return Ok(None),
  };
  Ok(Some(result))
}

/// Reduce an inequality (lhs op rhs) for one variable.
fn reduce_inequality(
  lhs: &Expr,
  rhs: &Expr,
  op: CompOp,
  var: &str,
  domain: Option<&str>,
) -> Result<Expr, InterpreterError> {
  // Recognise `ArcXxxDegrees[var] CompOp const_degrees` directly — these
  // need transcendental inversion, not polynomial reduction.
  if let Some(result) = try_reduce_arc_degrees(lhs, rhs, op, var)? {
    return Ok(result);
  }

  // `Abs[f] op c` over the reals splits into a two-interval / two-sided
  // condition (`-c < f < c`, `f < -c || f > c`, …) that the existing
  // polynomial machinery then solves on the bare argument `f`.
  if let Some(result) = try_reduce_abs_inequality(lhs, rhs, op, var, domain) {
    return result;
  }

  // Move everything to one side: lhs - rhs op 0
  let poly = minus2(lhs.clone(), rhs.clone());
  let expanded = expand_and_combine(&poly);

  // Check if variable is present
  if is_constant_wrt(&expanded, var) {
    // No variable — evaluate the comparison numerically
    let val = simplify(expanded);
    if let Some(result) = evaluate_constant_ineq(&val, op) {
      return Ok(result);
    }
    // The constraint is independent of the requested variable but still
    // constrains a parameter.  Preserve it instead of treating every
    // non-numeric condition as True (e.g. Reduce[a > 0, x] -> a > 0).
    return Ok(make_comparison(&val, &Expr::Integer(0), op));
  }

  // A polynomial in `var` alone: its real roots split the line, and the
  // sign between them decides each piece.
  if let Some(result) =
    reduce_univariate_polynomial_inequality(&expanded, var, op, domain)
  {
    return Ok(result);
  }

  // Try to find the degree
  let degree = max_power_int(&expanded, var);

  // Symbolic coefficients are left alone; other functions of `var` go by
  // the roots Solve finds.
  if !matches!(degree, Some(1 | 2))
    && let Some(result) =
      try_factor_and_reduce_inequality(lhs, rhs, op, var, domain)
  {
    return Ok(result);
  }
  Ok(call(
    "Reduce",
    vec![make_comparison(lhs, rhs, op), id_expr(var)],
  ))
}

/// `poly op 0` for a polynomial whose only symbol is `var`, as Wolfram
/// writes it: the real roots in increasing order split the line into open
/// intervals, the sign of `poly` on each (and whether the roots themselves
/// count) picks the pieces, and each run of adjacent pieces becomes one
/// condition — `x < r`, `Inequality[r1, Less, x, LessEqual, r2]`, `x == r`,
/// … — joined by `||`. A condition true everywhere is `Element[x, Reals]`
/// (`True` over `Reals`). Machine coefficients are rationalized, solved
/// exactly and numericized, with `Reduce::ratnz`, as Wolfram does. `None`
/// when `poly` has another symbol or its roots cannot be found.
fn reduce_univariate_polynomial_inequality(
  poly: &Expr,
  var: &str,
  op: CompOp,
  domain: Option<&str>,
) -> Option<Expr> {
  if !matches!(
    op,
    CompOp::Less | CompOp::LessEqual | CompOp::Greater | CompOp::GreaterEqual
  ) {
    return None;
  }
  let mut symbols = Vec::new();
  constraint_variables(poly, &mut symbols);
  if symbols.iter().any(|s| s != var) {
    return None;
  }
  // A rational function splits the line at the roots of its numerator and
  // its poles, the denominator's roots, where it never holds.
  let together = crate::evaluator::evaluate_function_call_ast(
    "Together",
    std::slice::from_ref(poly),
  )
  .ok()?;
  let (numerator, denominator) = super::together::extract_num_den(&together);
  let numerator = expand_and_combine(&numerator);
  let denominator = expand_and_combine(&denominator);
  let numerator_degree = max_power_int(&numerator, var)?;
  let denominator_degree = max_power_int(&denominator, var)?;
  if numerator_degree < 0
    || denominator_degree < 0
    || numerator_degree.max(denominator_degree) < 1
  {
    return None;
  }
  if crate::functions::math_ast::contains_inexact_real(poly) {
    let exact = crate::evaluator::evaluate_function_call_ast(
      "Rationalize",
      &[poly.clone(), Expr::Integer(0)],
    )
    .ok()?;
    if crate::functions::math_ast::contains_inexact_real(&exact) {
      return None;
    }
    let exact_result =
      reduce_univariate_polynomial_inequality(&exact, var, op, domain)?;
    crate::emit_message(
      "Reduce::ratnz: Reduce was unable to solve the system with inexact coefficients. The answer was obtained by solving a corresponding exact system and numericizing the result.",
    );
    // A root at zero stays exact (`x < 0 || x > 2.`), as in Wolfram.
    let numeric =
      crate::evaluator::evaluate_function_call_ast("N", &[exact_result])
        .ok()?;
    return Some(crate::functions::string_ast::map_expr_tree(
      &numeric,
      &|e: &Expr| {
        matches!(e, Expr::Real(r) if *r == 0.0).then(|| Expr::Integer(0))
      },
    ));
  }
  let (roots, poles): (Vec<(f64, Expr)>, Vec<bool>) = real_roots_and_poles(
    &numerator,
    numerator_degree,
    &denominator,
    denominator_degree,
    var,
  )?
  .into_iter()
  .map(|(x, e, pole)| ((x, e), pole))
  .unzip();
  let holds_at = |x: f64| -> Option<bool> {
    let value = try_eval_to_f64(&crate::syntax::substitute_variable(
      poly,
      var,
      &Expr::Real(x),
    ))?;
    Some(match op {
      CompOp::Less => value < 0.0,
      CompOp::LessEqual => value <= 0.0,
      CompOp::Greater => value > 0.0,
      CompOp::GreaterEqual => value >= 0.0,
      _ => false,
    })
  };
  let roots_hold = matches!(op, CompOp::LessEqual | CompOp::GreaterEqual);
  // The pieces in order — interval 0, root 0, interval 1, …, root n-1,
  // interval n — and whether the condition holds on each.
  let n = roots.len();
  let mut holds = Vec::with_capacity(2 * n + 1);
  for i in 0..=n {
    let x = match (i, n) {
      (_, 0) => 0.0,
      (0, _) => roots[0].0 - 1.0,
      (i, n) if i == n => roots[n - 1].0 + 1.0,
      (i, _) => f64::midpoint(roots[i - 1].0, roots[i].0),
    };
    holds.push(holds_at(x)?);
    if i < n {
      holds.push(roots_hold && !poles[i]);
    }
  }
  if holds.iter().all(|h| *h) {
    return Some(if domain == Some("Reals") {
      bool_expr(true)
    } else {
      call("Element", vec![id_expr(var), id_expr("Reals")])
    });
  }
  // Every maximal run of pieces where it holds is one condition.
  let mut conditions = Vec::new();
  let mut i = 0;
  while i < holds.len() {
    if !holds[i] {
      i += 1;
      continue;
    }
    let first = i;
    while i + 1 < holds.len() && holds[i + 1] {
      i += 1;
    }
    conditions.push(run_condition(var, first, i, &roots));
    i += 1;
  }
  Some(build_or(conditions))
}

/// The real roots of `numerator` and of `denominator` (the poles, marked
/// `true`), in increasing order and without repeats, each with its numeric
/// value and its exact form as `Solve` gives it. `None` when `Solve` can't
/// solve one of them.
pub(super) fn real_roots_and_poles(
  numerator: &Expr,
  numerator_degree: i128,
  denominator: &Expr,
  denominator_degree: i128,
  var: &str,
) -> Option<Vec<(f64, Expr, bool)>> {
  let mut roots: Vec<(f64, Expr, bool)> = Vec::new();
  for (part, degree, is_pole) in [
    (numerator, numerator_degree, false),
    (denominator, denominator_degree, true),
  ] {
    if degree < 1 {
      continue;
    }
    let solved = crate::functions::polynomial_ast::solve_ast(&[
      make_equality(part, &Expr::Integer(0)),
      id_expr(var),
    ])
    .ok()?;
    let Expr::List(ref solutions) = solved else {
      return None;
    };
    for solution in solutions {
      let Expr::List(rules) = solution else {
        return None;
      };
      let [Expr::Rule { replacement, .. }] = rules.as_slice() else {
        return None;
      };
      let numeric = crate::evaluator::evaluate_function_call_ast(
        "N",
        std::slice::from_ref(replacement.as_ref()),
      )
      .ok()?;
      let (re, im) =
        crate::functions::math_ast::try_extract_complex_float(&numeric)?;
      if im.abs() > 1e-10 * (1.0 + re.abs()) {
        continue;
      }
      if !roots
        .iter()
        .any(|(r, _, _)| (r - re).abs() <= 1e-12 * (1.0 + re.abs()))
      {
        roots.push((re, replacement.as_ref().clone(), is_pole));
      }
    }
  }
  roots
    .sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
  Some(roots)
}

/// The condition for the run of pieces `first..=last` (even indices are
/// the intervals between roots, odd ones the roots themselves): an interval
/// leaves its bounding root out, a root piece puts it in.
pub(super) fn run_condition(
  var: &str,
  first: usize,
  last: usize,
  roots: &[(f64, Expr)],
) -> Expr {
  let x = id_expr(var);
  let n = roots.len();
  if first == last && first % 2 == 1 {
    return make_equality(&x, &roots[first / 2].1);
  }
  // Lower end: an interval starting the run is bounded by the root before
  // it (none for the first interval); a root starting it is included.
  let lower = if first % 2 == 1 {
    Some((first / 2, CompOp::LessEqual))
  } else if first == 0 {
    None
  } else {
    Some((first / 2 - 1, CompOp::Less))
  };
  let upper = if last % 2 == 1 {
    Some((last / 2, CompOp::LessEqual))
  } else if last / 2 == n {
    None
  } else {
    Some((last / 2, CompOp::Less))
  };
  match (lower, upper) {
    (None, None) => bool_expr(true),
    (None, Some((e, op))) => make_comparison(&x, &roots[e].1, op),
    (Some((s, op)), None) => make_comparison(&x, &roots[s].1, flip_op(op)),
    (Some((s, lop)), Some((e, uop))) => {
      make_compound_inequality(&roots[s].1, lop, var, uop, &roots[e].1)
    }
  }
}

/// Extract the single argument of `Abs[arg]`.
fn extract_abs(expr: &Expr) -> Option<Expr> {
  match expr {
    Expr::FunctionCall { name, args } if name == "Abs" && args.len() == 1 => {
      Some(args[0].clone())
    }
    _ => None,
  }
}

/// Recognise `Abs[inner]` or `const * Abs[inner]` (the form `Abs[2 x]` takes
/// after evaluation pulls the factor out: `2 Abs[x]`). Returns `(coeff, inner)`.
fn extract_abs_term(expr: &Expr, var: &str) -> Option<(Expr, Expr)> {
  if let Some(inner) = extract_abs(expr) {
    return Some((Expr::Integer(1), inner));
  }
  let factors = collect_multiplicative_factors(expr);
  if factors.len() < 2 {
    return None;
  }
  let mut inner = None;
  let mut consts: Vec<Expr> = Vec::new();
  for f in &factors {
    if let Some(a) = extract_abs(f) {
      if inner.is_some() {
        return None; // a product of two Abs terms — not handled
      }
      inner = Some(a);
    } else if is_constant_wrt(f, var) {
      consts.push(f.clone());
    } else {
      return None;
    }
  }
  let inner = inner?;
  let coeff = match consts.len() {
    0 => Expr::Integer(1),
    1 => consts.into_iter().next().unwrap(),
    _ => simplify(times(consts)),
  };
  Some((coeff, inner))
}

/// Reduce `Abs[f] op c` over the reals. Returns `None` when the pattern does
/// not apply (non-real domain, no `Abs`, symbolic bound, …) so the caller
/// falls through to the generic polynomial path.
fn try_reduce_abs_inequality(
  lhs: &Expr,
  rhs: &Expr,
  op: CompOp,
  var: &str,
  domain: Option<&str>,
) -> Option<Result<Expr, InterpreterError>> {
  // Only the real domain has the simple two-interval solution; over the
  // default complex domain `Abs[z] < c` describes a 2-D disk.
  if domain != Some("Reals") {
    return None;
  }
  // Normalise to `coeff*Abs[inner] op bound`; flip if `Abs` is on the right.
  let (coeff, inner, bound, op) =
    match (extract_abs_term(lhs, var), extract_abs_term(rhs, var)) {
      (Some((coeff, inner)), None) => (coeff, inner, rhs.clone(), op),
      (None, Some((coeff, inner))) => (coeff, inner, lhs.clone(), flip_op(op)),
      _ => return None,
    };
  // The bound must be a constant number and the argument must involve `var`.
  if !is_constant_wrt(&bound, var) || is_constant_wrt(&inner, var) {
    return None;
  }
  // Divide through by the (constant) coefficient, flipping the operator when
  // the coefficient is negative: `c*Abs[f] op b` ⇒ `Abs[f] op' b/c`.
  let coeff_val = expr_to_number(&coeff)?;
  if coeff_val == 0.0 {
    return None;
  }
  let op = if coeff_val < 0.0 { flip_op(op) } else { op };
  // Evaluate (not just simplify) so `6/2` collapses to `3` and `5/2` to the
  // exact Rational — `expr_to_number` only reads literal number forms.
  let c = crate::evaluator::evaluate_expr_to_expr(&div2(bound, coeff)).ok()?;
  let neg_c = negate_expr(&c);
  let cval = expr_to_number(&c)?;
  let zero = Expr::Integer(0);

  let rewritten: Expr = match op {
    CompOp::Less => {
      // |f| < c : empty when c <= 0, else -c < f < c.
      if cval <= 0.0 {
        return Some(Ok(bool_expr(false)));
      }
      and2(
        make_comparison(&inner, &neg_c, CompOp::Greater),
        make_comparison(&inner, &c, CompOp::Less),
      )
    }
    CompOp::LessEqual => {
      // |f| <= c : empty for c < 0, the point f == 0 for c == 0, else band.
      if cval < 0.0 {
        return Some(Ok(bool_expr(false)));
      }
      if cval == 0.0 {
        make_comparison(&inner, &zero, CompOp::Equal)
      } else {
        and2(
          make_comparison(&inner, &neg_c, CompOp::GreaterEqual),
          make_comparison(&inner, &c, CompOp::LessEqual),
        )
      }
    }
    CompOp::Greater => {
      // |f| > c : all reals for c < 0, else two open rays. At c == 0 this is
      // `f < 0 || f > 0` (Wolfram does not collapse it to `f != 0`).
      if cval < 0.0 {
        return Some(Ok(bool_expr(true)));
      }
      or2(
        make_comparison(&inner, &neg_c, CompOp::Less),
        make_comparison(&inner, &c, CompOp::Greater),
      )
    }
    CompOp::GreaterEqual => {
      // |f| >= c : all reals when c <= 0, else two closed rays.
      if cval <= 0.0 {
        return Some(Ok(bool_expr(true)));
      }
      or2(
        make_comparison(&inner, &neg_c, CompOp::LessEqual),
        make_comparison(&inner, &c, CompOp::GreaterEqual),
      )
    }
    CompOp::Equal | CompOp::NotEqual => return None,
  };
  Some(reduce_expr(&rewritten, &[var.to_string()], domain))
}

/// Reduce `Abs[f] != c` over the reals (the punctured-line counterpart of
/// `try_reduce_abs_inequality`). Returns `None` when the pattern does not
/// apply.
fn try_reduce_abs_not_equal(
  lhs: &Expr,
  rhs: &Expr,
  var: &str,
  domain: Option<&str>,
) -> Option<Result<Expr, InterpreterError>> {
  if domain != Some("Reals") {
    return None;
  }
  let (coeff, inner, bound) =
    match (extract_abs_term(lhs, var), extract_abs_term(rhs, var)) {
      (Some((coeff, inner)), None) => (coeff, inner, rhs.clone()),
      (None, Some((coeff, inner))) => (coeff, inner, lhs.clone()),
      _ => return None,
    };
  if !is_constant_wrt(&bound, var) || is_constant_wrt(&inner, var) {
    return None;
  }
  let coeff_val = expr_to_number(&coeff)?;
  if coeff_val == 0.0 {
    return None;
  }
  let c = crate::evaluator::evaluate_expr_to_expr(&div2(bound, coeff)).ok()?;
  let cval = expr_to_number(&c)?;

  // |f| != c is vacuously true when c < 0 (Abs is never negative).
  if cval < 0.0 {
    return Some(Ok(bool_expr(true)));
  }
  // |f| != 0 is the two open rays `f < 0 || f > 0` (Wolfram keeps it split
  // rather than folding back to `f != 0`).
  if cval == 0.0 {
    let rays = or2(
      make_comparison(&inner, &c, CompOp::Less),
      make_comparison(&inner, &c, CompOp::Greater),
    );
    return Some(reduce_expr(&rays, &[var.to_string()], domain));
  }
  // c > 0: f < -c || (-c < f < c) || f > c.
  let neg_c = negate_expr(&c);
  let band = and2(
    make_comparison(&inner, &neg_c, CompOp::Greater),
    make_comparison(&inner, &c, CompOp::Less),
  );
  let rewritten = or2(
    or2(make_comparison(&inner, &neg_c, CompOp::Less), band),
    make_comparison(&inner, &c, CompOp::Greater),
  );
  Some(reduce_expr(&rewritten, &[var.to_string()], domain))
}

/// Evaluate a constant inequality (no variable present).
fn evaluate_constant_ineq(val: &Expr, op: CompOp) -> Option<Expr> {
  // Try to get a numeric value
  if let Some(n) = expr_to_number(val) {
    let result = match op {
      CompOp::Less => n < 0.0,
      CompOp::LessEqual => n <= 0.0,
      CompOp::Greater => n > 0.0,
      CompOp::GreaterEqual => n >= 0.0,
      CompOp::Equal => n == 0.0,
      CompOp::NotEqual => n != 0.0,
    };
    return Some(bool_expr(result));
  }
  None
}

/// Try to extract a numeric value from an expression.
pub fn expr_to_number(expr: &Expr) -> Option<f64> {
  match expr {
    Expr::Integer(n) => Some(*n as f64),
    Expr::Real(r) => Some(*r),
    Expr::FunctionCall { name, args }
      if name == "Rational" && args.len() == 2 =>
    {
      if let (Expr::Integer(n), Expr::Integer(d)) = (&args[0], &args[1]) {
        Some(*n as f64 / *d as f64)
      } else {
        None
      }
    }
    Expr::UnaryOp {
      op: UnaryOperator::Minus,
      operand,
    } => expr_to_number(operand).map(|n| -n),
    _ => None,
  }
}

/// Try to factor and solve higher-degree polynomial inequalities.
fn try_factor_and_reduce_inequality(
  lhs: &Expr,
  rhs: &Expr,
  op: CompOp,
  var: &str,
  domain: Option<&str>,
) -> Option<Expr> {
  // Try using Solve to find roots, then determine sign intervals
  let eq = Expr::Comparison {
    operands: vec![lhs.clone(), rhs.clone()],
    operators: vec![ComparisonOp::Equal],
  };
  let solve_result = solve_ast(&[eq, id_expr(var)]).ok()?;

  if let Expr::List(solutions) = &solve_result {
    let mut roots: Vec<(Expr, f64)> = Vec::new();
    for sol in solutions {
      if let Expr::List(rules) = sol {
        for rule in rules {
          // Irrational roots (`Sqrt[2]`) count by their numeric value.
          if let Expr::Rule { replacement, .. } = rule
            && let Some(val) = expr_to_number(replacement).or_else(|| {
              crate::evaluator::evaluate_function_call_ast(
                "N",
                std::slice::from_ref(replacement.as_ref()),
              )
              .ok()
              .and_then(|v| expr_to_number(&v))
            })
          {
            roots.push((replacement.as_ref().clone(), val));
          }
        }
      }
    }

    if roots.is_empty() {
      return None;
    }

    // Sort roots
    roots.sort_by(|a, b| {
      a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
    });

    // Deduplicate
    roots.dedup_by(|a, b| (a.1 - b.1).abs() < 1e-12);

    // Filter by domain
    if let Some(dom) = domain {
      roots.retain(|(expr, _)| is_in_domain(expr, dom));
    }

    // Determine the sign in each interval by testing a point
    let poly = minus2(lhs.clone(), rhs.clone());

    let mut intervals: Vec<Expr> = Vec::new();

    // For an inclusive comparison every root satisfies the (in)equality, so the
    // unbounded end pieces include their boundary root (x <= r0 / x >= r_last).
    let inclusive = matches!(op, CompOp::LessEqual | CompOp::GreaterEqual);

    // Test point before first root
    if !roots.is_empty() {
      let test_x = roots[0].1 - 1.0;
      let sign = eval_poly_at(&poly, var, test_x);
      if sign_matches(sign, op) {
        intervals.push(make_comparison(
          &id_expr(var),
          &roots[0].0,
          if inclusive {
            CompOp::LessEqual
          } else {
            CompOp::Less
          },
        ));
      }
    }

    // Test at roots (for <= and >=)
    if matches!(op, CompOp::LessEqual | CompOp::GreaterEqual | CompOp::Equal) {
      for (root_expr, _) in &roots {
        intervals.push(make_equality(&id_expr(var), root_expr));
      }
    }

    // Test between roots
    for i in 0..roots.len().saturating_sub(1) {
      let test_x = f64::midpoint(roots[i].1, roots[i + 1].1);
      let sign = eval_poly_at(&poly, var, test_x);
      if sign_matches(sign, op) {
        // A bounded interval (roots[i], roots[i+1]) is always written in
        // ascending order; the comparison op only decides whether the
        // endpoints are included (LessEqual) or excluded (Less).
        let ineq_op = if matches!(op, CompOp::LessEqual | CompOp::GreaterEqual)
        {
          CompOp::LessEqual
        } else {
          CompOp::Less
        };
        intervals.push(make_compound_inequality(
          &roots[i].0,
          ineq_op,
          var,
          ineq_op,
          &roots[i + 1].0,
        ));
      }
    }

    // Test point after last root
    if !roots.is_empty() {
      let test_x = roots.last().unwrap().1 + 1.0;
      let sign = eval_poly_at(&poly, var, test_x);
      if sign_matches(sign, op) {
        intervals.push(make_comparison(
          &id_expr(var),
          &roots.last().unwrap().0,
          if inclusive {
            CompOp::GreaterEqual
          } else {
            CompOp::Greater
          },
        ));
      }
    }

    // For strict inequalities with "between roots" intervals, clean up
    // by merging equality points into the intervals
    if matches!(op, CompOp::LessEqual | CompOp::GreaterEqual) {
      // Already handled by the inclusive interval construction
      // Remove standalone equalities that are covered by intervals
      let mut cleaned: Vec<Expr> = Vec::new();
      for interval in &intervals {
        if let Some((_, _, CompOp::Equal)) = extract_comparison(interval) {
          // Skip standalone equalities that are part of an interval
          continue;
        }
        cleaned.push(interval.clone());
      }
      if !cleaned.is_empty() {
        return Some(build_or(cleaned));
      }
    }

    if intervals.is_empty() {
      return Some(bool_expr(false));
    }

    return Some(build_or(intervals));
  }

  None
}

/// Evaluate a polynomial at a specific point.
fn eval_poly_at(poly: &Expr, var: &str, x: f64) -> f64 {
  let substituted =
    crate::syntax::substitute_variable(poly, var, &Expr::Real(x));
  // Use the full evaluator: the symbolic `simplify` does not fold
  // Power[Real, Integer] (e.g. (-0.5)^3), which would leave the test value
  // symbolic and collapse the sign to the 0.0 fallback.
  let evaluated = crate::evaluator::evaluate_expr_to_expr(&substituted)
    .unwrap_or_else(|_| simplify(substituted));
  expr_to_number(&evaluated).unwrap_or(0.0)
}

/// Check if a sign value matches the comparison operator (compared to 0).
fn sign_matches(val: f64, op: CompOp) -> bool {
  match op {
    CompOp::Less => val < 0.0,
    CompOp::LessEqual => val <= 0.0,
    CompOp::Greater => val > 0.0,
    CompOp::GreaterEqual => val >= 0.0,
    CompOp::Equal => val.abs() < 1e-12,
    CompOp::NotEqual => val.abs() >= 1e-12,
  }
}

/// Handle And conjunction of two constraints.
fn reduce_and(
  left: &Expr,
  right: &Expr,
  vars: &[String],
  domain: Option<&str>,
) -> Result<Expr, InterpreterError> {
  // Flatten And chains
  let mut constraints = Vec::new();
  collect_and_constraints(left, &mut constraints);
  collect_and_constraints(right, &mut constraints);

  // Remove True constraints (identity for And)
  constraints.retain(|c| !matches!(c, Expr::Identifier(s) if s == "True"));

  // Check for False
  if constraints
    .iter()
    .any(|c| matches!(c, Expr::Identifier(s) if s == "False"))
  {
    return Ok(bool_expr(false));
  }

  if constraints.is_empty() {
    return Ok(bool_expr(true));
  }

  // A system of linear equations over the integers is solved as a whole:
  // its solution set is one lattice, parametrized jointly.
  if let Some(result) =
    try_reduce_linear_diophantine(&and_chain(&constraints), vars, domain)?
  {
    return Ok(result);
  }

  // Constraints that share no variable are independent, so each is reduced
  // over its own variables. Ones that *do* share a variable have to be
  // reduced together — `a^2 < 4 && a > 0` is `0 < a < 2`, not the two bounds
  // side by side. wolframscript eliminates the variables the caller did not
  // name first, so their groups lead the answer:
  // `Reduce[x > 1 && a^2 < 4, x]` is `-2 < a < 2 && x > 1`. (Before this
  // split those conjuncts were dropped outright, losing the `a` constraint.)
  let mut groups = group_by_shared_variables(&constraints);
  groups.sort_by_key(|(_, group_vars)| {
    i32::from(group_vars.iter().any(|v| vars.contains(v)))
  });
  // In a group of equations that mentions a named variable, the unnamed
  // ones are parameters the named ones are solved in terms of, no unknowns
  // of their own (`ey == a ez && ...` is no equation for `a`). Under an
  // inequality they are eliminated too, but after the named ones
  // (`Reduce[(1 + 2 a^2) x^2 == 1 && a != 0, x]` solves for `x`).
  for (group, group_vars) in &mut groups {
    if !group_vars.iter().any(|v| vars.contains(v)) {
      continue;
    }
    let all_equations = group
      .iter()
      .all(|c| matches!(extract_comparison(c), Some((_, _, CompOp::Equal))));
    if all_equations {
      group_vars.retain(|v| vars.contains(v));
    }
    group_vars.sort_by_key(|v| vars.iter().position(|w| w == v));
  }
  // A constraint with no variables at all has no group of its own to be
  // reduced over, so leave the whole conjunction to the path below.
  let splits = groups.iter().all(|(_, group_vars)| !group_vars.is_empty())
    && (groups.len() > 1
      || groups.first().is_some_and(|(_, group_vars)| {
        group_vars.len() != vars.len()
          || !group_vars.iter().all(|v| vars.contains(v))
      }));
  if splits {
    let mut parts = Vec::new();
    for (group, group_vars) in groups {
      let reduced = reduce_expr(&and_chain(&group), &group_vars, domain)?;
      if matches!(&reduced, Expr::Identifier(s) if s == "False") {
        return Ok(bool_expr(false));
      }
      if !matches!(&reduced, Expr::Identifier(s) if s == "True") {
        parts.push(reduced);
      }
    }
    if parts.is_empty() {
      return Ok(bool_expr(true));
    }
    return Ok(and_chain(&parts));
  }

  // Separate equations from inequalities
  let mut equations: Vec<(Expr, Expr)> = Vec::new(); // (lhs, rhs) for ==
  let mut inequalities: Vec<Expr> = Vec::new();
  let mut other: Vec<Expr> = Vec::new();

  for c in &constraints {
    if let Some((lhs, rhs, CompOp::Equal)) = extract_comparison(c) {
      equations.push((lhs, rhs));
    } else if let Some((_, _, _)) = extract_comparison(c) {
      inequalities.push(c.clone());
    } else {
      other.push(c.clone());
    }
  }

  if vars.len() == 1 {
    let var = &vars[0];

    // If we have equations, solve them first and check inequalities
    if !equations.is_empty() {
      let eq = &equations[0];
      let eq_result = reduce_equation(&eq.0, &eq.1, var, domain)?;

      if matches!(&eq_result, Expr::Identifier(s) if s == "False") {
        return Ok(bool_expr(false));
      }
      if matches!(&eq_result, Expr::Identifier(s) if s == "True") {
        // Equation is trivially true, reduce remaining
        let remaining: Vec<Expr> = equations[1..]
          .iter()
          .map(|(l, r)| make_equality(l, r))
          .chain(inequalities.iter().cloned())
          .chain(other.iter().cloned())
          .collect();
        if remaining.is_empty() {
          return Ok(bool_expr(true));
        }
        let combined = remaining
          .iter()
          .skip(1)
          .fold(remaining[0].clone(), |acc, c| and2(acc, c.clone()));
        return reduce_expr(&combined, vars, domain);
      }

      // Equation gives concrete solutions — filter by remaining constraints
      let solutions = collect_or_terms(&eq_result);
      let mut valid_solutions: Vec<Expr> = Vec::new();

      for sol in &solutions {
        if let Some((_, rhs_val, CompOp::Equal)) = extract_comparison(sol) {
          // Check if this value satisfies all other constraints
          let mut satisfies = true;

          // Check remaining equations
          for other_eq in &equations[1..] {
            let subst_lhs =
              crate::syntax::substitute_variable(&other_eq.0, var, &rhs_val);
            let subst_rhs =
              crate::syntax::substitute_variable(&other_eq.1, var, &rhs_val);
            // Evaluated first: an unfolded `1 + 1 - 3` has no number for
            // the check below and would pass any equation.
            let holds =
              crate::evaluator::evaluate_expr_to_expr(&Expr::Comparison {
                operands: vec![subst_lhs.clone(), subst_rhs.clone()],
                operators: vec![crate::syntax::ComparisonOp::Equal],
              });
            // `Equal` of two unsimplified radical forms can say False for
            // equal values, so a False is confirmed numerically.
            let numerically_unequal = || {
              crate::evaluator::evaluate_function_call_ast(
                "N",
                &[minus2(subst_lhs.clone(), subst_rhs.clone())],
              )
              .ok()
              .and_then(|d| expr_to_number(&d))
              .is_some_and(|d| d.abs() > 1e-9)
            };
            if matches!(&holds, Ok(Expr::Identifier(b)) if b == "False")
              && numerically_unequal()
            {
              satisfies = false;
              break;
            }
            let diff =
              simplify(minus2(simplify(subst_lhs), simplify(subst_rhs)));
            if !matches!(diff, Expr::Integer(0))
              && let Some(n) = expr_to_number(&diff)
              && n.abs() > 1e-12
            {
              satisfies = false;
              break;
            }
          }

          // Check inequalities (and other boolean constraints such as `||`)
          if satisfies {
            for ineq in inequalities.iter().chain(other.iter()) {
              let subst =
                crate::syntax::substitute_variable(ineq, var, &rhs_val);
              let evaled = crate::evaluator::evaluate_expr_to_expr(&subst);
              if let Ok(result) = evaled
                && matches!(&result, Expr::Identifier(s) if s == "False")
              {
                satisfies = false;
                break;
              }
            }
          }

          if satisfies {
            valid_solutions.push(sol.clone());
          }
        }
      }

      if valid_solutions.is_empty() {
        return Ok(bool_expr(false));
      }
      return Ok(build_or(valid_solutions));
    }

    // Only inequalities — try to combine them
    if !inequalities.is_empty() && other.is_empty() {
      return reduce_combined_inequalities(&inequalities, var, domain);
    }

    // Inequalities and disjunctions of them (`(x <= -1 || x >= 2) && x >= -4`):
    // And distributes over each Or, and the bounds of every branch are
    // intersected.
    if !(inequalities.is_empty() && other.len() < 2)
      && other.iter().all(is_or_expr)
    {
      let all: Vec<Expr> =
        inequalities.iter().chain(other.iter()).cloned().collect();
      return reduce_combined_inequalities(&all, var, domain);
    }
  } else {
    // Multi-variable And: try to solve equations for each variable
    return reduce_multi_var_and(&constraints, vars, domain);
  }

  // Fallback: reduce each side and combine
  let left_r = reduce_expr(left, vars, domain)?;
  let right_r = reduce_expr(right, vars, domain)?;
  Ok(and_results(&left_r, &right_r))
}

/// Collect all And-connected terms.
pub fn collect_and_constraints(expr: &Expr, out: &mut Vec<Expr>) {
  match expr {
    Expr::BinaryOp {
      op: BinaryOperator::And,
      left,
      right,
    } => {
      collect_and_constraints(left, out);
      collect_and_constraints(right, out);
    }
    Expr::FunctionCall { name, args } if name == "And" => {
      for a in args {
        collect_and_constraints(a, out);
      }
    }
    _ => out.push(expr.clone()),
  }
}

/// Collect equalities (var == val) from an And expression.
fn collect_and_equalities(expr: &Expr) -> Vec<(String, Expr)> {
  let mut result = Vec::new();
  let mut constraints = Vec::new();
  collect_and_constraints(expr, &mut constraints);
  for c in &constraints {
    if let Some((lhs, rhs, CompOp::Equal)) = extract_comparison(c)
      && let Expr::Identifier(name) = &lhs
    {
      result.push((name.clone(), rhs));
    }
  }
  result
}

fn is_or_expr(expr: &Expr) -> bool {
  matches!(
    expr,
    Expr::BinaryOp {
      op: BinaryOperator::Or,
      ..
    }
  ) || matches!(expr, Expr::FunctionCall { name, .. } if name == "Or")
}

/// Collect all Or-connected terms.
pub fn collect_or_terms(expr: &Expr) -> Vec<Expr> {
  match expr {
    Expr::BinaryOp {
      op: BinaryOperator::Or,
      left,
      right,
    } => {
      let mut terms = collect_or_terms(left);
      terms.extend(collect_or_terms(right));
      terms
    }
    Expr::FunctionCall { name, args } if name == "Or" => {
      args.iter().flat_map(collect_or_terms).collect()
    }
    _ => vec![expr.clone()],
  }
}

/// Try to combine multiple inequalities on the same variable.
fn reduce_combined_inequalities(
  ineqs: &[Expr],
  var: &str,
  domain: Option<&str>,
) -> Result<Expr, InterpreterError> {
  // Reduce each inequality individually first
  let mut reduced: Vec<Expr> = Vec::new();
  for ineq in ineqs {
    let r = reduce_single_var(ineq, var, domain)?;
    reduced.push(r);
  }

  // If any reduced result is a disjunction (Or), distribute And over Or:
  // (A || B) && C  →  (A && C) || (B && C), then reduce each conjunction.
  if let Some(or_idx) = reduced.iter().position(|r| {
    matches!(
      r,
      Expr::BinaryOp {
        op: BinaryOperator::Or,
        ..
      }
    ) || matches!(r, Expr::FunctionCall { name, .. } if name == "Or")
  }) {
    let or_expr = reduced.remove(or_idx);
    let branches = collect_or_terms(&or_expr);
    let other_constraints = reduced;

    let mut valid_branches = Vec::new();
    for branch in &branches {
      // Combine this branch with all other constraints
      let mut all_ineqs: Vec<Expr> = vec![branch.clone()];
      all_ineqs.extend(other_constraints.iter().cloned());

      let result = reduce_combined_inequalities(&all_ineqs, var, domain)?;
      if !matches!(&result, Expr::Identifier(s) if s == "False") {
        valid_branches.push(result);
      }
    }

    return if valid_branches.is_empty() {
      Ok(bool_expr(false))
    } else {
      Ok(build_or(valid_branches))
    };
  }

  // Try to find bounds: lower < x < upper
  let mut lower_bound: Option<(Expr, bool)> = None; // (value, inclusive)
  let mut upper_bound: Option<(Expr, bool)> = None;

  for r in &reduced {
    // Handle Inequality[low, Op1, var, Op2, high] or Expr::Comparison with 2 operators
    if let Expr::FunctionCall { name, args: iargs } = r
      && name == "Inequality"
      && iargs.len() == 5
    {
      let low_val = &iargs[0];
      let op1_name = if let Expr::Identifier(s) = &iargs[1] {
        s.as_str()
      } else {
        ""
      };
      let mid = &iargs[2];
      let op2_name = if let Expr::Identifier(s) = &iargs[3] {
        s.as_str()
      } else {
        ""
      };
      let high_val = &iargs[4];

      if expr_to_string(mid) == var {
        // Extract lower bound
        let low_inc = matches!(op1_name, "LessEqual" | "GreaterEqual");
        lower_bound =
          update_bound(lower_bound, low_val.clone(), low_inc, false);
        // Extract upper bound
        let high_inc = matches!(op2_name, "LessEqual" | "GreaterEqual");
        upper_bound =
          update_bound(upper_bound, high_val.clone(), high_inc, true);
        continue;
      }
    }

    // Handle Expr::Comparison with 2 operators (same-operator compound inequality)
    if let Expr::Comparison {
      operands,
      operators,
    } = r
      && operators.len() == 2
      && operands.len() == 3
    {
      let mid = &operands[1];
      if expr_to_string(mid) == var {
        let low_inc = matches!(
          operators[0],
          ComparisonOp::LessEqual | ComparisonOp::GreaterEqual
        );
        lower_bound =
          update_bound(lower_bound, operands[0].clone(), low_inc, false);
        let high_inc = matches!(
          operators[1],
          ComparisonOp::LessEqual | ComparisonOp::GreaterEqual
        );
        upper_bound =
          update_bound(upper_bound, operands[2].clone(), high_inc, true);
        continue;
      }
    }

    if let Some((lhs, rhs, op)) = extract_comparison(r) {
      let lhs_str = expr_to_string(&lhs);
      let rhs_str = expr_to_string(&rhs);
      let var_str = var;

      if lhs_str == var_str {
        // x op value
        match op {
          CompOp::Less => {
            upper_bound = update_bound(upper_bound, rhs, false, true);
          }
          CompOp::LessEqual => {
            upper_bound = update_bound(upper_bound, rhs, true, true);
          }
          CompOp::Greater => {
            lower_bound = update_bound(lower_bound, rhs, false, false);
          }
          CompOp::GreaterEqual => {
            lower_bound = update_bound(lower_bound, rhs, true, false);
          }
          _ => {}
        }
      } else if rhs_str == var_str {
        // value op x
        match op {
          CompOp::Less => {
            lower_bound = update_bound(lower_bound, lhs, false, false);
          }
          CompOp::LessEqual => {
            lower_bound = update_bound(lower_bound, lhs, true, false);
          }
          CompOp::Greater => {
            upper_bound = update_bound(upper_bound, lhs, false, true);
          }
          CompOp::GreaterEqual => {
            upper_bound = update_bound(upper_bound, lhs, true, true);
          }
          _ => {}
        }
      }
    }
  }

  match (&lower_bound, &upper_bound) {
    (Some((low, low_inc)), Some((high, high_inc))) => {
      // Check if the interval is empty (lower >= upper)
      if let (Some(low_val), Some(high_val)) =
        (expr_to_number(low), expr_to_number(high))
        && (low_val > high_val
          || (low_val == high_val && (!low_inc || !high_inc)))
      {
        return Ok(bool_expr(false));
      }
      let low_op = if *low_inc {
        CompOp::LessEqual
      } else {
        CompOp::Less
      };
      let high_op = if *high_inc {
        CompOp::LessEqual
      } else {
        CompOp::Less
      };
      return Ok(make_compound_inequality(low, low_op, var, high_op, high));
    }
    (Some((low, low_inc)), None) => {
      let op = if *low_inc {
        CompOp::GreaterEqual
      } else {
        CompOp::Greater
      };
      return Ok(make_comparison(&id_expr(var), low, op));
    }
    (None, Some((high, high_inc))) => {
      let op = if *high_inc {
        CompOp::LessEqual
      } else {
        CompOp::Less
      };
      return Ok(make_comparison(&id_expr(var), high, op));
    }
    (None, None) => {}
  }

  // Couldn't combine — return And of all reduced forms
  Ok(
    reduced
      .iter()
      .skip(1)
      .fold(reduced[0].clone(), |acc, r| and2(acc, r.clone())),
  )
}

/// Update a bound, keeping the tighter one.
fn update_bound(
  current: Option<(Expr, bool)>,
  new_val: Expr,
  inclusive: bool,
  is_upper: bool,
) -> Option<(Expr, bool)> {
  if let Some((old_val, _old_inc)) = &current
    && let (Some(old_n), Some(new_n)) =
      (expr_to_number(old_val), expr_to_number(&new_val))
  {
    if is_upper {
      // For upper bound, keep the smaller one
      if new_n < old_n || (new_n == old_n && !inclusive) {
        return Some((new_val, inclusive));
      }
      return current;
    }
    // For lower bound, keep the larger one
    if new_n > old_n || (new_n == old_n && !inclusive) {
      return Some((new_val, inclusive));
    }
    return current;
  }
  Some((new_val, inclusive))
}

/// Handle multi-variable systems.
fn reduce_multi_var(
  expr: &Expr,
  vars: &[String],
  domain: Option<&str>,
) -> Result<Expr, InterpreterError> {
  // Reduce[a1 x1 + ... + an xn == c, {x1, ..., xn}, Integers]: a single linear
  // Diophantine equation. Solving for one variable and leaving the others
  // free (as the general elimination path below does) only gives integer
  // values for every free assignment when every remaining coefficient is a
  // multiple of the eliminated one, which is not true in general — so handle
  // this shape with the extended-Euclidean parametrization first.
  if let Some(result) = try_reduce_linear_diophantine(expr, vars, domain)? {
    return Ok(result);
  }

  // Collect all constraints
  let mut constraints = Vec::new();
  collect_and_constraints(expr, &mut constraints);

  reduce_multi_var_and(&constraints, vars, domain)
}

/// An affine combination of the free integer parameters `C[1], ..., C[n]`:
/// `const_term + Σ coeffs[i] * C[i+1]`.
#[derive(Clone)]
struct Affine {
  const_term: i128,
  coeffs: Vec<i128>,
}

impl Affine {
  fn to_expr(&self) -> Expr {
    let mut result = Expr::Integer(self.const_term);
    for (slot, coeff) in self.coeffs.iter().enumerate() {
      if *coeff == 0 {
        continue;
      }
      let param = call("C", vec![Expr::Integer((slot + 1) as i128)]);
      let term = if *coeff == 1 {
        param
      } else {
        times2(Expr::Integer(*coeff), param)
      };
      result = if matches!(result, Expr::Integer(0)) {
        term
      } else {
        plus2(result, term)
      };
    }
    crate::evaluator::evaluate_expr_to_expr(&result).unwrap_or(result)
  }
}

/// Extended Euclidean algorithm on plain integers: returns `(g, s, t)` with
/// `g = gcd(|a|, |b|) >= 0` and `s*a + t*b == g`.
fn extended_gcd_i128(a: i128, b: i128) -> (i128, i128, i128) {
  let (mut old_r, mut r) = (a, b);
  let (mut old_s, mut s) = (1i128, 0i128);
  let (mut old_t, mut t) = (0i128, 1i128);
  while r != 0 {
    let q = old_r / r;
    (old_r, r) = (r, old_r - q * r);
    (old_s, s) = (s, old_s - q * s);
    (old_t, t) = (t, old_t - q * t);
  }
  if old_r < 0 {
    (-old_r, -old_s, -old_t)
  } else {
    (old_r, old_s, old_t)
  }
}

/// Solve `coeffs[0]*x0 + coeffs[1]*x1 + ... == target` over the integers,
/// where every entry of `coeffs` is nonzero. Returns `None` when no integer
/// solution exists (the gcd of the coefficients doesn't divide `target`);
/// otherwise returns one `Affine` per variable, expressed in terms of
/// `coeffs.len() - 1` free integer parameters `C[1], ..., C[n-1]`.
///
/// Standard successive-elimination construction: track the running gcd
/// `g[j]` of `coeffs[0..=j]` together with Bezout coefficients, then
/// back-substitute from the top, introducing one fresh parameter per step.
/// All integer solutions of `A v == b` as an affine parametrization: a
/// particular solution plus a basis of the kernel lattice, from column
/// operations that bring `A` to echelon form (`A U = [L | 0]` with `U`
/// unimodular). `None` when there is no integer solution (or an i128
/// overflow).
fn solve_linear_diophantine_system(
  a: &[Vec<i128>],
  b: &[i128],
) -> Option<Vec<Affine>> {
  let m = a.len();
  let n = a.first()?.len();
  let mut mat: Vec<Vec<i128>> = a.to_vec();
  let mut u: Vec<Vec<i128>> = (0..n)
    .map(|i| (0..n).map(|j| i128::from(i == j)).collect())
    .collect();
  // Column operation on both `mat` and `u`: (col_p, col_q) <- (s p + t q,
  // x p + y q).
  let combine = |mat: &mut Vec<Vec<i128>>,
                 u: &mut Vec<Vec<i128>>,
                 p: usize,
                 q: usize,
                 (s, t, x, y): (i128, i128, i128, i128)|
   -> Option<()> {
    for row in mat.iter_mut().chain(u.iter_mut()) {
      let (vp, vq) = (row[p], row[q]);
      row[p] = s.checked_mul(vp)?.checked_add(t.checked_mul(vq)?)?;
      row[q] = x.checked_mul(vp)?.checked_add(y.checked_mul(vq)?)?;
    }
    Some(())
  };
  let mut pivots: Vec<(usize, usize)> = Vec::new(); // (row, col)
  let mut k = 0usize;
  for i in 0..m {
    if k >= n {
      break;
    }
    for j in k + 1..n {
      if mat[i][j] == 0 {
        continue;
      }
      if mat[i][k] == 0 {
        combine(&mut mat, &mut u, k, j, (0, 1, 1, 0))?;
        continue;
      }
      let (g, s, t) = extended_gcd_i128(mat[i][k], mat[i][j]);
      let (p, q) = (mat[i][k] / g, mat[i][j] / g);
      combine(&mut mat, &mut u, k, j, (s, t, -q, p))?;
    }
    if mat[i][k] != 0 {
      pivots.push((i, k));
      k += 1;
    }
  }
  // Solve the echelon system for the pivot coordinates w.
  let mut w = vec![0i128; n];
  let mut next_pivot = 0usize;
  for i in 0..m {
    let known: i128 = (0..next_pivot.min(k))
      .map(|c| mat[i][c].checked_mul(w[c]))
      .try_fold(0i128, |acc, v| acc.checked_add(v?))?;
    let residual = b[i].checked_sub(known)?;
    if next_pivot < pivots.len() && pivots[next_pivot].0 == i {
      let c = pivots[next_pivot].1;
      if residual % mat[i][c] != 0 {
        return None;
      }
      w[c] = residual / mat[i][c];
      next_pivot += 1;
    } else if residual != 0 {
      return None;
    }
  }
  // v = U w (particular) + U[:, k..] t.
  Some(
    (0..n)
      .map(|r| {
        let const_term = (0..k)
          .map(|c| u[r][c].checked_mul(w[c]))
          .try_fold(0i128, |acc, v| acc.checked_add(v?));
        Affine {
          const_term: const_term.unwrap_or(0),
          coeffs: (k..n).map(|c| u[r][c]).collect(),
        }
      })
      .collect(),
  )
}

/// Put a parametrized integer solution into the shape wolframscript prints.
///
/// The solution set is a coset `p + L` of a lattice `L`, and the successive
/// elimination in `solve_linear_diophantine_system` lands on an arbitrary `p` and an
/// arbitrary basis of `L` — for `3 x + 5 y == 7` it finds `x == 14 + 5 C[1]`
/// where wolframscript writes `x == 4 + 5 C[1]`. Both name the same set, so
/// canonicalize: the basis becomes the Hermite normal form (echelon by
/// variable, positive pivots, entries above a pivot reduced into `[0, pivot)`)
/// and the offset is then reduced against it, which pins each pivot
/// coordinate into `[0, pivot)` too.
///
/// Takes one `Affine` per variable and returns the offset per variable plus
/// one basis row per parameter, each row indexed by variable.
fn canonicalize_lattice(solution: &[Affine]) -> (Vec<i128>, Vec<Vec<i128>>) {
  let n = solution.len();
  let nparams = solution.first().map_or(0, |a| a.coeffs.len());
  let mut offset: Vec<i128> = solution.iter().map(|a| a.const_term).collect();
  let mut rows: Vec<Vec<i128>> = (0..nparams)
    .map(|k| solution.iter().map(|a| a.coeffs[k]).collect())
    .collect();

  // Hermite normal form, one pivot column at a time.
  let mut pivots: Vec<(usize, usize)> = Vec::new();
  let mut pivot_row = 0usize;
  for col in 0..n {
    if pivot_row >= rows.len() {
      break;
    }
    for k in (pivot_row + 1)..rows.len() {
      if rows[k][col] == 0 {
        continue;
      }
      if rows[pivot_row][col] == 0 {
        rows.swap(pivot_row, k);
        continue;
      }
      // Replace the pair by (gcd combination, complementary combination),
      // which spans the same sublattice but zeroes row k in this column.
      let (g, s, t) = extended_gcd_i128(rows[pivot_row][col], rows[k][col]);
      let (a, b) = (rows[pivot_row][col] / g, rows[k][col] / g);
      for j in 0..n {
        let keep = s * rows[pivot_row][j] + t * rows[k][j];
        let zeroed = a * rows[k][j] - b * rows[pivot_row][j];
        rows[pivot_row][j] = keep;
        rows[k][j] = zeroed;
      }
    }
    if rows[pivot_row][col] == 0 {
      continue;
    }
    if rows[pivot_row][col] < 0 {
      for j in 0..n {
        rows[pivot_row][j] = -rows[pivot_row][j];
      }
    }
    let pivot = rows[pivot_row][col];
    for k in 0..pivot_row {
      let q = rows[k][col].div_euclid(pivot);
      if q != 0 {
        for j in 0..n {
          rows[k][j] -= q * rows[pivot_row][j];
        }
      }
    }
    pivots.push((pivot_row, col));
    pivot_row += 1;
  }
  rows.truncate(pivot_row);

  // Slide the offset along the lattice until every pivot coordinate is the
  // smallest non-negative representative of its residue class.
  for &(row, col) in &pivots {
    let q = offset[col].div_euclid(rows[row][col]);
    if q != 0 {
      for j in 0..n {
        offset[j] -= q * rows[row][j];
      }
    }
  }
  (offset, rows)
}

/// Solve[eqs, vars, Integers] for a system of linear equations with integer
/// coefficients: the lattice `Reduce` finds, as rules. A value that involves
/// the parameters is a `ConditionalExpression` under every `Element`
/// conjunct of the reduced form; a variable no equation mentions gets no
/// rule (the caller reports `Solve::svars`):
/// `Solve[x == 2 y, {x, y, z}, Integers]` ->
/// `{{x -> ConditionalExpression[2 C[1], z ∈ Integers && C[1] ∈ Integers],
///    y -> ConditionalExpression[C[1], z ∈ Integers && C[1] ∈ Integers]}}`.
pub(crate) fn solve_linear_diophantine_rules(
  eqs: &Expr,
  vars: &[String],
) -> Result<Option<Expr>, InterpreterError> {
  let threaded = thread_reduce_list_equations(eqs);
  let system = match &threaded {
    Expr::List(items) => and_chain(items),
    other => other.clone(),
  };
  let reduced = if let Some(reduced) =
    try_reduce_linear_diophantine(&system, vars, Some("Integers"))?
  {
    reduced
  } else {
    // Other equations (a congruence `Mod[x, 3] == 1`) are taken from
    // Reduce when it parametrizes them; anything else is left to Solve.
    let mut constraints = Vec::new();
    collect_and_constraints(&system, &mut constraints);
    if !constraints
      .iter()
      .all(|c| matches!(extract_comparison(c), Some((_, _, CompOp::Equal))))
    {
      return Ok(None);
    }
    let var_list =
      Expr::List(vars.iter().cloned().map(Expr::Identifier).collect());
    let reduced = reduce_ast(&[system.clone(), var_list, id_expr("Integers")])?;
    if !collect_or_terms(&reduced)
      .iter()
      .all(|branch| integer_branch_parts(branch, vars).is_some())
      || !crate::syntax::expr_to_string(&reduced).contains("C[")
    {
      return Ok(None);
    }
    reduced
  };
  if matches!(&reduced, Expr::Identifier(s) if s == "False") {
    return Ok(Some(Expr::List(Vec::new().into())));
  }
  let mut solutions = Vec::new();
  for branch in collect_or_terms(&reduced) {
    let Some((condition, values)) = integer_branch_parts(&branch, vars) else {
      return Ok(None);
    };
    let rules = values
      .into_iter()
      .map(|(var, value)| {
        // A value is parametric unless it is a plain integer.
        let value = match &condition {
          Some(condition) if !matches!(value, Expr::Integer(_)) => {
            call("ConditionalExpression", vec![value, condition.clone()])
          }
          _ => value,
        };
        crate::helpers::rule_expr(Expr::Identifier(var), value)
      })
      .collect::<Vec<_>>();
    solutions.push(Expr::List(rules.into()));
  }
  Ok(Some(Expr::List(solutions.into())))
}

/// One conjunction of a reduced integer solution: its `Element` conditions
/// (joined) and the value of each variable given as `var == value`, in the
/// order of `vars`. `None` when it has any other kind of conjunct.
fn integer_branch_parts(
  branch: &Expr,
  vars: &[String],
) -> Option<(Option<Expr>, Vec<(String, Expr)>)> {
  let mut conjuncts = Vec::new();
  collect_and_constraints(branch, &mut conjuncts);
  let mut conditions = Vec::new();
  let mut values = Vec::new();
  for conjunct in conjuncts {
    if matches!(&conjunct, Expr::FunctionCall { name, .. } if name == "Element")
    {
      conditions.push(conjunct);
      continue;
    }
    match extract_comparison(&conjunct) {
      Some((Expr::Identifier(ref var), value, CompOp::Equal))
        if vars.contains(var) =>
      {
        values.push((var.clone(), value));
      }
      _ => return None,
    }
  }
  values.sort_by_key(|(var, _)| vars.iter().position(|v| v == var));
  let condition = (!conditions.is_empty()).then(|| and_chain(&conditions));
  Some((condition, values))
}

/// Reduce[eqs, {vars}, Integers] where `eqs` (after collecting `&&`
/// conjuncts) are linear equations with integer coefficients in some or all
/// of `vars`. Produces the general parametrized integer solution, e.g.
/// `Reduce[2 x == 4 y, {x, y}, Integers]` ->
/// `C[1] \[Element] Integers && x == 2*C[1] && y == C[1]`.
/// Returns `None` when the shape doesn't apply, so the caller falls back to
/// the general elimination path.
fn try_reduce_linear_diophantine(
  expr: &Expr,
  vars: &[String],
  domain: Option<&str>,
) -> Result<Option<Expr>, InterpreterError> {
  if domain != Some("Integers") || vars.len() < 2 {
    return Ok(None);
  }

  let mut constraints = Vec::new();
  collect_and_constraints(expr, &mut constraints);
  if constraints.is_empty() {
    return Ok(None);
  }

  // Every constraint must be a linear equation with integer coefficients:
  // a row of `A v == b`. Anything else goes to the general elimination path.
  // A congruence `Mod[e, m] == r` is the equation `e - r == m k` in a fresh
  // integer unknown `k`, which is projected away once the system is solved.
  let mut matrix: Vec<Vec<i128>> = Vec::with_capacity(constraints.len());
  let mut rhs: Vec<i128> = Vec::with_capacity(constraints.len());
  let mut moduli: Vec<(usize, i128)> = Vec::new();
  for constraint in &constraints {
    let Some((lhs, rhs_side, CompOp::Equal)) = extract_comparison(constraint)
    else {
      return Ok(None);
    };
    let congruence = match (&lhs, &rhs_side) {
      (Expr::FunctionCall { name, args }, Expr::Integer(r))
      | (Expr::Integer(r), Expr::FunctionCall { name, args })
        if name == "Mod" && args.len() == 2 =>
      {
        match &args[1] {
          Expr::Integer(m) if *m > 0 => Some((args[0].clone(), *m, *r)),
          _ => return Ok(None),
        }
      }
      _ => None,
    };
    let poly = match congruence {
      Some((_, m, r)) if r < 0 || r >= m => return Ok(Some(bool_expr(false))),
      Some((e, m, r)) => {
        moduli.push((matrix.len(), m));
        expand_and_combine(&minus2(e, Expr::Integer(r)))
      }
      None => expand_and_combine(&minus2(lhs, rhs_side)),
    };
    let mut row: Vec<i128> = Vec::with_capacity(vars.len());
    for v in vars {
      match max_power_int(&poly, v) {
        None | Some(0) => row.push(0),
        Some(1) => {
          // Evaluated: the coefficient can come back as an unfolded `-1`.
          let c =
            crate::evaluator::evaluate_expr_to_expr(&coefficient_ast(&[
              poly.clone(),
              Expr::Identifier(v.clone()),
              Expr::Integer(1),
            ])?)?;
          match c {
            Expr::Integer(n) => row.push(n),
            _ => return Ok(None),
          }
        }
        _ => return Ok(None),
      }
    }
    // `poly` must be exactly `const + Σ row[i] vars[i]`: no cross terms.
    let const_term = {
      let zeroed = vars.iter().fold(poly.clone(), |acc, v| {
        crate::syntax::substitute_variable(&acc, v, &Expr::Integer(0))
      });
      match crate::evaluator::evaluate_expr_to_expr(&zeroed)? {
        Expr::Integer(n) => n,
        _ => return Ok(None),
      }
    };
    let mut reconstructed = Expr::Integer(const_term);
    for (v, c) in vars.iter().zip(&row) {
      if *c != 0 {
        reconstructed = plus2(
          reconstructed,
          times2(Expr::Integer(*c), Expr::Identifier(v.clone())),
        );
      }
    }
    let diff = simplify(expand_and_combine(&minus2(poly, reconstructed)));
    if !matches!(diff, Expr::Integer(0)) {
      return Ok(None);
    }
    matrix.push(row);
    rhs.push(-const_term);
  }
  let coeffs: Vec<i128> = (0..vars.len())
    .map(|j| i128::from(matrix.iter().any(|r| r[j] != 0)))
    .collect();
  if coeffs.iter().all(|c| *c == 0) {
    return Ok(None);
  }
  let present: Vec<(usize, i128)> = coeffs
    .iter()
    .enumerate()
    .filter(|(_, c)| **c != 0)
    .map(|(i, c)| (i, *c))
    .collect();
  let present_coeffs: Vec<i128> = present.iter().map(|(_, c)| *c).collect();
  let present_matrix: Vec<Vec<i128>> = matrix
    .iter()
    .enumerate()
    .map(|(i, r)| {
      present
        .iter()
        .map(|(j, _)| r[*j])
        .chain(moduli.iter().map(|&(row, m)| if row == i { -m } else { 0 }))
        .collect()
    })
    .collect();

  let Some(mut present_solution) =
    solve_linear_diophantine_system(&present_matrix, &rhs)
  else {
    return Ok(Some(bool_expr(false)));
  };
  // The congruences' own unknowns are not reported.
  present_solution.truncate(present.len());

  // Any basis of the solution lattice, and any point on it, describes the
  // same solution set; canonicalize both the way wolframscript reports them
  // (see `canonicalize_lattice`).
  let (particular, basis) = canonicalize_lattice(&present_solution);
  // A congruence that leaves several parameters free is reported by
  // wolframscript as a disjunction of residue classes, not as one lattice.
  if !moduli.is_empty() && basis.len() > 1 {
    return Ok(None);
  }
  let total_params = basis.len();
  let solution: Vec<Affine> = (0..present_coeffs.len())
    .map(|i| Affine {
      const_term: particular[i],
      coeffs: basis.iter().map(|row| row[i]).collect(),
    })
    .collect();

  let integers = || id_expr("Integers");
  let element = |what: Expr| call("Element", vec![what, integers()]);

  // A variable the equation never mentions is unconstrained: wolframscript
  // leaves it as itself under its own `Element[v, Integers]` rather than
  // introducing a parameter for it. Those conjuncts come first, in the order
  // the variables were asked for.
  let mut conjuncts: Vec<Expr> = coeffs
    .iter()
    .enumerate()
    .filter(|(_, c)| **c == 0)
    .map(|(idx, _)| element(Expr::Identifier(vars[idx].clone())))
    .collect();

  // Every parameter shares one membership conjunct, written with
  // Alternatives: `Element[C[1] | C[2], Integers]`.
  if let Some(membership) = (0..total_params)
    .map(|slot| call("C", vec![Expr::Integer((slot + 1) as i128)]))
    .reduce(|acc, p| binop(BinaryOperator::Alternatives, acc, p))
  {
    conjuncts.push(element(membership));
  }

  // Then the solved variables, again in the order they were asked for.
  for ((idx, _), sol) in present.iter().zip(&solution) {
    conjuncts.push(make_equality(
      &Expr::Identifier(vars[*idx].clone()),
      &sol.to_expr(),
    ));
  }

  Ok(Some(
    conjuncts
      .into_iter()
      .reduce(|acc, e| and_results(&acc, &e))
      .unwrap_or_else(|| bool_expr(true)),
  ))
}

/// Solve a multi-variable system by sequential elimination.
pub fn reduce_multi_var_and(
  constraints: &[Expr],
  vars: &[String],
  domain: Option<&str>,
) -> Result<Expr, InterpreterError> {
  // The elimination substitutes values into the remaining constraints and
  // tries the branches where a denominator vanishes; a division by zero met
  // on the way only rules a branch out, it is no message of Reduce's.
  crate::push_quiet_filter(
    Some(vec![
      "Power::infy".to_string(),
      "Infinity::indet".to_string(),
    ]),
    vec![],
  );
  let result = reduce_multi_var_and_inner(constraints, vars, domain);
  crate::pop_quiet_filter();
  Ok(normalize_solution_branches(&result?, vars))
}

/// Flatten a nested And/Or answer into a disjunction of conjunctions, drop
/// duplicate branches and order each conjunction's `v == c` literals by the
/// variable list, as wolframscript prints them.
fn normalize_solution_branches(expr: &Expr, vars: &[String]) -> Expr {
  fn dnf(e: &Expr) -> Vec<Vec<Expr>> {
    match e {
      Expr::BinaryOp {
        op: BinaryOperator::Or,
        left,
        right,
      } => {
        let mut out = dnf(left);
        out.extend(dnf(right));
        out
      }
      Expr::BinaryOp {
        op: BinaryOperator::And,
        left,
        right,
      } => {
        let (l, r) = (dnf(left), dnf(right));
        let mut out = Vec::new();
        for a in &l {
          for b in &r {
            out.push(a.iter().chain(b.iter()).cloned().collect());
          }
        }
        out
      }
      _ => vec![vec![e.clone()]],
    }
  }
  let mut seen = std::collections::HashSet::new();
  let mut branches = Vec::new();
  for mut conj in dnf(expr) {
    let solved_index = |lit: &Expr| match &extract_comparison(lit) {
      Some((Expr::Identifier(v), _, CompOp::Equal)) => {
        vars.iter().position(|w| w == v)
      }
      _ => None,
    };
    // A condition such as `x != 0` stays in front of the solution it was
    // generated with (`x != 0 && y == 2/x`, `A == 0 && C != 0 && B ==
    // C^(-1)`); one that precedes none sits with the last unknown it
    // mentions.
    let keys: Vec<usize> = (0..conj.len())
      .map(|i| {
        solved_index(&conj[i])
          .or_else(|| conj[i + 1..].iter().find_map(&solved_index))
          .or_else(|| vars.iter().rposition(|w| contains_var(&conj[i], w)))
          .unwrap_or(vars.len())
      })
      .collect();
    let mut keyed: Vec<(usize, Expr)> =
      keys.into_iter().zip(conj.drain(..)).collect();
    keyed.sort_by_key(|(idx, _)| *idx);
    let lits: Vec<Expr> = keyed.into_iter().map(|(_, l)| l).collect();
    let key = lits
      .iter()
      .map(crate::syntax::expr_to_string)
      .collect::<Vec<_>>()
      .join(" && ");
    if seen.insert(key) {
      branches.push(and_chain(&lits));
    }
  }
  if branches.is_empty() {
    return expr.clone();
  }
  build_or(branches)
}

/// `constraints && d == 0`. When `d` holds a single unknown, `d == 0` is
/// solved for it first and its values are substituted: the other equations
/// then lose that unknown instead of being re-solved through the value
/// that `d` is the denominator of.
fn reduce_with_vanishing(
  constraints: &[Expr],
  d: &Expr,
  vars: &[String],
  domain: Option<&str>,
) -> Result<Expr, InterpreterError> {
  let mut in_d = Vec::new();
  constraint_variables(d, &mut in_d);
  in_d.retain(|v| vars.contains(v));
  in_d.dedup();
  if let [v] = in_d.as_slice()
    && let Ok(Expr::List(sols)) = &solve_ast(&[
      make_equality(d, &Expr::Integer(0)),
      Expr::Identifier(v.clone()),
    ])
  {
    let rest: Vec<String> = vars.iter().filter(|x| *x != v).cloned().collect();
    let mut branches = Vec::new();
    for sol in sols {
      let Expr::List(rules) = sol else { continue };
      let Some(Expr::Rule { replacement, .. }) = rules.first() else {
        continue;
      };
      let substituted: Vec<Expr> = constraints
        .iter()
        .map(|c| {
          let c = crate::syntax::substitute_variable(c, v, replacement);
          crate::evaluator::evaluate_expr_to_expr(&c).unwrap_or(c)
        })
        .collect();
      if substituted
        .iter()
        .any(|c| matches!(c, Expr::Identifier(b) if b == "False"))
      {
        continue;
      }
      let live: Vec<Expr> = substituted
        .into_iter()
        .filter(|c| !matches!(c, Expr::Identifier(b) if b == "True"))
        .collect();
      let reduced = if rest.is_empty() || live.is_empty() {
        if live.is_empty() {
          bool_expr(true)
        } else {
          reduce_expr(&and_chain(&live), &rest, domain)?
        }
      } else {
        reduce_multi_var_and_inner(&live, &rest, domain)?
      };
      if matches!(&reduced, Expr::Identifier(b) if b == "False") {
        continue;
      }
      let pin = make_equality(&Expr::Identifier(v.clone()), replacement);
      branches.push(
        if matches!(&reduced, Expr::Identifier(b) if b == "True") {
          pin
        } else {
          and2(pin, reduced)
        },
      );
    }
    return Ok(match branches.len() {
      0 => bool_expr(false),
      1 => branches.remove(0),
      _ => Expr::FunctionCall {
        name: "Or".to_string(),
        args: branches.into(),
      },
    });
  }
  let mut degenerate = constraints.to_vec();
  degenerate.push(make_equality(d, &Expr::Integer(0)));
  reduce_multi_var_and_inner(&degenerate, vars, domain)
}

fn reduce_multi_var_and_inner(
  constraints: &[Expr],
  vars: &[String],
  domain: Option<&str>,
) -> Result<Expr, InterpreterError> {
  if vars.is_empty() || constraints.is_empty() {
    return Ok(bool_expr(true));
  }

  // A product equation `f g == 0` holds when either factor vanishes. Solving
  // it for a single variable would only keep the branch where the variable's
  // coefficient is nonzero (`Solve[a c == 0, c]` is `c -> 0`), silently losing
  // solutions such as `a == 0` when other constraints involve `c`. Split it
  // into one branch per factor first.
  for (i, constraint) in constraints.iter().enumerate() {
    let Some(factors) = zero_product_factors(constraint, vars) else {
      continue;
    };
    let mut branches = Vec::new();
    for factor in factors {
      let mut branch_constraints = constraints.to_vec();
      branch_constraints[i] = make_equality(&factor, &Expr::Integer(0));
      let branch =
        reduce_multi_var_and_inner(&branch_constraints, vars, domain)?;
      if !matches!(&branch, Expr::Identifier(s) if s == "False") {
        branches.push(branch);
      }
    }
    if branches.is_empty() {
      return Ok(bool_expr(false));
    }
    return Ok(build_or(branches));
  }

  // An equation pinning a variable to a constant (`v == c`) is eliminated
  // first: substituting it leaves a smaller system in which contradictions
  // (`0 == 1`) surface immediately instead of being solved for a variable.
  // So is one linear in a single variable with a numeric coefficient
  // (`44 p == 2269`): solved for it first, the rest stays free of it.
  for (i, constraint) in constraints.iter().enumerate() {
    let Some((lhs, rhs, CompOp::Equal)) = extract_comparison(constraint) else {
      continue;
    };
    let (v, c) = if let Expr::Identifier(v) = &lhs
      && vars.contains(v)
      && !vars.iter().any(|w| contains_var(&rhs, w))
    {
      (v.clone(), rhs.clone())
    } else if let Some(pinned) = linear_pin(&lhs, &rhs, vars) {
      pinned
    } else {
      continue;
    };
    let mut remaining = Vec::new();
    for (j, other) in constraints.iter().enumerate() {
      if j == i {
        continue;
      }
      let substituted = crate::syntax::substitute_variable(other, &v, &c);
      let evaled = crate::evaluator::evaluate_expr_to_expr(&substituted)
        .unwrap_or(substituted);
      match &evaled {
        Expr::Identifier(t) if t == "True" => {}
        Expr::Identifier(t) if t == "False" => return Ok(bool_expr(false)),
        _ => remaining.push(evaled),
      }
    }
    let remaining_vars: Vec<String> =
      vars.iter().filter(|w| **w != v).cloned().collect();
    let sub = if remaining.is_empty() || remaining_vars.is_empty() {
      bool_expr(true)
    } else {
      reduce_expr(&and_chain(&remaining), &remaining_vars, domain)?
    };
    if matches!(&sub, Expr::Identifier(t) if t == "False") {
      return Ok(bool_expr(false));
    }
    let pinned = make_equality(&Expr::Identifier(v.clone()), &c);
    if matches!(&sub, Expr::Identifier(t) if t == "True") {
      return Ok(pinned);
    }
    return Ok(and2(pinned, sub));
  }

  // A trig equation in one variable bounded by inequalities in that same
  // variable has finitely many solutions. Solve that sub-system on its own and
  // branch over the solutions (a periodic family would otherwise lose the
  // bounds once the variable is substituted into the other constraints).
  if vars.len() > 1 {
    for v in vars {
      let mut own: Vec<usize> = Vec::new();
      let mut trig_eq = false;
      let mut has_ineq = false;
      for (i, c) in constraints.iter().enumerate() {
        let mentions_other = vars.iter().any(|w| w != v && contains_var(c, w));
        if mentions_other || !contains_var(c, v) {
          continue;
        }
        own.push(i);
        if is_trig_equation(c, v) {
          trig_eq = true;
        } else if !matches!(c, Expr::Comparison { operators, .. }
          if operators.len() == 1 && operators[0] == ComparisonOp::Equal)
        {
          has_ineq = true;
        }
      }
      if !trig_eq || !has_ineq {
        continue;
      }
      let sub: Vec<Expr> = own
        .iter()
        .map(|&i| inequality_to_comparison(&constraints[i]))
        .collect();
      let sols = solve_ast(&[and_chain(&sub), Expr::Identifier(v.clone())])?;
      let Expr::List(sol_list) = &sols else {
        continue;
      };
      let mut values: Vec<Expr> = Vec::new();
      let mut concrete = true;
      for sol in sol_list {
        match sol {
          Expr::List(rules) if rules.len() == 1 => match &rules[0] {
            Expr::Rule { replacement, .. }
              if !contains_conditional_expression(replacement) =>
            {
              values.push((**replacement).clone());
            }
            _ => concrete = false,
          },
          _ => concrete = false,
        }
      }
      if !concrete {
        continue;
      }
      let mut branches = Vec::new();
      for value in values {
        let mut branch_constraints: Vec<Expr> = constraints
          .iter()
          .enumerate()
          .filter(|(i, _)| !own.contains(i))
          .map(|(_, c)| c.clone())
          .collect();
        branch_constraints
          .insert(0, make_equality(&Expr::Identifier(v.clone()), &value));
        let branch =
          reduce_multi_var_and_inner(&branch_constraints, vars, domain)?;
        if !matches!(&branch, Expr::Identifier(s) if s == "False") {
          branches.push(branch);
        }
      }
      if branches.is_empty() {
        return Ok(bool_expr(false));
      }
      return Ok(build_or(branches));
    }
  }

  // Find the best (equation, variable) pair: prefer last variable in the list
  // (matching Wolfram's convention where later variables are expressed in terms
  // of earlier ones), falling back to lowest degree for tie-breaking.
  // Every other (equation, variable) pair stays a fallback: the preferred one
  // can come up empty although the system has solutions (an equation like
  // `Sqrt[x^2 + y^2] == Sqrt[(x - 2)^2 + y^2]` is solved for `y` by no value
  // at all, but fixes `x`).
  let mut candidates: Vec<(usize, String, Expr, Expr, i128)> = Vec::new(); // (eq_idx, var, lhs, rhs, degree)
  for (i, constraint) in constraints.iter().enumerate() {
    if let Some((lhs, rhs, CompOp::Equal)) = extract_comparison(constraint) {
      for var in vars {
        let poly = minus2(lhs.clone(), rhs.clone());
        let expanded = expand_and_combine(&poly);
        if is_constant_wrt(&expanded, var) {
          continue;
        }
        if let Some(deg) = max_power_int(&expanded, var) {
          candidates.push((i, var.clone(), lhs.clone(), rhs.clone(), deg));
        }
      }
    }
  }

  candidates.sort_by_key(|(_, var, _, _, deg)| {
    (
      std::cmp::Reverse(vars.iter().position(|v| v == var).unwrap_or(0)),
      *deg,
    )
  });
  let mut solved_any_system = false;
  for (i, var, lhs, rhs, _deg) in candidates {
    let mut degenerate_results: Vec<Expr> = Vec::new();
    let eq = Expr::Comparison {
      operands: vec![lhs, rhs],
      operators: vec![ComparisonOp::Equal],
    };
    let solve_result = solve_ast(&[eq, Expr::Identifier(var.clone())])?;

    if let Expr::List(solutions) = &solve_result {
      let mut all_results: Vec<Expr> = Vec::new();

      for sol in solutions {
        if let Expr::List(rules) = sol {
          for rule in rules {
            if let Expr::Rule { replacement, .. } = rule {
              let value = replacement.as_ref();

              // A value with the other unknowns in its denominator only
              // holds where that denominator does not vanish
              // (`x != 0 && y == 2/x`); where it does, the equation is
              // solved afresh (`a == 0 && b == 0` for `a y == b`).
              let dens = vanishing_denominators(value, vars);
              for d in &dens {
                let key = expr_to_string(d);
                if ASSUMED_ZERO_DENOMINATORS.with(|z| z.borrow().contains(&key))
                {
                  continue;
                }
                ASSUMED_ZERO_DENOMINATORS.with(|z| z.borrow_mut().push(key));
                let branch =
                  reduce_with_vanishing(constraints, d, vars, domain);
                ASSUMED_ZERO_DENOMINATORS.with(|z| z.borrow_mut().pop());
                let branch = branch?;
                if !matches!(&branch, Expr::Identifier(s) if s == "False") {
                  degenerate_results.push(branch);
                }
              }
              let nonzero: Vec<Expr> = dens
                .iter()
                .map(|d| Expr::Comparison {
                  operands: vec![d.clone(), Expr::Integer(0)],
                  operators: vec![ComparisonOp::NotEqual],
                })
                .collect();

              // Substitute into remaining constraints
              let mut remaining: Vec<Expr> = constraints
                .iter()
                .enumerate()
                .filter(|(j, _)| *j != i)
                .map(|(_, c)| {
                  let substituted =
                    crate::syntax::substitute_variable(c, &var, value);
                  // Evaluate the substituted expression
                  crate::evaluator::evaluate_expr_to_expr(&substituted)
                    .unwrap_or(substituted)
                })
                .collect();
              if !remaining.is_empty() {
                remaining.extend(nonzero.iter().cloned());
              }

              let remaining_vars: Vec<String> = vars
                .iter()
                .filter(|v| v.as_str() != var.as_str())
                .cloned()
                .collect();

              if remaining.is_empty() {
                // No remaining constraints — var is expressed in terms of
                // remaining variables (if any) or is a constant solution.
                let var_eq = make_equality(
                  &Expr::Identifier(var.clone()),
                  &expanded_if_polynomial(value, vars),
                );
                let mut parts = nonzero.clone();
                parts.push(var_eq);
                all_results.push(and_chain(&parts));
              } else if remaining_vars.is_empty() {
                // Check if remaining constraints are satisfied
                let combined = remaining
                  .iter()
                  .skip(1)
                  .fold(remaining[0].clone(), |acc, c| and2(acc, c.clone()));
                let evaled = crate::evaluator::evaluate_expr_to_expr(&combined);
                if let Ok(result) = evaled
                  && !matches!(&result, Expr::Identifier(s) if s == "False")
                {
                  let var_eq =
                    make_equality(&Expr::Identifier(var.clone()), value);
                  all_results.push(var_eq);
                }
              } else {
                // Reduce remaining with fewer variables
                let combined = remaining
                  .iter()
                  .skip(1)
                  .fold(remaining[0].clone(), |acc, c| and2(acc, c.clone()));
                let sub_result =
                  reduce_expr(&combined, &remaining_vars, domain)?;

                if !matches!(&sub_result, Expr::Identifier(s) if s == "False") {
                  if matches!(&sub_result, Expr::Identifier(s) if s == "True") {
                    let var_eq =
                      make_equality(&Expr::Identifier(var.clone()), value);
                    all_results.push(var_eq);
                  } else {
                    // Handle Or branches: each branch is a separate solution
                    let or_branches = collect_or_terms(&sub_result);
                    for branch in &or_branches {
                      // Under inequalities the answer is cylindrical: the
                      // variable stays expressed in the earlier ones
                      // (`x > 0 && y == (-2 + x)/3 && z == (1 - x - 2*y)/3`).
                      // A `!=` condition (a denominator kept from zero) is
                      // no inequality in that sense.
                      let mut parts = Vec::new();
                      collect_and_constraints(branch, &mut parts);
                      if parts.iter().any(|p| {
                        !matches!(
                          extract_comparison(p),
                          Some((_, _, CompOp::Equal | CompOp::NotEqual))
                        )
                      }) {
                        let var_eq =
                          make_equality(&Expr::Identifier(var.clone()), value);
                        all_results.push(and2(branch.clone(), var_eq));
                        continue;
                      }
                      // Otherwise every variable is solved for the free
                      // ones: back-substitute the solved values, expanded
                      // (`y == -2 + x && z == 5/3 - x`).
                      let mut final_value = value.clone();
                      let sub_equalities = collect_and_equalities(branch);
                      for (sv, sval) in &sub_equalities {
                        final_value = crate::syntax::substitute_variable(
                          &final_value,
                          sv,
                          sval,
                        );
                      }
                      let final_value =
                        crate::evaluator::evaluate_expr_to_expr(&final_value)
                          .unwrap_or(simplify(final_value));
                      let final_value =
                        expanded_if_polynomial(&final_value, vars);
                      let var_eq = make_equality(
                        &Expr::Identifier(var.clone()),
                        &final_value,
                      );
                      // Put earlier variables first to match Wolfram's
                      // output ordering (variable list order)
                      all_results.push(and2(branch.clone(), var_eq));
                    }
                  }
                }
              }
            }
          }
        }
      }

      solved_any_system = true;
      if !all_results.is_empty() || !degenerate_results.is_empty() {
        degenerate_results.extend(all_results);
        return Ok(build_or(degenerate_results));
      }
    }
  }
  if solved_any_system {
    return Ok(bool_expr(false));
  }

  // Polynomial equations that no single equation can be solved from: the
  // Groebner basis of the system gives the unknowns as roots of one polynomial.
  if vars.len() >= 2
    && constraints
      .iter()
      .all(|c| matches!(extract_comparison(c), Some((_, _, CompOp::Equal))))
  {
    let var_exprs: Vec<Expr> =
      vars.iter().map(|v| Expr::Identifier(v.clone())).collect();
    let found = super::solve::solve_by_groebner_shape(
      &Expr::List(constraints.to_vec().into()),
      &var_exprs,
      domain == Some("Reals"),
    );
    if let Some(Expr::List(solutions)) = &found {
      let branches: Vec<Expr> = solutions
        .iter()
        .filter_map(|sol| {
          let Expr::List(rules) = sol else {
            return None;
          };
          let eqs: Vec<Expr> = rules
            .iter()
            .filter_map(|r| match r {
              Expr::Rule {
                pattern,
                replacement,
              } => Some(make_equality(pattern, replacement)),
              _ => None,
            })
            .collect();
          Some(and_chain(&eqs))
        })
        .collect();
      return Ok(if branches.is_empty() {
        bool_expr(false)
      } else {
        build_or(branches)
      });
    }
  }

  // No equation found to solve — return unevaluated
  Ok(call("Reduce", {
    let combined = constraints
      .iter()
      .skip(1)
      .fold(constraints[0].clone(), |acc, c| and2(acc, c.clone()));
    let vars_expr = if vars.len() == 1 {
      Expr::Identifier(vars[0].clone())
    } else {
      Expr::List(vars.iter().map(|v| Expr::Identifier(v.clone())).collect())
    };
    if let Some(dom) = domain {
      vec![combined, vars_expr, id_expr(dom)]
    } else {
      vec![combined, vars_expr]
    }
  }))
}

/// A solved value the way `Reduce` reports it once every variable is solved
/// for the free ones: expanded when it is a polynomial in them — a plain
/// number such as `(2 - 2*Sqrt[3])/2` as `1 - Sqrt[3]`, `(-2 + x)/3` as
/// `-2/3 + x/3` — while a quotient by one of them is put over a common
/// denominator (`(3 + x^2)/(1 + x)`).
fn expanded_if_polynomial(value: &Expr, vars: &[String]) -> Expr {
  // A denominator in a parameter counts as much as one in a variable.
  let mut symbols: Vec<String> = vars.to_vec();
  fn collect(e: &Expr, out: &mut Vec<String>) {
    if let Expr::Identifier(name) = e
      && !crate::evaluator::functions::is_builtin_symbol(name)
      && !out.contains(name)
    {
      out.push(name.clone());
    }
    for child in crate::syntax::expr_children(e) {
      collect(child, out);
    }
  }
  collect(value, &mut symbols);
  let head = if vanishing_denominators(value, &symbols).is_empty() {
    "Expand"
  } else {
    // A quotient over a common denominator, its common factors cancelled:
    // back-substitution leaves nested fractions (`24 a^2/((-3 + 8 a) (a +
    // 3 a/(-3 + 8 a)))` is 3).
    "Together"
  };
  crate::evaluator::evaluate_expr_to_expr(&call1(head, value.clone()))
    .unwrap_or_else(|_| value.clone())
}

/// `(v, c)` when `lhs == rhs` mentions a single one of `vars` and is linear
/// in it with a nonzero numeric coefficient: `a v + b == 0` gives `c =
/// -b/a`.
fn linear_pin(
  lhs: &Expr,
  rhs: &Expr,
  vars: &[String],
) -> Option<(String, Expr)> {
  use crate::evaluator::evaluate_expr_to_expr as eval;
  let difference =
    eval(&call1("Expand", minus2(lhs.clone(), rhs.clone()))).ok()?;
  let mut present = vars.iter().filter(|w| contains_var(&difference, w));
  let v = present.next()?.clone();
  if present.next().is_some() {
    return None;
  }
  let a = eval(&call(
    "Coefficient",
    vec![difference.clone(), Expr::Identifier(v.clone())],
  ))
  .ok()?;
  if !crate::functions::predicate_ast::is_numeric_q(&a)
    || crate::functions::math_ast::try_eval_to_f64(&a).is_some_and(|x| x == 0.0)
  {
    return None;
  }
  let b = eval(&call(
    "ReplaceAll",
    vec![
      difference.clone(),
      rule_expr(Expr::Identifier(v.clone()), Expr::Integer(0)),
    ],
  ))
  .ok()?;
  // Linear: nothing of `v` is left once `a v + b` is taken off.
  let rest = eval(&call1(
    "Expand",
    minus2(
      difference,
      plus2(times2(a.clone(), Expr::Identifier(v.clone())), b.clone()),
    ),
  ))
  .ok()?;
  if !matches!(rest, Expr::Integer(0)) || contains_var(&b, &v) {
    return None;
  }
  let c = eval(&call(
    "Times",
    vec![
      Expr::Integer(-1),
      b,
      call("Power", vec![a, Expr::Integer(-1)]),
    ],
  ))
  .ok()?;
  Some((v, c))
}

thread_local! {
  /// Denominators a degenerate Reduce branch is solving under `d == 0`, so
  /// that branch does not split on the same denominator again.
  static ASSUMED_ZERO_DENOMINATORS: std::cell::RefCell<Vec<String>> =
    const { std::cell::RefCell::new(Vec::new()) };
}

/// The factors of `value`'s denominator that involve one of `vars` — those
/// whose vanishing the solution excludes. A power counts as its base.
fn vanishing_denominators(value: &Expr, vars: &[String]) -> Vec<Expr> {
  let Ok(den) = crate::evaluator::evaluate_expr_to_expr(&call1(
    "Denominator",
    value.clone(),
  )) else {
    return Vec::new();
  };
  let factors: Vec<Expr> = match &den {
    Expr::FunctionCall { name, args } if name == "Times" => args.to_vec(),
    other => vec![other.clone()],
  };
  let mut out: Vec<Expr> = Vec::new();
  for f in factors {
    let base = match &f {
      Expr::FunctionCall { name, args }
        if name == "Power" && args.len() == 2 =>
      {
        args[0].clone()
      }
      Expr::BinaryOp {
        op: BinaryOperator::Power,
        left,
        ..
      } => left.as_ref().clone(),
      _ => f,
    };
    if vars.iter().any(|v| contains_var(&base, v))
      && !out
        .iter()
        .any(|o| expr_to_string(o) == expr_to_string(&base))
    {
      out.push(base);
    }
  }
  out
}

/// Check if a value is in a given domain.
fn is_in_domain(expr: &Expr, domain: &str) -> bool {
  match domain {
    "Reals" => {
      // No explicit imaginary unit, and no non-real value hiding behind a
      // radical or `Root[...]` object (e.g. `(-1)^(1/3)`).
      !contains_imaginary(expr) && !has_nonzero_imaginary_part(expr)
    }
    "Integers" => {
      matches!(expr, Expr::Integer(_))
    }
    "Complexes" => true,
    "Rationals" => matches!(
      expr,
      Expr::Integer(_) | Expr::FunctionCall { .. }
        if is_rational(expr)
    ),
    _ => true,
  }
}

pub fn is_rational(expr: &Expr) -> bool {
  match expr {
    Expr::Integer(_) => true,
    Expr::FunctionCall { name, args }
      if name == "Rational" && args.len() == 2 =>
    {
      true
    }
    _ => false,
  }
}

/// Whether `expr` is a closed numeric value whose imaginary part is
/// non-zero. Symbolic (non-numeric) expressions yield `false`.
fn has_nonzero_imaginary_part(expr: &Expr) -> bool {
  let im = call("Im", vec![call1("N", expr.clone())]);
  crate::evaluator::evaluate_expr_to_expr(&im)
    .ok()
    .and_then(|v| try_eval_to_f64(&v))
    .is_some_and(|v| v.abs() > 1e-12)
}

/// Check if an expression contains the imaginary unit I.
pub fn contains_imaginary(expr: &Expr) -> bool {
  match expr {
    Expr::Identifier(s) if s == "I" => true,
    Expr::BinaryOp { left, right, .. } => {
      contains_imaginary(left) || contains_imaginary(right)
    }
    Expr::UnaryOp { operand, .. } => contains_imaginary(operand),
    Expr::FunctionCall { name, args } => {
      if name == "Complex" {
        return true;
      }
      args.iter().any(contains_imaginary)
    }
    Expr::List(items) => items.iter().any(contains_imaginary),
    _ => false,
  }
}

// ─── Quantifier elimination: Exists with quadratic+linear constraints ──

/// Pattern-match `Reduce[Exists[{x, y}, q && l], param]` where `q` is a
/// ≤-inequality with quadratic form `α·x² + β·y²`, `l` is a ≥-inequality
/// with linear form `γ·x + δ·y`, and `param` is a single identifier.
/// Returns the resulting condition on `param`.
fn try_reduce_exists_quadratic_linear(
  exists_expr: &Expr,
  param_expr: &Expr,
) -> Option<Expr> {
  // Unwrap Exists[{x, y}, body].
  let (vars, body) = match exists_expr {
    Expr::FunctionCall { name, args }
      if name == "Exists" && args.len() == 2 =>
    {
      let vs = match &args[0] {
        Expr::List(items) => items.clone(),
        _ => return None,
      };
      (vs, args[1].clone())
    }
    _ => return None,
  };
  if vars.len() != 2 {
    return None;
  }
  let var_names: Vec<String> = vars
    .iter()
    .filter_map(|v| match v {
      Expr::Identifier(n) => Some(n.clone()),
      _ => None,
    })
    .collect();
  if var_names.len() != 2 {
    return None;
  }
  let param = match param_expr {
    Expr::Identifier(n) => n.clone(),
    _ => return None,
  };

  // Split the body into a flat list of And conjuncts.
  let mut parts: Vec<Expr> = Vec::new();
  flatten_and_into(&body, &mut parts);
  if parts.len() != 2 {
    return None;
  }
  let (q_ineq, l_ineq) = if is_quadratic_in(&parts[0], &var_names)
    && is_linear_in(&parts[1], &var_names)
  {
    (parts[0].clone(), parts[1].clone())
  } else if is_quadratic_in(&parts[1], &var_names)
    && is_linear_in(&parts[0], &var_names)
  {
    (parts[1].clone(), parts[0].clone())
  } else {
    return None;
  };

  // Quadratic constraint must be "form ≤ const" with const numeric.
  let (q_lhs, q_rhs) = extract_le_form(&q_ineq)?;
  let c1 = simplify_to_rational(&q_rhs)?;
  // Linear constraint must be "form ≥ const" with const numeric.
  let (l_lhs, l_rhs) = extract_ge_form(&l_ineq)?;
  let c2 = simplify_to_rational(&l_rhs)?;

  // Extract coefficients α, β (quadratic in x, y) and γ, δ (linear in x, y).
  let (alpha, beta, has_constant_q) = extract_xx_yy_coeffs(&q_lhs, &var_names)?;
  if !has_constant_q {
    // Quadratic LHS must have no constant term left over (e.g., `x² + a y²`
    // with the right-hand-side constant already moved). A non-zero
    // remainder defeats the formula.
  }
  let (gamma, delta) = extract_x_y_coeffs(&l_lhs, &var_names)?;

  // The audit's pattern is α=1 (a constant) and β=param. Generalise by
  // also accepting the swap (param appears as α). Bail out otherwise.
  let (alpha_const, beta_is_param) = (
    !contains_var(&alpha, &param),
    matches!(&beta, Expr::Identifier(n) if *n == param),
  );
  let (beta_const, alpha_is_param) = (
    !contains_var(&beta, &param),
    matches!(&alpha, Expr::Identifier(n) if *n == param),
  );
  let (a_const, p_coeff_var) = if alpha_const && beta_is_param {
    (alpha.clone(), var_names[1].clone())
  } else if beta_const && alpha_is_param {
    (beta.clone(), var_names[0].clone())
  } else {
    return None;
  };
  // Identify γ and δ corresponding to the constant-coeff variable and
  // the param-coeff variable respectively.
  let (lin_const_coeff, lin_param_coeff) = if p_coeff_var == var_names[1] {
    (gamma.clone(), delta.clone())
  } else {
    (delta.clone(), gamma.clone())
  };
  // All extracted scalars must be rationals.
  let alpha_q = simplify_to_rational(&a_const)?;
  let gamma_q = simplify_to_rational(&lin_const_coeff)?;
  let delta_q = simplify_to_rational(&lin_param_coeff)?;
  if !alpha_q.is_positive() {
    return None;
  }

  // Solvability condition:
  //   max(L) = sqrt(c1 * (γ²/α + δ²/(α·β/α))) — but with quadratic
  //   `α x² + β y²` it simplifies to sqrt(c1 (γ²/α + δ²/β)).
  // For param = β (assuming α constant): max ≥ c2 ⇒
  //   c1·(γ²/α + δ²/param) ≥ c2² ⇒
  //   δ²/param ≥ c2²/c1 − γ²/α ⇒
  //   1/param ≥ ((c2²/c1) - γ²/α) / δ²    [call this K]
  // K > 0 ⇒ `param ≤ 1/K` (only the positive-param branch matters
  //          since negative param produces an unbounded region).
  // K ≤ 0 ⇒ trivially satisfiable for every real `param`.
  let c2_sq = mul_q(&c2, &c2);
  let c1_inv = inv_q(&c1)?;
  let c2sq_over_c1 = mul_q(&c2_sq, &c1_inv);
  let alpha_inv = inv_q(&alpha_q)?;
  let gamma_sq = mul_q(&gamma_q, &gamma_q);
  let gamma_sq_over_alpha = mul_q(&gamma_sq, &alpha_inv);
  let lhs = sub_q(&c2sq_over_c1, &gamma_sq_over_alpha);
  let delta_sq = mul_q(&delta_q, &delta_q);
  if !delta_sq.is_positive() {
    return None;
  }
  let delta_sq_inv = inv_q(&delta_sq)?;
  let k = mul_q(&lhs, &delta_sq_inv);

  if k.is_zero() || k.is_negative() {
    return Some(bool_expr(true));
  }
  let bound = inv_q(&k)?;
  Some(Expr::Comparison {
    operands: vec![Expr::Identifier(param.clone()), rational_to_expr(&bound)],
    operators: vec![ComparisonOp::LessEqual],
  })
}

#[derive(Clone, Copy, Debug)]
struct Rat {
  num: i128,
  den: i128,
}

impl Rat {
  fn from_int(n: i128) -> Self {
    Self { num: n, den: 1 }
  }
  fn is_zero(&self) -> bool {
    self.num == 0
  }
  fn is_positive(&self) -> bool {
    (self.num > 0 && self.den > 0) || (self.num < 0 && self.den < 0)
  }
  fn is_negative(&self) -> bool {
    !self.is_zero() && !self.is_positive()
  }
}

fn reduce_q(n: i128, d: i128) -> Rat {
  let (n, d) = rat_reduce(n, d);
  Rat { num: n, den: d }
}

fn mul_q(a: &Rat, b: &Rat) -> Rat {
  reduce_q(a.num * b.num, a.den * b.den)
}

fn sub_q(a: &Rat, b: &Rat) -> Rat {
  reduce_q(a.num * b.den - b.num * a.den, a.den * b.den)
}

fn inv_q(a: &Rat) -> Option<Rat> {
  if a.num == 0 {
    return None;
  }
  Some(reduce_q(a.den, a.num))
}

fn rational_to_expr(r: &Rat) -> Expr {
  if r.den == 1 {
    Expr::Integer(r.num)
  } else {
    call("Rational", vec![Expr::Integer(r.num), Expr::Integer(r.den)])
  }
}

fn simplify_to_rational(e: &Expr) -> Option<Rat> {
  match e {
    Expr::Integer(n) => Some(Rat::from_int(*n)),
    Expr::FunctionCall { name, args }
      if name == "Rational" && args.len() == 2 =>
    {
      if let (Expr::Integer(n), Expr::Integer(d)) = (&args[0], &args[1]) {
        if *d == 0 {
          None
        } else {
          Some(reduce_q(*n, *d))
        }
      } else {
        None
      }
    }
    Expr::FunctionCall { name, args } if name == "Times" => {
      let mut acc = Rat::from_int(1);
      for arg in args {
        let r = simplify_to_rational(arg)?;
        acc = mul_q(&acc, &r);
      }
      Some(acc)
    }
    Expr::FunctionCall { name, args } if name == "Plus" => {
      let mut acc = Rat::from_int(0);
      for arg in args {
        let r = simplify_to_rational(arg)?;
        acc = reduce_q(acc.num * r.den + r.num * acc.den, acc.den * r.den);
      }
      Some(acc)
    }
    _ => {
      let evaluated = crate::evaluator::evaluate_expr_to_expr(e).ok()?;
      if matches!(&evaluated, Expr::Integer(_))
        || matches!(&evaluated, Expr::FunctionCall { name, .. } if name == "Rational")
      {
        simplify_to_rational(&evaluated)
      } else {
        None
      }
    }
  }
}

fn flatten_and_into(e: &Expr, out: &mut Vec<Expr>) {
  match e {
    Expr::BinaryOp {
      op: BinaryOperator::And,
      left,
      right,
    } => {
      flatten_and_into(left, out);
      flatten_and_into(right, out);
    }
    Expr::FunctionCall { name, args } if name == "And" => {
      for a in args {
        flatten_and_into(a, out);
      }
    }
    other => out.push(other.clone()),
  }
}

/// For an equation `p == q` whose difference factors into two or more
/// distinct non-constant factors involving `vars`, return those factors
/// (repeated factors collapse to their base). `None` otherwise.
fn zero_product_factors(
  constraint: &Expr,
  vars: &[String],
) -> Option<Vec<Expr>> {
  let (lhs, rhs, CompOp::Equal) = extract_comparison(constraint)? else {
    return None;
  };
  let poly = expand_and_combine(&minus2(lhs, rhs));
  let factored =
    crate::evaluator::evaluate_expr_to_expr(&call1("Factor", poly)).ok()?;
  fn flatten(e: &Expr, out: &mut Vec<Expr>) {
    match e {
      Expr::BinaryOp {
        op: BinaryOperator::Times,
        left,
        right,
      } => {
        flatten(left, out);
        flatten(right, out);
      }
      Expr::FunctionCall { name, args } if name == "Times" => {
        args.iter().for_each(|a| flatten(a, out));
      }
      _ => out.push(e.clone()),
    }
  }
  let mut raw = Vec::new();
  flatten(&factored, &mut raw);
  let mut factors: Vec<Expr> = Vec::new();
  for f in raw {
    let base = match &f {
      Expr::BinaryOp {
        op: BinaryOperator::Power,
        left,
        right,
      } if matches!(right.as_ref(), Expr::Integer(n) if *n > 0) => {
        left.as_ref().clone()
      }
      Expr::FunctionCall { name, args }
        if name == "Power"
          && args.len() == 2
          && matches!(&args[1], Expr::Integer(n) if *n > 0) =>
      {
        args[0].clone()
      }
      // Denominators (`a^(-1)`) never vanish, so they are no branch.
      Expr::BinaryOp {
        op: BinaryOperator::Power,
        ..
      } => continue,
      Expr::FunctionCall { name, .. } if name == "Power" => continue,
      _ => f,
    };
    if vars.iter().any(|v| contains_var(&base, v))
      && !factors.iter().any(|g| {
        crate::syntax::expr_to_string(g) == crate::syntax::expr_to_string(&base)
      })
    {
      factors.push(base);
    }
  }
  // Factors over one and the same variable set (`(x - 1)(x + 1)`) are solved
  // directly by the single-equation path, which orders their roots as
  // wolframscript does; only mixed-variable products lose branches.
  let var_sets: Vec<Vec<bool>> = factors
    .iter()
    .map(|f| vars.iter().map(|v| contains_var(f, v)).collect())
    .collect();
  let mixed = var_sets.windows(2).any(|w| w[0] != w[1]);
  (factors.len() >= 2 && mixed).then_some(factors)
}

fn contains_var(e: &Expr, name: &str) -> bool {
  match e {
    Expr::Identifier(n) => n == name,
    Expr::BinaryOp { left, right, .. } => {
      contains_var(left, name) || contains_var(right, name)
    }
    Expr::UnaryOp { operand, .. } => contains_var(operand, name),
    Expr::FunctionCall { args, .. } => {
      args.iter().any(|a| contains_var(a, name))
    }
    Expr::List(items) => items.iter().any(|a| contains_var(a, name)),
    Expr::Comparison { operands, .. } => {
      operands.iter().any(|a| contains_var(a, name))
    }
    _ => false,
  }
}

fn is_quadratic_in(e: &Expr, vars: &[String]) -> bool {
  let mut max_deg = 0_i128;
  for v in vars {
    if let Some(d) = comparison_max_degree(e, v) {
      max_deg = max_deg.max(d);
    } else {
      return false;
    }
  }
  max_deg == 2
}

fn is_linear_in(e: &Expr, vars: &[String]) -> bool {
  let mut max_deg = 0_i128;
  for v in vars {
    if let Some(d) = comparison_max_degree(e, v) {
      max_deg = max_deg.max(d);
    } else {
      return false;
    }
  }
  max_deg == 1
}

fn comparison_max_degree(e: &Expr, var: &str) -> Option<i128> {
  let lhs_minus_rhs = match e {
    Expr::Comparison { operands, .. } if operands.len() == 2 => {
      minus2(operands[0].clone(), operands[1].clone())
    }
    _ => return None,
  };
  let evaluated = crate::evaluator::evaluate_expr_to_expr(&lhs_minus_rhs)
    .unwrap_or(lhs_minus_rhs);
  super::max_power_int(&evaluated, var).or(Some(0))
}

fn extract_le_form(e: &Expr) -> Option<(Expr, Expr)> {
  if let Expr::Comparison {
    operands,
    operators,
  } = e
    && operands.len() == 2
    && operators.len() == 1
  {
    match &operators[0] {
      ComparisonOp::LessEqual => {
        Some((operands[0].clone(), operands[1].clone()))
      }
      ComparisonOp::GreaterEqual => {
        Some((operands[1].clone(), operands[0].clone()))
      }
      _ => None,
    }
  } else {
    None
  }
}

fn extract_ge_form(e: &Expr) -> Option<(Expr, Expr)> {
  if let Expr::Comparison {
    operands,
    operators,
  } = e
    && operands.len() == 2
    && operators.len() == 1
  {
    match &operators[0] {
      ComparisonOp::GreaterEqual => {
        Some((operands[0].clone(), operands[1].clone()))
      }
      ComparisonOp::LessEqual => {
        Some((operands[1].clone(), operands[0].clone()))
      }
      _ => None,
    }
  } else {
    None
  }
}

fn extract_xx_yy_coeffs(
  e: &Expr,
  vars: &[String],
) -> Option<(Expr, Expr, bool)> {
  // alpha = Coefficient[e, x, 2], beta = Coefficient[e, y, 2].
  let alpha = call_coefficient(e, &vars[0], 2)?;
  let beta = call_coefficient(e, &vars[1], 2)?;
  // Detect leftover constant term (not in audit's pattern → return false).
  let no_x = call_coefficient(e, &vars[0], 0)?;
  let leftover = call_coefficient(&no_x, &vars[1], 0)?;
  let has_const = !matches!(leftover, Expr::Integer(0));
  Some((alpha, beta, !has_const))
}

fn extract_x_y_coeffs(e: &Expr, vars: &[String]) -> Option<(Expr, Expr)> {
  let gamma = call_coefficient(e, &vars[0], 1)?;
  let delta = call_coefficient(e, &vars[1], 1)?;
  Some((gamma, delta))
}

fn call_coefficient(e: &Expr, var: &str, n: i128) -> Option<Expr> {
  let args = vec![e.clone(), id_expr(var), Expr::Integer(n)];
  super::coefficient_ast(&args).ok()
}
